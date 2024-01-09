from typing import Sequence, Tuple

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.tree_util as jtu
import networkx
import numpy as onp
import scipy.sparse as ssparse
from jax._src.api import _vjp
from jax._src.api_util import argnums_partial
from jax.extend.linear_util import wrap_init
from jaxtyping import Array

class BCOOStructure(eqx.Module):
    indices: Array
    nse: Array
    shape: Sequence[int]


class SparseProjection(eqx.Module):
    projection_matrix: Array
    output_coloring: Array
    sparse_def: BCOOStructure

def _output_connectivity_from_sparsity(sparsity: ssparse.spmatrix) -> ssparse.spmatrix:
    """Computes the connectivity of output elements, given a Jacobian sparsity.

    Args:
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse connectivity matrix for the output elements.
    """
    assert sparsity.ndim == 2
    return (sparsity @ sparsity.T).astype(bool)


def _input_connectivity_from_sparsity(sparsity: ssparse.spmatrix) -> ssparse.spmatrix:
    """Computes the connectivity of input elements, given a Jacobian sparsity.

    Args:
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse connectivity matrix for the input elements.
    """
    assert sparsity.ndim == 2
    return (sparsity.T @ sparsity).astype(bool)


def _greedy_color(
    connectivity: ssparse.spmatrix,
    strategy: str,
) -> Tuple[onp.ndarray, int]:
    """Wraps `networkx.algorithms.coloring.greedy_color`.

    Args:
        connectivity: Sparse matrix giving the connectivity.
        strategy: The coloring strategy. See `networkx` documentation for details.

    Returns:
        A tuple containing the coloring vector and the number of colors used.
    """
    assert connectivity.ndim == 2
    assert connectivity.shape[0] == connectivity.shape[1]
    graph = networkx.convert_matrix.from_scipy_sparse_array(connectivity)
    coloring_dict = networkx.algorithms.coloring.greedy_color(graph, strategy)
    indices, colors = list(zip(*coloring_dict.items()))
    coloring = onp.asarray(colors)[onp.argsort(indices)]
    return coloring, onp.unique(coloring).size


def _expand_jacrev_jac(
    compressed_jac: jnp.ndarray,
    output_coloring: jnp.ndarray,
    sparsity: BCOOStructure,
) -> jsparse.BCOO:
    """Expands an output-compressed Jacobian into a sparse matrix.

    Args:
        compressed_jac: The compressed Jacobian.
        output_coloring: Coloring of the output elements.
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse Jacobian matrix.
    """
    assert compressed_jac.ndim == 2
    assert output_coloring.ndim == 1
    assert sparsity.shape == (output_coloring.size, compressed_jac.shape[1])
    row, col = sparsity.indices.T
    compressed_index = (output_coloring[row], col)
    data = compressed_jac[compressed_index]
    return jsparse.BCOO((data, sparsity.indices), shape=sparsity.shape)


def flatten_function(fun, in_tree):
    def flat(*args):
        theta = jtu.tree_unflatten(in_tree, args)
        return fun(theta)

    return flat


def tree_vjp(fun, primal_tree):
    primals, in_tree = jtu.tree_flatten(primal_tree)
    flat_f = flatten_function(fun, in_tree)

    N = len(primals)
    primals_vjp = [None] * N
    for i in range(N):
        f_partial, dyn_args = argnums_partial(
            wrap_init(flat_f),
            dyn_argnums=i,
            args=primals,
            require_static_args_hashable=False,
        )
        out, f_partial_vjp = _vjp(f_partial, *dyn_args)
        primals_vjp[i] = f_partial_vjp

    return jtu.tree_unflatten(in_tree, primals_vjp)




def sp_projection_matrices(sparse_patterns):
    def _projection_matrix(sparsity: jsparse.BCOO):
        sparsity_scipy = ssparse.coo_matrix(
            (sparsity.data, sparsity.indices.T), shape=sparsity.shape
        )
        connectivity = _output_connectivity_from_sparsity(sparsity_scipy)
        output_coloring, ncolors = _greedy_color(connectivity, "largest_first")
        output_coloring = jnp.asarray(output_coloring)
        assert output_coloring.size == sparsity.shape[0]

        projection_matrix = (
            jnp.arange(ncolors)[:, jnp.newaxis] == output_coloring[jnp.newaxis, :]
        )
        projection_matrix = projection_matrix.astype(jnp.float32)

        # We dont' need the data for the sparsity, only their structure.
        # HACK: Create an appropiate data structure for this.
        sparsity_structure = BCOOStructure(
            sparsity.indices, sparsity.nse, sparsity.shape
        )
        return SparseProjection(projection_matrix, output_coloring, sparsity_structure)

    res = jtu.tree_map(
        lambda node: _projection_matrix(node),
        sparse_patterns,
        is_leaf=lambda node: isinstance(node, jsparse.BCOO),
    )

    return res


def apply_sp_pullback(pullback, sp: SparseProjection):
    compressed_jacobian = jax.vmap(lambda ct: pullback(ct)[0])(sp.projection_matrix)
    compressed_jacobian = compressed_jacobian.reshape(compressed_jacobian.shape[0], -1)

    return _expand_jacrev_jac(compressed_jacobian, sp.output_coloring, sp.sparsity)


def sp_jacrev(fun, V):
    def _sp_jacfun(primal_tree):
        tree_pullback = tree_vjp(fun, primal_tree)

        tree_sp_jacobians = jtu.tree_map(
            apply_sp_pullback,
            tree_pullback,
            V,
            is_leaf=lambda node: isinstance(node, jtu.Partial),
        )

        return tree_sp_jacobians

    return _sp_jacfun
