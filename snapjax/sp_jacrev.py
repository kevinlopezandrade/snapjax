from typing import Callable, Sequence, Tuple

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
from jaxtyping import Array, PyTree


class BCOOStructure(eqx.Module):
    indices: Array
    nse: int = eqx.field(static=True)
    shape: Sequence[int] = eqx.field(static=True)


class SparseProjection(eqx.Module):
    projection_matrix: Array
    output_coloring: Array
    sparse_def: BCOOStructure


# TODO: The following functions: _output_connectivity_from_sparsity,
# _input_connectivity_from_sparsity, _greedy_color, _expand_jacrev_jac were
# taken from https://github.com/mfschubert/sparsejac. Ideally I should submit a
# pull request.


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
    """
    Given a function 'fun' which takes a PyTree creates
    another function that will compute 'fun' but on
    a flattened pytree. The jax.core contains a similar
    utility, but much more involved for their purposes.
    """

    def flat(*args):
        theta = jtu.tree_unflatten(in_tree, args)
        return fun(theta)

    return flat


def tree_vjp(fun, primal_tree: PyTree) -> PyTree:
    """
    Takes a function 'fun' which has only one positional
    argument, which is a PyTree, and returns a PyTree
    with the same structure that for each leafs contains
    the pullback 'vjp' evaluated at the corresponding leaf.

    Args:
        fun: Function that takes only one positional
            argument which is a PyTree.
        primal_tree: PyTree with the primal values for each
            vjp in the returned PyTree.
    """
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


def sp_projection_matrices(sparse_patterns: PyTree) -> PyTree:
    """
    Given a PyTree with sparse_patterns in BCOO format, it computes per leaf
    the structural orthogonal rows using a graph coloring algorithm, and the
    projection matrix used for computing sparse jacobians. It then
    returns a PyTree with the same structure where each leaf contains
    a SparseProjection needed for the computation of sparse jacobians.
    """

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
    """
    Computes the sparse jacobian using the projection matrix using
    the pullback (jvp).

    Args:
        pullback: vjp function to use.
        sp: Projection matrix to vmap over.
    """
    compressed_jacobian = jax.vmap(lambda ct: pullback(ct)[0])(sp.projection_matrix)
    # Reshape is needed since the sparse patterns are only 2d dimensional. But the jvp
    # returns 3d arrays.
    compressed_jacobian = compressed_jacobian.reshape(compressed_jacobian.shape[0], -1)

    return _expand_jacrev_jac(compressed_jacobian, sp.output_coloring, sp.sparse_def)


def sp_jacrev(fun: Callable[[PyTree], PyTree], V: PyTree) -> PyTree:
    """
    Retruns a function that will compute the jacobian of fun w.r.t
    to its first positional argument, making use of the sparsity
    structure of V.

    Args:
        fun: Function to use for computing the sparse jacobian.
        V: PyTree with the same structure as the input pytree
            of fun. Each leaf of the PyTree must be a SparseProjection
            leaf.
    """

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
