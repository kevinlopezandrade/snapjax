from typing import Tuple

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

from rnn import RNN


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


def flatten(fun, in_tree):
    def flat(*args):
        theta = jtu.tree_unflatten(in_tree, args)
        return fun(theta)

    return flat


def new_vjp(fun, pytree_primal):
    primals, in_tree = jtu.tree_flatten(pytree_primal)
    flat_f = flatten(fun, in_tree)

    primals_vjp = []
    for i in range(len(primals)):
        f_partial, dyn_args = argnums_partial(
            wrap_init(flat_f),
            dyn_argnums=(i,),
            args=primals,
            require_static_args_hashable=False,
        )
        out, f_partial_vjp = _vjp(f_partial, *dyn_args)
        primals_vjp.append(f_partial_vjp)

    return jtu.tree_unflatten(in_tree, primals_vjp)


def projection_matrices(sparse_patterns: RNN):
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

        return projection_matrix

    res = jtu.tree_map(
        lambda node: _projection_matrix(node),
        sparse_patterns,
        is_leaf= lambda node: isinstance(node, jsparse.BCOO)
    )

    return res


theta = RNN(4, 4, use_bias=True, key=jax.random.PRNGKey(7))
h = jnp.ones(4)
x = jnp.ones(4)
f = jtu.Partial(RNN.f, h=h, x=x)


# Initial Overhead, for getting the sparse pattern.
jacobian_func = jax.jacrev(f)
res_orig = jacobian_func(theta)
sparse_patterns = jtu.tree_map(
    lambda node: jsparse.BCOO.fromdense(jnp.abs(node) > 0), res_orig
)

V = projection_matrices(sparse_patterns)

# Spare Jacobian Computation
theta_vjp = new_vjp(f, theta)

res = jtu.tree_map(
    lambda f, x: jax.vmap(f)(x),
    theta_vjp,
    V,
    is_leaf=lambda node: isinstance(node, jtu.Partial)
)

for leaf in jtu.tree_leaves(res):
    print(leaf)
