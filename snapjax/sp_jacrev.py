from typing import Any, Callable, Sequence, Tuple, TypeVar

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.tree_util as jtu
import networkx
import numpy as onp
import scipy.sparse as ssparse
from jax._src.api import _std_basis, _vjp
from jax._src.api_util import argnums_partial
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.ad import flatten_fun_for_sparse_ad
from jax.extend.linear_util import wrap_init
from jaxtyping import Array, PyTree


def standard_jacobian(jacobian: Array) -> Array:
    return jacobian.reshape(jacobian.shape[0], -1)


_T = TypeVar("_T")
is_sparse = lambda x: isinstance(x, JAXSparse)


class BCOOStructure(eqx.Module):
    indices_csr: Array
    indices_csc: Array
    nse: int = eqx.field(static=True)
    jacobian_shape: Sequence[int] = eqx.field(static=True)


class SparseProjection(eqx.Module):
    projection_matrix: Array
    output_coloring: Array
    sparse_def: BCOOStructure


class DenseProjection(eqx.Module):
    projection_matrix: Array
    jacobian_shape: Sequence[int] = eqx.field(static=True)


# Wrapper just to differentiate between BCOO and sparse patterns.
class SparseMask(eqx.Module):
    indices: Array
    jacobian_shape: Sequence[int] = eqx.field(static=True)
    orig_jacobian_shape: Sequence[int] | None = eqx.field(static=True)
    # NOTE: orig_jacobian_shape will be deprecated since we use standard jacobians
    # now.

    def __init__(
        self,
        indices: Array,
        shape: Sequence[int],
        orig_shape: Sequence[int] | None = None,
    ):
        self.indices = indices
        self.jacobian_shape = shape
        self.orig_jacobian_shape = orig_shape


class Mask(eqx.Module):
    jacobian_mask: Array


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
    transpose: bool = False,
) -> jsparse.BCOO:
    """Expands an output-compressed Jacobian into a sparse matrix.

    Args:
        compressed_jac: The compressed Jacobian.
        output_coloring: Coloring of the output elements.
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse Jacobian matrix TRANSPOSED.
    """
    assert compressed_jac.ndim == 2
    assert output_coloring.ndim == 1
    assert sparsity.jacobian_shape == (output_coloring.size, compressed_jac.shape[1])
    if transpose:
        indices = sparsity.indices_csc
        shape = sparsity.jacobian_shape[::-1]
    else:
        indices = sparsity.indices_csr
        shape = sparsity.jacobian_shape

    row, col = indices.T
    compressed_index = (output_coloring[row], col)
    data = compressed_jac[compressed_index]

    # Indices are choosen from the csc ordering
    # which is not sorted by the row indices, rather
    # by the column indices.
    if transpose:
        indices = sparsity.indices_csc[:, ::-1]

    return jsparse.BCOO(
        (data, indices),
        shape=shape,
        indices_sorted=True,
        unique_indices=True,
    )


def flatten_function(fun, in_tree):
    """
    Given a function 'fun' which takes a PyTree creates
    another function that will compute 'fun' but on
    a flattened pytree. The jax.core contains a similar
    utility, but much more involved for their purposes.
    """

    def flat(*args):
        theta = jtu.tree_unflatten(in_tree, args)
        return fun(*theta)

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
    primals, in_tree = jtu.tree_flatten(primal_tree, is_leaf=is_sparse)
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

        if is_sparse(primals[i]):
            # Here I must flatten again since its a sparse array.
            f_to_reflaten = f_partial.call_wrapped
            fun_reflat, argnums_flat, args_flat, postprocess_gradients = (
                flatten_fun_for_sparse_ad(f_to_reflaten, 0, dyn_args)
            )
            # Repartial again for the vjp
            f_partial, dyn_args = argnums_partial(
                wrap_init(fun_reflat),
                dyn_argnums=0,
                args=args_flat,
                require_static_args_hashable=False,
            )

        out, f_partial_vjp = _vjp(
            f_partial, *dyn_args
        )  # dyn_args must only contain just one arg
        primals_vjp[i] = f_partial_vjp

    return jtu.tree_unflatten(in_tree, primals_vjp)


class _FlagSpJac:
    pass


def new_tree_vjp(
    fun, primal_tree: PyTree, V: PyTree, transpose: bool = False
) -> PyTree:
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
    primals, in_tree = jtu.tree_flatten(primal_tree, is_leaf=is_sparse)
    flat_f = flatten_function(fun, in_tree)

    _V, _ = jtu.tree_flatten(
        V,
        is_leaf=lambda x: isinstance(x, _FlagSpJac)
        or isinstance(x, (SparseProjection, DenseProjection)),
    )

    N = len(primals)

    assert len(_V) == N

    primals_vjp = [None] * N
    for i in range(N):
        f_partial, dyn_args = argnums_partial(
            wrap_init(flat_f),
            dyn_argnums=i,
            args=primals,
            require_static_args_hashable=False,
        )

        if is_sparse(primals[i]):
            # Here I must flatten again since its a sparse array.
            f_to_reflaten = f_partial.call_wrapped
            fun_reflat, argnums_flat, args_flat, postprocess_gradients = (
                flatten_fun_for_sparse_ad(f_to_reflaten, 0, dyn_args)
            )
            # Repartial again for the vjp
            f_partial, dyn_args = argnums_partial(
                wrap_init(fun_reflat),
                dyn_argnums=0,
                args=args_flat,
                require_static_args_hashable=False,
            )

        out, f_partial_vjp = _vjp(
            f_partial, *dyn_args
        )  # dyn_args must only contain just one arg
        if not isinstance(_V[i], _FlagSpJac):
            primals_vjp[i] = apply_sp_pullback(
                f_partial_vjp, _V[i], transpose=transpose
            )
        else:
            primals_vjp[i] = jax.vmap(f_partial_vjp)(_std_basis(out))[0]

    return jtu.tree_unflatten(in_tree, primals_vjp)


def make_jacobian_projection(sparse_patterns: _T) -> _T:
    """
    Given a PyTree with sparse_patterns in BCOO format, it computes per leaf
    the structural orthogonal rows using a graph coloring algorithm, and the
    projection matrix used for computing sparse jacobians. It then
    returns a PyTree with the same structure where each leaf contains
    a SparseProjection needed for the computation of sparse jacobians.
    """

    def _projection_matrix(sparsity: SparseMask):
        sparsity_scipy = ssparse.coo_matrix(
            (jnp.ones(sparsity.indices.shape[0]), sparsity.indices.T),
            shape=sparsity.jacobian_shape,
        )

        connectivity = _output_connectivity_from_sparsity(sparsity_scipy)
        output_coloring, ncolors = _greedy_color(connectivity, "DSATUR")
        output_coloring = jnp.asarray(output_coloring)
        assert output_coloring.size == sparsity.jacobian_shape[0]

        projection_matrix = (
            jnp.arange(ncolors)[:, jnp.newaxis] == output_coloring[jnp.newaxis, :]
        )

        # HACK: Promote either to float32 or float64
        projection_matrix = projection_matrix * 1.0

        # We dont' need the data for the sparsity, only their structure.
        # HACK: Create an appropiate data structure for this.
        indices_csr = sparsity.indices
        indices_csc = indices_csr[jnp.lexsort(indices_csr.T)]
        sparsity_structure = BCOOStructure(
            indices_csr,
            indices_csc,
            sparsity_scipy.nnz,
            sparsity.jacobian_shape,
        )
        return SparseProjection(projection_matrix, output_coloring, sparsity_structure)

    def _construct_projection(node: SparseMask | Mask | Any):
        if isinstance(node, SparseMask):
            return _projection_matrix(node)
        elif isinstance(node, Mask):
            return DenseProjection(
                jnp.eye(node.jacobian_mask.shape[0]),
                jacobian_shape=node.jacobian_mask.shape,
            )
        else:
            raise ValueError("Leaf must be a SparseMask, Mask or a BCOO")

    res = jtu.tree_map(
        lambda node: _construct_projection(node),
        sparse_patterns,
        is_leaf=lambda node: isinstance(node, (SparseMask, Mask)),
    )

    return res


def apply_sp_pullback(
    pullback, sp: SparseProjection | DenseProjection, transpose: bool = False
):
    """
    Computes the sparse jacobian using the projection matrix using
    the pullback (jvp).

    Args:
        pullback: vjp function to use.
        sp: Projection matrix to vmap over.
    """
    if isinstance(sp, DenseProjection):
        return standard_jacobian(jax.vmap(pullback)(sp.projection_matrix)[0])

    # NOTE: When you evaluate pullback to a cotangent vector even if the
    # jacobian jax gives if you use jax.jacobian is of 3 or more dimensions
    # what is actually doing in pullback is the v @ jacobian.reshape(output_dim, -1)
    # as it should be doing, but then it gives you a reshaped version.
    compressed_jacobian = jax.vmap(pullback)(sp.projection_matrix)[0]
    # Reshape is needed since the sparse patterns are only 2d dimensional. But the jvp
    # returns 3d arrays.
    compressed_jacobian = standard_jacobian(compressed_jacobian)

    return _expand_jacrev_jac(
        compressed_jacobian, sp.output_coloring, sp.sparse_def, transpose
    )


def old_sp_jacrev(
    fun: Callable[[_T], Array], V: _T, transpose: bool = False
) -> Callable[[_T], Array]:
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
    apply_pullback = jtu.Partial(apply_sp_pullback, transpose=transpose)

    def _sp_jacfun(primal_tree):
        tree_pullback = tree_vjp(fun, primal_tree)

        tree_sp_jacobians = jtu.tree_map(
            apply_pullback,
            tree_pullback,
            V,
            is_leaf=lambda node: isinstance(node, jtu.Partial),
        )

        return tree_sp_jacobians

    return _sp_jacfun


def sp_jacrev(
    fun: Callable[[_T], Array],
    V: _T,
    transpose: bool = False,
    argnums: int | Sequence[int] = 0,
) -> Callable[[_T], Array]:
    """
    Retruns a function that will compute the jacobian of fun w.r.t
    to its first positional argument, making use of the sparsity
    structure of V. Extra arguments can be specified but the V will only
    be applied to the first one.

    Args:
        fun: Function to use for computing the sparse jacobian.
        V: PyTree with the same structure as the input pytree
            of fun. Each leaf of the PyTree must be a SparseProjection
            leaf.
    """

    def _sp_jacfun(*args):
        f_partial, dyn_args = argnums_partial(
            wrap_init(fun), argnums, args, require_static_args_hashable=False
        )

        # NOTE: V, when we flatten again might introduce extra None's
        # we cannot control this since it depends in the PyTree of the
        # users. Therefore to tag the extra dyna arguments we use a flag.
        _V = (V, *(_FlagSpJac() for _ in range(1, len(dyn_args))))

        tree_jacobians = new_tree_vjp(
            f_partial.call_wrapped, dyn_args, _V, transpose=transpose
        )

        if len(_V) == 1:
            return tree_jacobians[0]
        else:
            return tree_jacobians

    return _sp_jacfun
