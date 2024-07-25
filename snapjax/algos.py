from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import config
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Scalar

from snapjax.cells.base import (
    RTRLCell,
    RTRLLayer,
    RTRLStacked,
    Stacked,
    State,
    is_rtrl_cell,
)
from snapjax.losses import l2
from snapjax.sp_jacrev import (
    DenseProjection,
    Mask,
    SparseMask,
    SparseProjection,
    standard_jacobian,
)
from snapjax.spp_primitives.primitives import spp_csr_matmul

config.update("jax_numpy_rank_promotion", "raise")


@jax.jit
def make_zeros_jacobians_sp(jacobian_projection: RTRLStacked):
    # Jacobians are saved as the tranpose jacobians.
    def _sparse_jacobian(leaf: SparseProjection | DenseProjection):
        if isinstance(leaf, DenseProjection):
            return standard_jacobian(jnp.zeros(leaf.jacobian_shape))

        zeros = jnp.zeros(leaf.sparse_def.nse)
        indices = leaf.sparse_def.indices_csc[:, ::-1]  # For the transpose.
        structure = (zeros, indices)
        return BCOO(
            structure,
            shape=leaf.sparse_def.jacobian_shape[::-1],  # For the transpose.
            # This is guaranteed since csc is sorted by colum and we transpose.
            indices_sorted=True,
            unique_indices=True,
        )

    zero_jacobians = jtu.tree_map(
        lambda leaf: _sparse_jacobian(leaf),
        jacobian_projection,
        is_leaf=lambda node: isinstance(node, (SparseProjection, DenseProjection)),
    )

    return zero_jacobians


@jax.jit
def make_zeros_jacobians(model: RTRLStacked):
    cells = eqx.filter(model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)

    def _cell_zero_jacobian(cell: RTRLCell):
        if cell.custom_trace_update:
            return cell.make_zero_traces()
        else:
            return jtu.tree_map(
                lambda leaf: standard_jacobian(leaf), cell.make_zero_jacobians()
            )

    zero_jacobians = jtu.tree_map(
        lambda cell: _cell_zero_jacobian(cell),
        cells,
        is_leaf=lambda leaf: is_rtrl_cell(leaf),
    )

    return zero_jacobians


@jax.jit
def make_zeros_grads(model: RTRLStacked):
    def zeros_in_leaf(leaf):
        if isinstance(leaf, BCOO):
            return jnp.zeros(leaf.nse)
        elif eqx.is_array(leaf):
            return jnp.zeros(shape=leaf.shape)

    zero_grads = jtu.tree_map(
        zeros_in_leaf, model, is_leaf=lambda node: isinstance(node, BCOO)
    )

    return zero_grads


def make_perturbations(model: RTRLStacked):
    perturbations = []
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = layer.cell
            zeros = cell.init_perturbation()
            perturbations.append(zeros)

    return tuple(perturbations)


@jax.jit
def dense_coo_product(D: Array, J_T: BCOO, sp_T: Array):
    # TO CSR
    J_T = BCSR.from_bcoo(J_T)
    D_T = D.T

    data = spp_csr_matmul(J_T.data, J_T.indices, J_T.indptr, D_T, sp_T)

    return BCOO((data, sp_T), shape=J_T.shape, indices_sorted=True, unique_indices=True)


@jax.jit
def sparse_matching_addition(A: BCOO, B: BCOO) -> BCOO:
    # Assumes A and B have the same sparsity
    # pattern and that their indices are ordered.
    # then addition is just adding the A.data + B.data
    res = A.data + B.data

    return BCOO(
        (res, A.indices), indices_sorted=True, unique_indices=True, shape=A.shape
    )


@jax.jit
def compute_masked_jacobian(
    jacobian_mask: RTRLCell, I_t: RTRLCell, dynamics: Array, J_t_prev: RTRLCell
):
    """
    Note that I_t and J_t_prev must be actually the transposed ones if they are in BCOO.
    Returns the I + D*J(t_1) tranposed if in BCOO.
    """

    def _update(i_t: BCOO | Array, j_t_prev: BCOO | Array, j_mask: Mask | SparseMask):
        if isinstance(j_mask, SparseMask):
            # j_t_prev comes as the transpose jacobian.
            # NOTE: Actually we can ignore the j_mask, since
            # the compressed was computed using j_mask as its
            # sparsity pattern. And we avoid transposing and sorting.
            # We leave it just for consistency in the api.
            prod = dense_coo_product(dynamics, j_t_prev, j_t_prev.indices)
            res = sparse_matching_addition(i_t, prod)
            return res
        else:
            return standard_jacobian(j_mask.jacobian_mask) * (
                standard_jacobian(i_t) + dynamics @ standard_jacobian(j_t_prev)
            )

    J_t = jtu.tree_map(
        lambda i_t, j_t_prev, j_mask: _update(i_t, j_t_prev, j_mask),
        I_t,
        J_t_prev,
        jacobian_mask,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    return J_t


@jax.jit
def update_cell_jacobians(
    I_t: RTRLCell,
    dynamics: Array,
    J_t_prev: RTRLCell,
    jacobian_cell_mask: RTRLCell | None = None,
):
    # Apply masks to the jacobians.
    if jacobian_cell_mask:
        J_t = compute_masked_jacobian(jacobian_cell_mask, I_t, dynamics, J_t_prev)
        return J_t
    else:
        J_t = jtu.tree_map(
            lambda i_t, j_t_prev: standard_jacobian(i_t)
            + dynamics @ standard_jacobian(j_t_prev),
            I_t,
            J_t_prev,
        )
        return J_t


@jax.jit
def make_init_state(model: RTRLStacked) -> Stacked[State]:
    states: Sequence[State] = []
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = layer.cell
            states.append(cell.init_state())

    return tuple(states)


from snapjax.rtrl.base import RTRLApprox, RTRLExact


def forward_rtrl(
    model: RTRLStacked,
    jacobians_prev: RTRLStacked,
    h_prev: Stacked[State],
    input: Array,
    target: Array,
    mask: float = 1.0,
    jacobian_mask: RTRLStacked | None = None,
    jacobian_projection: RTRLStacked | None = None,
    loss_func: Callable[[Array, Array, Array], Scalar] | None = None,
):
    algo = RTRLApprox(loss_func=loss_func, use_scan=True)
    return algo.step(
        model,
        jacobians_prev,
        h_prev,
        input,
        target,
        mask,
        jacobian_mask,
        jacobian_projection,
    )


def rtrl(
    model: RTRLStacked,
    inputs: Array,
    targets: Array,
    mask: Array,
    jacobian_mask: RTRLStacked | None = None,
    jacobian_projection: RTRLStacked | None = None,
    loss_func: Callable[[Array, Array, float], Scalar] = l2,
    use_scan: bool = True,
    mean: bool = False,
):
    algo = RTRLApprox(loss_func=loss_func, use_scan=use_scan)
    return algo.rtrl(model, inputs, targets, mask, jacobian_mask, jacobian_projection)


def rtrl_exact(
    model: RTRLStacked,
    inputs: Array,
    targets: Array,
    mask: Array,
    jacobian_mask: RTRLStacked | None = None,
    jacobian_projection: RTRLStacked | None = None,
    loss_func: Callable[[Array, Array, float], Scalar] = l2,
    use_scan: bool = True,
    mean: bool = False,
):
    algo = RTRLExact(loss_func=loss_func, use_scan=use_scan)
    return algo.rtrl(model, inputs, targets, mask, jacobian_mask, jacobian_projection)
