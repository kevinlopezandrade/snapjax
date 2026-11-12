"""Public RTRL entry points and backwards-compatible helper exports."""

from typing import Callable

from jax import config
from jaxtyping import Array, Scalar

from snapjax.cells.base import RTRLStacked, Stacked, State
from snapjax.jacobians import (
    compute_masked_jacobian,
    dense_coo_product,
    sparse_matching_addition,
    update_cell_jacobians,
)
from snapjax.losses import l2
from snapjax.state import (
    make_init_state,
    make_perturbations,
    make_zeros_grads,
    make_zeros_jacobians,
    make_zeros_jacobians_sp,
)
from snapjax.rtrl.base import RTRLApprox, RTRLExact

config.update("jax_numpy_rank_promotion", "raise")


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
