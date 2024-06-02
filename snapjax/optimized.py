from typing import Any, Callable, Tuple, cast

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array

from snapjax.algos import step_loss, update_jacobians_rtrl, update_rtrl_cells_grads
from snapjax.cells.base import RTRLStacked, Stacked, State, is_rtrl_cell
from snapjax.losses import l2


def make_rtrl_step_loss_and_grad(
    loss_func: Callable[[Array, Array, float], Array] = l2
):
    return jax.jit(
        jax.value_and_grad(jtu.Partial(step_loss, loss_func=loss_func), has_aux=True)
    )


# This should be a pure function.
def forward_rtrl(
    model: RTRLStacked,
    jacobians_prev: RTRLStacked,
    h_prev: Stacked[State],
    input: Array,
    target: Array,
    mask: float,
    perturbations: Stacked[Array],
    jacobian_mask: RTRLStacked | None = None,
    jacobian_projection: RTRLStacked | None = None,
    step_loss_and_grad: Any = None,
):
    theta_rtrl, theta_spatial = eqx.partition(
        model,
        lambda leaf: is_rtrl_cell(leaf),
        is_leaf=is_rtrl_cell,
    )

    (loss_t, aux), (grads) = step_loss_and_grad(
        (theta_spatial, perturbations),
        theta_rtrl,
        h_prev,
        input,
        target,
        mask,
        jacobian_projection=jacobian_projection,
    )

    h_t, y_hat, inmediate_jacobians = aux
    spatial_grads, hidden_states_grads = grads

    jacobians = update_jacobians_rtrl(
        jacobians_prev,
        inmediate_jacobians,
        theta_rtrl,
        jacobian_mask=jacobian_mask,
    )

    grads = update_rtrl_cells_grads(
        spatial_grads, hidden_states_grads, jacobians, theta_rtrl
    )

    grads = jtu.tree_map(
        lambda grad, mat: grad if isinstance(mat, BCOO) else grad.reshape(mat.shape),
        grads,
        model,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    h_t = cast(Stacked[State], h_t)
    grads = cast(RTRLStacked, grads)
    jacobians = cast(RTRLStacked, jacobians)
    loss_t = cast(float, loss_t)
    y_hat = cast(Array, y_hat)

    return h_t, grads, jacobians, loss_t, y_hat
