from functools import partial
from typing import Callable, List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import config
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Scalar

from snapjax.cells.base import RTRLCell, RTRLLayer, RTRLStacked, State, is_rtrl_cell
from snapjax.losses import l2
from snapjax.sp_jacrev import SparseProjection
from snapjax.spp_primitives.primitives import spp_csr_matmul

config.update("jax_numpy_rank_promotion", "raise")


def make_zeros_jacobians(model: RTRLStacked):
    cells = eqx.filter(model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)

    def _cell_zero_jacobian(cell: RTRLCell):
        return [cell.make_zero_jacobians(cell) for _ in range(2)]

    zero_jacobians = jtu.tree_map(
        lambda cell: _cell_zero_jacobian(cell),
        cells,
        is_leaf=lambda leaf: is_rtrl_cell(leaf),
    )

    return zero_jacobians


def make_zeros_grads(model: RTRLStacked):
    def zeros_in_leaf(leaf):
        return jnp.zeros(shape=leaf.shape)

    zero_grads = jax.tree_map(zeros_in_leaf, model)

    return zero_grads


def make_perturbations(model: RTRLStacked):
    perturbations = []
    cell_index = 0
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = model.layers[cell_index].cell
            perturbations.append(jnp.zeros(cell.hidden_size))
            cell_index += 1

    return tuple(perturbations)


def update_cell_jacobians(
    I_t: RTRLCell,
    dynamics: Array,
    J_t_prev: RTRLCell,
):
    # RTRL
    J_t = jax.tree_map(
        lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
        I_t,
        J_t_prev,
    )
    return J_t


def update_jacobians_rtrl(
    jacobians_prev: RTRLStacked,
    inmediate_jacobians: List[Tuple[RTRLCell, Array]],
):
    # Jax will do loop unrolling here, but number of layers is not that big
    # so it will be fine.
    jacobians = jacobians_prev
    cell_index = 0
    for layer in jacobians_prev.layers:
        if isinstance(layer, RTRLLayer):
            I_t, D_t, _ = inmediate_jacobians[cell_index]
            J_t_prev = jacobians.layers[cell_index].cell
            J_t = update_cell_jacobians(I_t, D_t, J_t_prev[0])

            jacobians = eqx.tree_at(
                lambda model: model.layers[cell_index].cell[0], jacobians, J_t
            )

            cell_index += 1

    # Do an inverse pass to
    _, D_2, L_2 = inmediate_jacobians[1]
    J_2_prev = jacobians_prev.layers[0].cell[1]
    # update = D_2 @ J_2_prev
    # update = update + L_2 @ jacobians.layers[0].cell[0]
    update = jtu.tree_map(lambda mat: D_2 @ mat, J_2_prev)
    update = jtu.tree_map(lambda a, b: a + L_2 @ b, update, jacobians.layers[0].cell[0])

    jacobians = eqx.tree_at(lambda model: model.layers[0].cell[1], jacobians, update)

    return jacobians


def update_rtrl_cells_grads(
    grads: RTRLStacked,
    hidden_states_grads: List[Array],
    jacobians: RTRLStacked,
):
    cell_index = 0
    for layer in grads.layers[1:]:
        if isinstance(layer, RTRLLayer):
            ht_grad = hidden_states_grads[1]
            rtrl_cell_jac = jacobians.layers[1].cell[0]
            rtrl_cell_grads = jax.tree_map(
                lambda jacobian: ht_grad @ jacobian,
                rtrl_cell_jac,
            )
            grads = eqx.tree_at(
                lambda grads: grads.layers[1].cell,
                grads,
                rtrl_cell_grads,
                is_leaf=lambda x: x is None,
            )

            # cell_index += 1

    ht_grad = hidden_states_grads[1]
    rtrl_cell_jac = jacobians.layers[0].cell[1]
    rtrl_cell_grads = jtu.tree_map(lambda jacobian: ht_grad @ jacobian, rtrl_cell_jac)

    grads = eqx.tree_at(
        lambda grads: grads.layers[0].cell,
        grads,
        rtrl_cell_grads,
        is_leaf=lambda x: x is None,
    )

    return grads


def step_loss(
    model_spatial: RTRLStacked,
    model_rtrl: RTRLStacked,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    y_t: Array,
    loss_func: Callable[[Array, Array], Array] = l2,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, inmediate_jacobians, y_hat = model.f(h_prev, x_t, perturbations)

    res = loss_func(y_t, y_hat)

    return res, (h_t, y_hat, inmediate_jacobians)


# This should be a pure function.
def forward_rtrl(
    model: RTRLStacked,
    jacobians_prev: RTRLStacked,
    h_prev: Sequence[State],
    input: Array,
    target: Array,
    loss_func: Callable[[Array, Array], Array] = l2,
):
    theta_rtrl, theta_spatial = eqx.partition(
        model,
        lambda leaf: is_rtrl_cell(leaf),
        is_leaf=is_rtrl_cell,
    )
    step_loss_and_grad = jax.value_and_grad(
        jtu.Partial(step_loss, loss_func=loss_func), argnums=(0, 4), has_aux=True
    )
    perturbations = make_perturbations(theta_rtrl)

    (loss_t, aux), (grads) = step_loss_and_grad(
        theta_spatial,
        theta_rtrl,
        h_prev,
        input,
        perturbations,
        target,
    )

    h_t, y_hat, inmediate_jacobians = aux
    spatial_grads, hidden_states_grads = grads

    jacobians = update_jacobians_rtrl(jacobians_prev, inmediate_jacobians)

    grads = update_rtrl_cells_grads(spatial_grads, hidden_states_grads, jacobians)

    return h_t, grads, jacobians, loss_t, y_hat


def make_init_state(model: RTRLStacked):
    states: Sequence[State] = []
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = layer.cell
            states.append(type(cell).init_state(cell))

    return tuple(states)


def rtrl(
    model: RTRLStacked,
    inputs: Array,
    targets: Array,
    loss_func: Callable[[Array, Array], Scalar] = l2,
    use_scan: bool = True,
):
    count = 0
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            count += 1

    if count > 2:
        raise ValueError("Not supported for more than 2 RTRLLayer")

    def forward_repack(carry, data):
        input, target = data
        h_prev, acc_grads, jacobians_prev, acc_loss = carry

        out = forward_rtrl(
            model,
            jacobians_prev,
            h_prev,
            input,
            target,
            loss_func=loss_func,
        )
        h_t, grads, jacobians_t, loss_t, y_hat = out
        acc_loss = acc_loss + loss_t
        acc_grads = jtu.tree_map(
            lambda acc_grads, grads: acc_grads + grads, acc_grads, grads
        )

        return (h_t, acc_grads, jacobians_t, acc_loss), y_hat

    h_init: Sequence[State] = make_init_state(model)
    acc_grads: RTRLStacked = make_zeros_grads(model)
    zero_jacobians = make_zeros_jacobians(model)
    acc_loss = 0.0

    if use_scan:
        carry_T, _ = jax.lax.scan(
            forward_repack,
            init=(h_init, acc_grads, zero_jacobians, acc_loss),
            xs=(inputs, targets),
        )
    else:
        carry = (h_init, acc_grads, zero_jacobians, acc_loss)
        T = inputs.shape[0]
        y_hats = [None] * T
        for i in range(inputs.shape[0]):
            carry, y_hat = forward_repack(carry, (inputs[i], targets[i]))
            y_hats[i] = y_hat

        y_hats = jnp.stack(y_hats)
        carry_T = carry

    h_T, acc_grads, jacobians_T, acc_loss = carry_T

    return acc_loss, acc_grads
