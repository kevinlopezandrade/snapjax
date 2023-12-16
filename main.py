from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import State, StateIndex, make_with_state
from jax import config
from jaxtyping import Array

from pta_cell import (
    Any,
    List,
    PTACell,
    PTALayerRTRL,
    StackedRTRL,
    zero_influence_pytree,
)

config.update("jax_numpy_rank_promotion", "raise")


def is_pta_cell_or_jacobian_index(node: Any):
    if isinstance(node, PTACell) or isinstance(node, StateIndex):
        return True
    else:
        return False


def make_zeros_jacobian(model: StackedRTRL):
    def zeros_in_leaf(leaf):
        if eqx.is_array(leaf):
            return jnp.zeros(shape=(model.hidden_size, *leaf.shape))
        else:
            return None

    zero_influences = jax.tree_map(zeros_in_leaf, model_rtrl)

    return zero_influences


def step_loss_test(
    model: StackedRTRL,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    jacobians_state: eqx.nn.State,
    y_t: Array,
):
    h_t, y_hat, new_state = model(h_prev, x_t, perturbations, jacobians_state)

    diff = (y_t - y_hat) ** 2

    return jnp.sum(diff), (h_t, new_state)


def step_loss(
    model_spatial: StackedRTRL,
    model_rtrl: StackedRTRL,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    jacobians_state: eqx.nn.State,
    y_t: Array,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, y_hat, new_state = model(h_prev, x_t, perturbations, jacobians_state)

    diff = (y_t - y_hat) ** 2

    return jnp.sum(diff), (h_t, new_state)


# This should be a pure function.
@eqx.filter_jit
def forward_rtrl(
    theta_spatial: StackedRTRL,
    theta_rtrl: StackedRTRL,
    h_prev: Array,
    input: Array,
    perturbations: Array,
    jacobians_prev: StackedRTRL,
    inmediate_jacobians_state: eqx.nn.State,
    target: Array,
):
    step_loss_and_grad = jax.value_and_grad(step_loss, argnums=(0, 4), has_aux=True)
    (loss, aux), (grads) = step_loss_and_grad(
        theta_spatial,
        theta_rtrl,
        h_prev,
        input,
        perturbations,
        inmediate_jacobians_state,
        target,
    )

    h_t, inmediate_jacobians_state = aux
    spatial_grads, hidden_states_grads = grads

    jacobians = jacobians_prev
    for i in range(model.num_layers):
        inmediate_jacobian, dynamics = inmediate_jacobians_state.get(
            theta_rtrl.layers[i].jacobian_index
        )

        J_t_prev = jacobians.layers[i].cell

        # RTRL
        J_t = jax.tree_map(
            lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
            inmediate_jacobian,
            J_t_prev,
        )

        jacobians = eqx.tree_at(lambda model: model.layers[i].cell, jacobians, J_t)

        # SNAP 1, just keep the non zero entries that appear
        # in the i_t.

    for i in range(model.num_layers):
        ht_grad = hidden_states_grads[i]
        rtrl_grads = jax.tree_map(
            lambda jacobian: ht_grad @ jacobian,
            jacobians.layers[i].cell,
        )

        spatial_grads = eqx.tree_at(
            lambda model: model.layers[i].cell,
            spatial_grads,
            rtrl_grads,
            is_leaf=lambda x: x is None,
        )

    return spatial_grads


num_layers = 10
hidden_size = 20
input_size = 20
key = jax.random.PRNGKey(7)

model, inmediate_jacobians_state = make_with_state(StackedRTRL)(
    key, num_layers=num_layers, hidden_size=hidden_size, input_size=input_size
)

model_rtrl, model_spatial = eqx.partition(
    model,
    lambda leaf: is_pta_cell_or_jacobian_index(leaf),
    is_leaf=is_pta_cell_or_jacobian_index,
)

assert eqx.tree_equal(
    eqx.combine(model_rtrl, model_spatial, is_leaf=is_pta_cell_or_jacobian_index), model
)

jacobian = make_zeros_jacobian(model_rtrl)
input = jnp.ones(shape=(input_size,))
output = jnp.zeros(shape=(input_size,))
h_prev = jnp.ones(shape=(num_layers, hidden_size))
perturbations = jnp.zeros(shape=(num_layers, hidden_size))
