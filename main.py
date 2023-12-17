import timeit
from functools import partial
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import State, StateIndex, make_with_state
from jax import config
from jaxtyping import Array

from rnn import RNN, StackedRNN

config.update("jax_numpy_rank_promotion", "raise")


def is_rnn_cell_or_jacobian_index(node: Any):
    if isinstance(node, RNN) or isinstance(node, StateIndex):
        return True
    else:
        return False


def make_zeros_jacobian(model: StackedRNN):
    def zeros_in_leaf(leaf):
        if eqx.is_array(leaf):
            return jnp.zeros(shape=(model.hidden_size, *leaf.shape))
        else:
            return None

    zero_influences = jax.tree_map(zeros_in_leaf, model)

    return zero_influences


def make_zeros_grads(model: StackedRNN):
    def zeros_in_leaf(leaf):
        if eqx.is_array(leaf):
            return jnp.zeros(shape=leaf.shape)
        elif isinstance(leaf, StateIndex):
            return None
        else:
            return None

    zero_grads = jax.tree_map(
        zeros_in_leaf, model, is_leaf=lambda leaf: isinstance(leaf, StateIndex)
    )

    return zero_grads


def step_loss(
    model_spatial: StackedRNN,
    model_rtrl: StackedRNN,
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


def sparse_multiplication(D_t: Array, J_t_prev: Array):
    """
    J_t_prev: Is of shape, [n_hidden_units, n_hidden_units, n_hidden_units]
    """

    def product(J_t_theta: Array, i: int):
        res = jax.vmap(lambda J_t_theta_i_column: D_t[:, i] * J_t_theta_i_column[i])(
            J_t_theta
        )

        return res

    final = []
    for i in range(J_t_prev.shape[0]):
        final.append(product(J_t_prev[i], i))

    final = jnp.stack(final)

    return final


# This should be a pure function.
@eqx.filter_jit
def forward_rtrl(
    theta_spatial: StackedRNN,
    theta_rtrl: StackedRNN,
    h_prev: Array,
    input: Array,
    perturbations: Array,
    acc_grads: StackedRNN,
    jacobians_prev: StackedRNN,
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

    grads = spatial_grads
    for i in range(model.num_layers):
        ht_grad = hidden_states_grads[i]
        rtrl_grads = jax.tree_map(
            lambda jacobian: ht_grad @ jacobian,
            jacobians.layers[i].cell,
        )

        grads = eqx.tree_at(
            lambda model: model.layers[i].cell,
            grads,
            rtrl_grads,
            is_leaf=lambda x: x is None,
        )

    acc_grads = jax.tree_map(lambda acc_grad, grad: acc_grad + grad, acc_grads, grads)

    return h_t, acc_grads, jacobians, inmediate_jacobians_state, loss


def rtrl(
    model: StackedRNN, inputs: Array, jacobian_state: eqx.nn.State, targets: Array
):
    model_rtrl, model_spatial = eqx.partition(
        model,
        lambda leaf: is_rnn_cell_or_jacobian_index(leaf),
        is_leaf=is_rnn_cell_or_jacobian_index,
    )
    perturbations = jnp.zeros(shape=(model.num_layers, model.hidden_size))

    def f_repack(carry, data):
        h, acc_grads, jacobians, inmediate_jacobians_state = carry
        input, target = data

        h, acc_grads, jacobians, inmediate_jacobians_state, loss = forward_rtrl(
            model_spatial,
            model_rtrl,
            h,
            input,
            perturbations,
            acc_grads,
            jacobians,
            inmediate_jacobians_state,
            target,
        )

        return (h, acc_grads, jacobians, inmediate_jacobians_state), loss

    h_init = jnp.zeros(shape=(model.num_layers, model.hidden_size))
    acc_grads = make_zeros_grads(model)
    zero_jacobians = make_zeros_jacobian(model_rtrl)

    carry_out, losses = jax.lax.scan(
        f_repack,
        init=(h_init, acc_grads, zero_jacobians, jacobian_state),
        xs=(inputs, targets),
    )

    return jnp.sum(losses), carry_out[1]


def forward_sequence(model: StackedRNN, inputs: Array, jacobian_state: eqx.nn.State):
    hidden_state = jnp.zeros(shape=(model.num_layers, model.hidden_size))
    perturbations = jnp.zeros(shape=(model.num_layers, model.hidden_size))

    def f_repack(carry: Tuple[Array, eqx.nn.State], input: Array):
        h, jacobian_state = carry
        h, out, jacobian_state = model(h, input, perturbations, jacobian_state)
        return (h, jacobian_state), out

    # Ful forward pass over the sequence
    _, out = jax.lax.scan(
        lambda carry, input: f_repack(carry, input),
        init=(hidden_state, jacobian_state),
        xs=inputs,
    )

    return out


def loss_func(
    model: StackedRNN, inputs: Array, jacobian_state: eqx.nn.State, outputs: Array
):
    pred = forward_sequence(model, inputs, jacobian_state)
    errors = jnp.sum((pred - outputs) ** 2, axis=1)
    return jnp.sum(errors)


def bptt(
    model: StackedRNN, inputs: Array, jacobian_state: eqx.nn.State, outputs: Array
):
    loss, grads = eqx.filter_value_and_grad(loss_func)(
        model, inputs, jacobian_state, outputs
    )

    return loss, grads


T = 10
num_layers = 2
hidden_size = 10
input_size = 10
key = jax.random.PRNGKey(7)

model, inmediate_jacobians_state = make_with_state(StackedRNN)(
    key, num_layers=num_layers, hidden_size=hidden_size, input_size=input_size
)

inputs = jnp.ones(shape=(T, input_size), dtype=jnp.float32)
outputs = jnp.zeros(shape=(T, input_size), dtype=jnp.float32)
loss, grads = bptt(model, inputs, inmediate_jacobians_state, outputs)
loss_rtrl, grads_rtrl = rtrl(model, inputs, inmediate_jacobians_state, outputs)

for (path_a, leaf_a), (path_b, leaf_b) in zip(
    jax.tree_util.tree_leaves_with_path(grads),
    jax.tree_util.tree_leaves_with_path(grads_rtrl),
):
    print(path_a, path_b)
    print(jnp.allclose(leaf_a, leaf_b))

# As expected by the online setup the gradients of the first weight matrices
# are not equal to bptt.
