from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import StateIndex
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


@jax.jit
def sparse_multiplication(D_t: Array, J_t_prev: Array):
    if J_t_prev.ndim == 2:
        return D_t @ J_t_prev

    indices = jnp.arange(0, D_t.shape[0])
    relevant = J_t_prev[indices, indices, :]
    res = jax.vmap(jnp.outer, in_axes=(1, 0))(D_t, relevant)
    return res


# This should be a pure function.
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
    matrix_product: Callable[[Array, Array], Array] = jnp.matmul,
    use_snap_1: bool = False,
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
    for i in range(theta_spatial.num_layers):
        inmediate_jacobian, dynamics = inmediate_jacobians_state.get(
            theta_rtrl.layers[i].jacobian_index
        )

        J_t_prev = jacobians.layers[i].cell

        # RTRL
        J_t = jax.tree_map(
            lambda i_t, j_t_prev: i_t + matrix_product(dynamics, j_t_prev),
            inmediate_jacobian,
            J_t_prev,
        )

        if use_snap_1:
            mask = jax.tree_map(
                lambda matrix: (jnp.abs(matrix) > 0.0).astype(jnp.float32),
                inmediate_jacobian,
            )

            # How efficient is to do this ?
            # I could potentially do the spatial grads calculation taking
            # into account
            J_t = jax.tree_map(lambda mask, j_t: mask * j_t, mask, J_t)

        jacobians = eqx.tree_at(lambda model: model.layers[i].cell, jacobians, J_t)

    grads = spatial_grads
    for i in range(theta_spatial.num_layers):
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
    model: StackedRNN,
    inputs: Array,
    jacobian_state: eqx.nn.State,
    targets: Array,
    matrix_product: Callable[[Array, Array], Array] = jnp.matmul,
    use_snap_1: bool = False,
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
            matrix_product=matrix_product,
            use_snap_1=use_snap_1,
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

    # carry_out = (h_init, acc_grads, zero_jacobians, jacobian_state)
    # losses = []
    # for i in range(inputs.shape[0]):
    #     carry_out, loss = f_repack(carry_out, (inputs[i], targets[i]))
    #     losses.append(loss)
    #
    # losses = jnp.stack(losses)

    h_out, acc_grads, jacobians, _ = carry_out

    return jnp.sum(losses), carry_out[1], jacobians


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
