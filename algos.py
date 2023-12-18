from typing import Any, Callable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import StateIndex
from jax import config
from jaxtyping import Array

from rnn import RNN, StackedRNN

config.update("jax_numpy_rank_promotion", "raise")


def is_rnn_cell(node: Any):
    if isinstance(node, RNN):
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
        else:
            return None

    zero_grads = jax.tree_map(zeros_in_leaf, model)

    return zero_grads


def step_loss(
    model_spatial: StackedRNN,
    model_rtrl: StackedRNN,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    y_t: Array,
):
    print("Compiling step_loss")
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, y_hat, inmediate_jacobians = model(h_prev, x_t, perturbations)

    diff = (y_t - y_hat) ** 2

    return jnp.sum(diff), (h_t, y_hat, inmediate_jacobians)


@jax.jit
def sparse_multiplication(D_t: Array, J_t_prev: Array):
    print("Compiling sparse_multiplication")
    if J_t_prev.ndim == 2:
        return D_t @ J_t_prev

    indices = jnp.arange(0, D_t.shape[0])
    relevant = J_t_prev[indices, indices, :]
    res = jax.vmap(jnp.outer, in_axes=(1, 0))(D_t, relevant)
    return res


@eqx.filter_jit
def update_cell_jacobians(
    I_t: RNN,
    dynamics: Array,
    J_t_prev: RNN,
    matrix_product: Callable[[Array, Array], Array] = jnp.matmul,
    use_snap_1: bool = False,
):
    print("Compiling update_cell_jacobians")
    # RTRL
    J_t = jax.tree_map(
        lambda i_t, j_t_prev: i_t + matrix_product(dynamics, j_t_prev),
        I_t,
        J_t_prev,
    )

    if use_snap_1:
        mask = jax.tree_map(
            lambda matrix: (jnp.abs(matrix) > 0.0).astype(jnp.float32),
            I_t,
        )

        # How efficient is to do this ?
        # I could potentially do the spatial grads calculation taking
        # into account something else.
        J_t = jax.tree_map(lambda mask, j_t: mask * j_t, mask, J_t)

    return J_t


def update_jacobians_rtrl(
    jacobians_prev: StackedRNN,
    inmediate_jacobians: List[Tuple[RNN, Array]],
    matrix_product: Callable[[Array, Array], Array] = jnp.matmul,
    use_snap_1: bool = False,
):
    print("Compiling update_jacobians_rtrl")
    # Jax will do loop unrolling here, but number of layers is not that big
    # so it will be fine.
    jacobians = jacobians_prev
    for i in range(jacobians_prev.num_layers):
        I_t, D_t = inmediate_jacobians[i]
        J_t_prev = jacobians.layers[i].cell
        J_t = update_cell_jacobians(
            I_t, D_t, J_t_prev, matrix_product, use_snap_1=use_snap_1
        )

        jacobians = eqx.tree_at(lambda model: model.layers[i].cell, jacobians, J_t)

    return jacobians


def update_cells_grads(
    grads: StackedRNN, hidden_states_grads: List[Array], jacobians: StackedRNN
):
    print("Compiling update_cells_grads")
    layers = grads.num_layers
    for i in range(layers):
        ht_grad = hidden_states_grads[i]
        cell_grads = jax.tree_map(
            lambda jacobian: ht_grad @ jacobian, jacobians.layers[i].cell
        )
        grads = eqx.tree_at(
            lambda grads: grads.layers[i].cell,
            grads,
            cell_grads,
            is_leaf=lambda x: x is None,
        )

    return grads


# This should be a pure function.
def forward_rtrl(
    theta_spatial: StackedRNN,
    theta_rtrl: StackedRNN,
    acc_grads: StackedRNN,
    jacobians_prev: StackedRNN,
    h_prev: Array,
    input: Array,
    target: Array,
    perturbations: Array,
    matrix_product: Callable[[Array, Array], Array] = jnp.matmul,
    use_snap_1: bool = False,
):
    print("Compiling forward_rtrl")
    step_loss_and_grad = jax.value_and_grad(step_loss, argnums=(0, 4), has_aux=True)

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

    jacobians = update_jacobians_rtrl(
        jacobians_prev, inmediate_jacobians, matrix_product, use_snap_1
    )

    grads = update_cells_grads(spatial_grads, hidden_states_grads, jacobians)
    acc_grads = jax.tree_map(lambda acc_grad, grad: acc_grad + grad, acc_grads, grads)

    return h_t, acc_grads, jacobians, loss_t, y_hat


def rtrl(
    model: StackedRNN,
    inputs: Array,
    targets: Array,
    matrix_product: Callable[[Array, Array], Array] = jnp.matmul,
    use_snap_1: bool = False,
):
    model_rtrl, model_spatial = eqx.partition(
        model,
        lambda leaf: is_rnn_cell(leaf),
        is_leaf=is_rnn_cell,
    )
    perturbations = jnp.zeros(shape=(model.num_layers, model.hidden_size))

    def forward_repack(carry, data):
        print("Compiling forward_repack")
        input, target = data
        h_prev, acc_grads, jacobians_prev, acc_loss = carry
        (
            h_t,
            acc_grads,
            jacobians_t,
            loss_t,
            y_hat,
        ) = forward_rtrl(
            model_spatial,
            model_rtrl,
            acc_grads,
            jacobians_prev,
            h_prev,
            input,
            target,
            perturbations,
            matrix_product=matrix_product,
            use_snap_1=use_snap_1,
        )

        acc_loss = acc_loss + loss_t

        return (h_t, acc_grads, jacobians_t, acc_loss), y_hat

    h_init = jnp.zeros(shape=(model.num_layers, model.hidden_size))
    acc_grads = make_zeros_grads(model)
    zero_jacobians = make_zeros_jacobian(model_rtrl)
    acc_loss = 0.0

    carry_T, _ = jax.lax.scan(
        forward_repack,
        init=(h_init, acc_grads, zero_jacobians, acc_loss),
        xs=(inputs, targets),
    )

    # carry_out = (h_init, acc_grads, zero_jacobians, jacobian_state)
    # losses = []
    # for i in range(inputs.shape[0]):
    #     carry_out, loss = f_repack(carry_out, (inputs[i], targets[i]))
    #     losses.append(loss)
    #
    # losses = jnp.stack(losses)

    h_T, acc_grads, jacobians_T, acc_loss = carry_T

    return acc_loss, acc_grads, jacobians_T


def forward_sequence(model: StackedRNN, inputs: Array):
    hidden_state = jnp.zeros(shape=(model.num_layers, model.hidden_size))
    perturbations = jnp.zeros(shape=(model.num_layers, model.hidden_size))

    def f_repack(h: Array, input: Array):
        h, out, _ = model(h, input, perturbations)
        return h, out

    # Ful forward pass over the sequence
    _, out = jax.lax.scan(
        lambda carry, input: f_repack(carry, input),
        init=hidden_state,
        xs=inputs,
    )

    return out


def loss_func(model: StackedRNN, inputs: Array, targets: Array):
    pred = forward_sequence(model, inputs)
    errors = jnp.sum((pred - targets) ** 2, axis=1)
    return jnp.sum(errors)


def bptt(model: StackedRNN, inputs: Array, targets: Array):
    loss, grads = eqx.filter_value_and_grad(loss_func)(model, inputs, targets)

    return loss, grads
