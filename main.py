import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import make_with_state
from jax import config
from jaxtyping import Array, PyTree

from pta_cell import List, PTACell, PTALayerRTRL, StackedRTRL, zero_influence_pytree

config.update("jax_numpy_rank_promotion", "raise")


def is_pta_cell(node: PyTree):
    if isinstance(node, PTACell):
        return True
    else:
        return False


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


def forward_rtrl(
    theta_spatial: StackedRTRL,
    theta_rtrl: StackedRTRL,
    h_prev: Array,
    input: Array,
    perturbations: Array,
    prev_influences: List[PTACell],
    jacobians_state: eqx.nn.State,
    target: Array,
):
    """
    Before I had the jacobian to be [4, 4, 4] for only one hidden state.
    since my function -> R^4.

    Now my function -> R^Lx4, where L is the number of Layers.
    So if I want to find the jacobian of respect to hidden state at
    layer
    """
    step_loss_and_grad = jax.value_and_grad(step_loss, argnums=(0, 4), has_aux=True)
    (loss, aux), (grads) = step_loss_and_grad(
        theta_spatial,
        theta_rtrl,
        h_prev,
        input,
        perturbations,
        jacobians_state,
        target,
    )

    h_t, jacobians_state = aux
    spatial_grads, hidden_states_grads = grads

    new_influences: List[PTACell] = []
    for layer, prev_influence in zip(theta_rtrl.layers, prev_influences):
        inmediate_jacobian, dynamics = jacobians_state.get(layer.jacobian_index)

        J_t = jax.tree_map(
            lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
            inmediate_jacobian,
            prev_influence,
        )
        new_influences.append(J_t)

    print(spatial_grads)
    for hidden_state_grad, new_influence in zip(hidden_states_grads, new_influences):
        grads = jax.tree_map(lambda j_t: hidden_state_grad @ j_t, new_influence)

        print(grads)



num_layers = 2
hidden_size = 4
input_size = 4
key = jax.random.PRNGKey(7)

model, jacobians_state = make_with_state(StackedRTRL)(
    key, num_layers=num_layers, hidden_size=hidden_size, input_size=input_size
)


model_spatial, model_rtrl = eqx.partition(model, eqx.is_array, is_leaf=is_pta_cell)


input = jnp.ones(shape=(input_size,))
output = jnp.zeros(shape=(input_size,))
h_prev = jnp.ones(shape=(num_layers, hidden_size))
perturbations = jnp.zeros(shape=(num_layers, hidden_size))

zero_influences = [zero_influence_pytree(layer.cell) for layer in model.layers]

forward_rtrl(
    model_spatial,
    model_rtrl,
    h_prev,
    input,
    perturbations,
    zero_influences,
    jacobians_state,
    output,
)
