import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from pta_cell import PTACell, StackedRTRL


def is_pta_cell(node: PyTree):
    if isinstance(node, PTACell):
        return True
    else:
        return False


def step_loss(
    y_t: Array, model: StackedRTRL, h_prev: Array, input: Array, perturbations: Array
):
    _, y_hat = model(h_prev, input, perturbations)
    return jnp.sum((y_hat - y_t) ** 2)


num_layers = 4
hidden_size = 4
input_size = 4
key = jax.random.PRNGKey(7)
model = StackedRTRL(
    key, num_layers=num_layers, hidden_size=hidden_size, input_size=input_size
)

input = jnp.ones(shape=(input_size,))
output = jnp.zeros(shape=(input_size,))

h_prev = jnp.ones(shape=(num_layers, hidden_size))
perturbations = jnp.zeros(shape=(num_layers, hidden_size))

gradient = jax.grad(step_loss, argnums=(1, 4))

grad_theta, grad_h = gradient(output, model, h_prev, input, perturbations)
grad_weights_hh_first_layer = grad_theta.layers[0].cell.weights_hh

cell = model.layers[0].cell
hidden_func_grad = jax.jacrev(PTACell.f, argnums=(0,))
ht_grad = hidden_func_grad(cell, h_prev[0], input)[0]
grad_alternative_computation = grad_h[0] @ ht_grad.weights_hh
print(grad_weights_hh_first_layer)
print(grad_alternative_computation)
print(jnp.allclose(grad_weights_hh_first_layer, grad_alternative_computation))
