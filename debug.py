import equinox as eqx
import jax
import jax.numpy as jnp
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


def step_loss_new(
    model_spatial: StackedRTRL,
    model_rtrl: StackedRTRL,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    y_t: Array,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, y_hat = model(h_prev, x_t, perturbations)

    diff = (y_t - y_hat) ** 2

    return jnp.sum(diff)


def combine_apply(
    model_spatial: StackedRTRL,
    model_rtrl: StackedRTRL,
    h_prev: Array,
    input: Array,
    perturbations: Array,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_new, y_out = StackedRTRL.f(model, h_prev, input, perturbations)
    return h_new


def forward_rtrl(
    theta_spatial: StackedRTRL,
    theta_rtrl: StackedRTRL,
    h_prev: Array,
    input: Array,
    perturbations: Array,
):
    """
    Before I had the jacobian to be [4, 4, 4] for only one hidden state.
    since my function -> R^4.

    Now my function -> R^Lx4, where L is the number of Layers.
    So if I want to find the jacobian of respect to hidden state at
    layer 
    """
    jacobian_func = jax.jacrev(combine_apply, argnums=(1, 2))
    jacobian = jacobian_func(theta_spatial, theta_rtrl, h_prev, input, perturbations)
    inmediate_jacobian, dynamics = jacobian

    def test(model_spatial, model_rtrl, h_prev, input, perturbations):
        return combine_apply(model_spatial, model_rtrl, h_prev, input, perturbations)[1]

    test_jacobian_func = jax.jacrev(test, argnums=(1, 2))
    jacobian_test = test_jacobian_func(
        theta_spatial, theta_rtrl, h_prev, input, perturbations
    )
    inmediate_jacobian_test, dynamics_test = jacobian_test

    print(
        jnp.allclose(
            inmediate_jacobian.layers[0].cell.weights_hh[1],
            inmediate_jacobian_test.layers[0].cell.weights_hh,
        )
    )

    # This is \delta h_0 \ delta W_hh(1) so the jacobian of h_0 wrt to weights
    # of the second layer, which should be zero since it does not contribute
    # at all.
    print(inmediate_jacobian.layers[1].cell.weights_hh[0])

    # This can be avoided actually. \delta h_1 \ W_hh(0) which I actually
    # don't need.
    print(inmediate_jacobian.layers[0].cell.weights_hh[1])


num_layers = 2
hidden_size = 4
input_size = 4
key = jax.random.PRNGKey(7)
model = StackedRTRL(
    key, num_layers=num_layers, hidden_size=hidden_size, input_size=input_size
)

model_spatial, model_rtrl = eqx.partition(model, eqx.is_array, is_leaf=is_pta_cell)


input = jnp.ones(shape=(input_size,))
output = jnp.zeros(shape=(input_size,))
h_prev = jnp.ones(shape=(num_layers, hidden_size))
perturbations = jnp.zeros(shape=(num_layers, hidden_size))

forward_rtrl(model_spatial, model_rtrl, h_prev, input, perturbations)
