from typing import Any, List, Optional, Tuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree, Scalar


class RNN(eqx.Module):
    weights_hh: Float32[Array, "n_hidden n_hidden"]
    weights_ih: Float32[Array, "n_hidden input_dim"]
    bias: Optional[Float32[Array, "n_hidden"]]
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        hhkey, ihkey, bkey = jax.random.split(key, 3)

        # Use Glorot Initialization for the IH Matrix.
        lim = jnp.sqrt(1 / hidden_size)

        self.weights_hh = jax.random.uniform(
            hhkey, shape=(hidden_size, input_size), minval=-lim, maxval=lim
        )

        self.weights_ih = jax.random.uniform(
            ihkey, shape=(hidden_size, input_size), minval=-lim, maxval=lim
        )

        if use_bias:
            self.bias = jax.random.uniform(
                bkey, (hidden_size,), minval=-lim, maxval=lim
            )
        else:
            self.bias = None

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias

    @jax.jit
    def f(
        self, h: Float32[Array, "hidden_size"], x: Float32[Array, "input_size"]
    ) -> Float32[Array, "hidden_size"]:
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0

        h_new = self.weights_hh @ jnp.tanh(h) + (self.weights_ih @ x) + bias

        return h_new

    def __call__(
        self, h: Float32[Array, "hidden_size"], x: Float32[Array, "input_size"]
    ) -> Float32[Array, "hidden_size"]:
        return self.f(h, x)


def zero_influence_pytree(cell: RNN) -> RNN:
    def zeros_jacobian(leaf):
        return jnp.zeros(shape=(cell.hidden_size, *leaf.shape))

    return jax.tree_map(zeros_jacobian, cell)


class RNNLayerRTRL(eqx.Module):
    cell: RNN
    C: eqx.nn.Linear
    D: eqx.nn.Linear
    jacobian_index: eqx.nn.StateIndex

    def __init__(
        self, hidden_size: int, input_size: int, use_bias: bool, key: PRNGKeyArray
    ):
        cell_key, c_key, d_key = jax.random.split(key, 3)
        self.cell = RNN(hidden_size, input_size, use_bias=use_bias, key=cell_key)
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=c_key)
        self.D = eqx.nn.Linear(hidden_size, input_size, use_bias=False, key=d_key)
        self.jacobian_index = eqx.nn.StateIndex(
            (
                zero_influence_pytree(self.cell),
                jnp.zeros(shape=(self.cell.hidden_size, self.cell.hidden_size)),
            )
        )

    def __call__(
        self,
        h_prev: Float32[Array, "ndim"],
        input: Float32[Array, "ndim"],
        perturbation: Float32[Array, "ndim"],
        jacobian_state: eqx.nn.State,
    ):
        """
        Returns h_(t), y_(t)
        """
        # To the RNN Cell
        h_out = self.cell(h_prev, input) + perturbation

        # Compute Jacobian and set them in the state.
        jacobian_func = jax.jit(jax.jacrev(RNN.f, argnums=(0, 1)))
        inmediate_jacobian, dynamics = jacobian_func(self.cell, h_prev, input)
        new_jacobian_state = jacobian_state.set(
            self.jacobian_index, (inmediate_jacobian, dynamics)
        )

        # Project out
        y_out = self.C(jnp.tanh(h_out)) + self.D(input)

        return h_out, y_out, new_jacobian_state


class StackedRNN(eqx.Module):
    """
    It acts as a unique cell.
    """

    layers: List[RNNLayerRTRL]
    num_layers: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        num_layers: int,
        hidden_size: int,
        input_size: int,
        use_bias: bool = True,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias

        self.layers = []
        keys = jax.random.split(key, num=num_layers + 1)
        for i in range(num_layers):
            layer = RNNLayerRTRL(
                hidden_size, hidden_size, use_bias=use_bias, key=keys[i]
            )
            self.layers.append(layer)

    def f(
        self,
        h_prev: Float32[Array, "num_layers hidden_size"],
        input: Float32[Array, "ndim"],
        perturbations: Float32[Array, "num_layers hidden_size"],
        jacobians_state: eqx.nn.State,
    ) -> Tuple[
        Float32[Array, "num_layers hidden_size"],
        Float32[Array, "hidden_size"],
        eqx.nn.State,
    ]:
        h_collect: List[Array] = []
        out = input
        for i, cell in enumerate(self.layers):
            h_out, out, jacobians_state = cell(
                h_prev[i], out, perturbations[i], jacobians_state
            )
            h_collect.append(h_out)

        h_new = jnp.stack(h_collect)

        return h_new, out, jacobians_state

    def __call__(
        self,
        h_prev: Float32[Array, "num_layers hidden_size"],
        input: Float32[Array, "ndim"],
        perturbations: Float32[Array, "num_layers hidden_size"],
        jacobians_state: eqx.nn.State,
    ):
        return self.f(h_prev, input, perturbations, jacobians_state)
