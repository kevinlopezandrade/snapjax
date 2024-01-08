from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import RTRLCell, RTRLLayer, RTRLStacked
from snapjax.sp_jacrev import sp_jacrev, sp_projection_matrices


class RNN(RTRLCell):
    weights_hh: Array
    weights_ih: Array
    bias: Array
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
    def f(self, h: Array, x: Array) -> Array:
        print("Compiling call to RNN.f")
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0
        weights_hh = self.weights_hh
        weights_ih = self.weights_ih

        h_new = weights_hh @ jnp.tanh(h) + (weights_ih @ x) + bias

        return h_new


def _get_sp_rnn(cell: RNN):
    h = jnp.ones(cell.hidden_size)
    x = jnp.ones(cell.input_size)

    jacobian_fun = jax.jacrev(RNN.f, argnums=0)
    jacobian = jacobian_fun(cell, h, x)

    sp = jtu.tree_map(
        lambda mat: BCOO.fromdense(jnp.abs(mat.reshape(mat.shape[0], -1)) > 0), jacobian
    )

    return sp


class RNNLayer(RTRLLayer):
    cell: RNN
    C: eqx.nn.Linear
    D: eqx.nn.Linear
    cell_sp_projection: RNN = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        cell_key, c_key, d_key = jax.random.split(key, 3)
        self.cell = RNN(hidden_size, input_size, use_bias=use_bias, key=cell_key)
        self.cell_sp_projection = sp_projection_matrices(_get_sp_rnn(self.cell))
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=c_key)
        self.D = eqx.nn.Linear(hidden_size, input_size, use_bias=False, key=d_key)

    def f(
        self,
        h_prev: Array,
        input: Array,
        perturbation: Array,
    ) -> Tuple[Array, Array, RNN, Array]:
        """
        Returns h_(t), y_(t)
        """
        print("Compiling call to RNNLayerRTRL.f")
        # To the RNN Cell
        h_out = self.cell.f(h_prev, input) + perturbation

        # Compute Jacobian and dynamics
        jacobian_func = jax.jit(jax.jacrev(RNN.f, argnums=(0, 1)))
        inmediate_jacobian, dynamics = jacobian_func(self.cell, h_prev, input)

        # Project out
        y_out = self.C(jnp.tanh(h_out)) + self.D(input)

        return h_out, y_out, inmediate_jacobian, dynamics

    def f_sp(
        self, h_prev: Array, input: Array, perturbation: Array
    ) -> Tuple[Array, Array, RNN, Array]:
        # To the RNN Cell
        h_out = self.cell.f(h_prev, input) + perturbation

        # Compute Jacobian in sparse format.
        sp_jacobian_fun = sp_jacrev(
            jtu.Partial(RNN.f, h=h_prev, x=input), self.cell_sp_projection
        )
        sp_inmediate_jacobian = sp_jacobian_fun(self.cell)

        # Compute dynamics
        dynamics_fun = jax.jacrev(RNN.f, argnums=1)
        dynamics = dynamics_fun(self.cell, h_prev, input)

        # Project out
        y_out = self.C(jnp.tanh(h_out)) + self.D(input)

        return h_out, y_out, sp_inmediate_jacobian, dynamics


class StackedRNN(RTRLStacked):
    """
    It acts as a unique cell.
    """

    layers: List[RNNLayer]
    num_layers: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    sparse: bool = eqx.field(static=True)

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        input_size: int,
        use_bias: bool = True,
        sparse: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias
        self.sparse = sparse

        self.layers = []
        keys = jax.random.split(key, num=num_layers)
        for i in range(num_layers):
            layer = RNNLayer(hidden_size, hidden_size, use_bias=use_bias, key=keys[i])
            self.layers.append(layer)

    def f(self, h_prev: Array, input: Array, perturbations: Array):
        print("Compiling call to StackedRNN.f")
        h_collect: List[Array] = []
        inmediate_jacobians_collect = [None] * self.num_layers
        for i, cell in enumerate(self.layers):
            if self.sparse:
                f = cell.f_sp
            else:
                f = cell.f

            h_out, input, inmediate_jacobian, dynamics = f(
                h_prev[i], input, perturbations[i]
            )
            h_collect.append(h_out)
            inmediate_jacobians_collect[i] = (inmediate_jacobian, dynamics)

        h_new = jnp.stack(h_collect)

        return h_new, input, inmediate_jacobians_collect
