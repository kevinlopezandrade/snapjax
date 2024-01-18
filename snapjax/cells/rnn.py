from typing import Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State
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

    def f(self, state: Sequence[Array], input: Array) -> Array:
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0

        h = state
        x = input

        h_new = jnp.tanh(self.weights_hh @ h) + (self.weights_ih @ x) + bias

        return h_new

    @staticmethod
    def init_state(cell: "RNN"):
        return jnp.zeros(cell.hidden_size)

    @staticmethod
    def make_zero_jacobians(cell: "RNN"):
        zero_jacobians = jtu.tree_map(
            lambda leaf: jnp.zeros((cell.hidden_size, *leaf.shape)), cell
        )
        return zero_jacobians

    @staticmethod
    def make_sp_pattern(cell: "RNN"):
        h = jnp.ones(cell.hidden_size)
        x = jnp.ones(cell.input_size)

        jacobian_fun = jax.jacrev(RNN.f, argnums=0)
        jacobian = jacobian_fun(cell, h, x)

        sp = jtu.tree_map(
            lambda mat: BCOO.fromdense(jnp.abs(mat.reshape(mat.shape[0], -1)) > 0),
            jacobian,
        )

        return sp


class RNNLayer(RTRLLayer):
    cell: RNN
    C: eqx.nn.Linear
    D: eqx.nn.Linear

    def __init__(
        self,
        hidden_size: int = 10,
        input_size: int = 10,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        cell_key, c_key, d_key = jax.random.split(key, 3)
        self.cell = RNN(hidden_size, input_size, use_bias=use_bias, key=cell_key)
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=c_key)
        self.D = eqx.nn.Linear(hidden_size, input_size, use_bias=False, key=d_key)

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RNN = None,
    ) -> Tuple[State, Jacobians, Array]:
        """
        Returns h_(t), y_(t)
        """
        # To the RNN Cell
        h_out = self.cell.f(state, input) + perturbation

        # Compute Jacobian and dynamics
        if sp_projection_cell:
            sp_jacobian_fun = sp_jacrev(
                jtu.Partial(RNN.f, state=state, input=input), sp_projection_cell
            )
            inmediate_jacobian = sp_jacobian_fun(self.cell)
            dynamics_fun = jax.jacrev(RNN.f, argnums=1)
            dynamics = dynamics_fun(self.cell, state, input)
        else:
            jacobian_func = jax.jit(jax.jacrev(RNN.f, argnums=(0, 1)))
            inmediate_jacobian, dynamics = jacobian_func(self.cell, state, input)

        # Project out
        y_out = self.C(h_out) + self.D(input)

        return h_out, (inmediate_jacobian, dynamics), y_out