from typing import Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State
from snapjax.cells.utils import construct_snap_n_mask
from snapjax.sp_jacrev import sp_jacrev


class RNN(RTRLCell):
    weights_hh: eqx.nn.Linear
    weights_ih: eqx.nn.Linear
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

        self.weights_hh = eqx.nn.Linear(
            hidden_size, hidden_size, use_bias=use_bias, key=hhkey
        )
        self.weights_ih = eqx.nn.Linear(
            input_size, hidden_size, use_bias=False, key=ihkey
        )

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias

    def f(self, state: Sequence[Array], input: Array) -> Array:
        h = state
        x = input

        h_new = jnp.tanh(self.weights_hh(h) + self.weights_ih(x))

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

    def make_snap_n_mask(self, n: int) -> "RNN":
        """
        Mask every weight.
        """

        def _get_mask(leaf: Array):
            return construct_snap_n_mask(leaf, n)

        mask = jtu.tree_map(
            _get_mask,
            self,
        )

        return mask


class RNNLayer(RTRLLayer):
    cell: RNN
    C: eqx.nn.Linear
    D: eqx.nn.Linear
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

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
        self.D = eqx.nn.Linear(input_size, hidden_size, use_bias=False, key=d_key)

        self.d_inp = input_size
        self.d_out = hidden_size

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
                jtu.Partial(RNN.f, state=state, input=input),
                sp_projection_cell,
                transpose=True,
            )
            inmediate_jacobian = sp_jacobian_fun(self.cell)
            dynamics_fun = jax.jacrev(RNN.f, argnums=1)
            dynamics = dynamics_fun(self.cell, state, input)
        else:
            jacobian_func = jax.jacrev(RNN.f, argnums=(0, 1))
            inmediate_jacobian, dynamics = jacobian_func(self.cell, state, input)

        # Project out
        y_out = self.C(h_out) + self.D(input)

        return h_out, (inmediate_jacobian, dynamics), y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C(h_out) + self.D(input)

        return h_out, y_out
