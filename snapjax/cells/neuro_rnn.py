from typing import Sequence, Tuple

import jax.numpy as jnp
from jaxtyping import Array

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State
from snapjax.cells.rnn import RNN


class NeuroRNN(RTRLCell):
    W: Array  # mxm
    U: Array  # mxd
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, W: Array, U: Array) -> None:
        self.W = W
        self.U = U

        self.input_size = U.shape[1]
        self.hidden_size = W.shape[0]

    def f(self, state: State, input: Array) -> State:
        h = state
        x = input

        h_new = self.W @ jnp.tanh(h) + jnp.sqrt(self.hidden_size) * (self.U + x)

        return h_new

    @staticmethod
    def init_state(cell: "LinearRNN") -> State:
        return RNN.init_state(cell)

    @staticmethod
    def make_zero_jacobians(cell: "LinearRNN") -> "LinearRNN":
        return RNN.make_zero_jacobians(cell)

    @staticmethod
    def make_sp_pattern(cell: "LinearRNN") -> "LinearRNN":
        return RNN.make_sp_pattern(cell)


class NeuroRNNLayer(RTRLLayer):
    cell: NeuroRNN
    C: Array
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, c: Array, cell: NeuroRNN, dt: float = 0.5) -> None:
        self.c = c
        self.cell = cell

        self.d_inp = self.cell.input_size
        self.d_out = c.shape[0]
        self.dt = dt

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell = None,
    ) -> Tuple[State, Jacobians, Array]:
        h_out = self.cell.f(state, input) + perturbation
        h_out = (1 - self.dt) * state + (self.dt * h_out)

        # Compute Jacobian and dynamics
        if sp_projection_cell:
            sp_jacobian_fun = sp_jacrev(
                jtu.Partial(NeuroRNN.f, state=state, input=input),
                sp_projection_cell,
                transpose=True,
            )
            inmediate_jacobian = sp_jacobian_fun(self.cell)
            dynamics_fun = jax.jacrev(NeuroRNN.f, argnums=1)
            dynamics = dynamics_fun(self.cell, state, input)
        else:
            jacobian_func = jax.jacrev(NeuroRNN.f, argnums=(0, 1))
            inmediate_jacobian, dynamics = jacobian_func(self.cell, state, input)

        y_out = self.c @ jnp.tan(h_out)

        return h_out, (inmediate_jacobian, dynamics), y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.c @ jnp.tanh(h_out)

        return h_out, y_out
