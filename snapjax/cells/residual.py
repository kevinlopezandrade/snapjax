from typing import Any, Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State


class LinearResidual(RTRLCell["LinearResidual"]):
    W: Array
    U: Array
    dt: float = eqx.field(static=True)

    def __init__(self, W: Array, U: Array, dt: float):
        self.W = W
        self.U = U
        self.dt = dt

        self.input_size = U.shape[1]
        self.hidden_size = U.shape[0]

    def f(self, state: State, input: Array) -> State:
        h_out = state + self.dt * (self.W @ state + self.U @ input)

        return h_out


class Residual(RTRLCell["Residual"]):
    W: Array
    U: Array
    dt: float = eqx.field(static=True)

    def __init__(self, W: Array, U: Array, dt: float):
        self.W = W
        self.U = U
        self.dt = dt

        self.input_size = U.shape[1]
        self.hidden_size = U.shape[0]

    def f(self, state: State, input: Array) -> State:
        h_out = state + self.dt * jnp.tanh(self.W @ state + self.U @ input)

        return h_out


class LinearDecoderLayer(RTRLLayer):
    cell: RTRLCell[Any]
    C: Array
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell[Any] = None,
    ) -> Tuple[State, Jacobians[RTRLCell[Any]], Array]:
        h_out, jacobians = self.cell.value_and_jacobian(
            state, input, sp_projection_cell
        )
        h_out = h_out + perturbation
        y_out = self.C @ h_out

        return h_out, jacobians, y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C @ h_out

        return h_out, y_out
