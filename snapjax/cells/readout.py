from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State


class LinearReadoutLayer(RTRLLayer):
    cell: RTRLCell
    C: Array
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, cell: RTRLCell, C: Array):
        self.cell = cell
        self.C = C

        self.d_inp = self.cell.input_size
        self.d_out = self.C.shape[0]

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell | None = None,
    ) -> Tuple[State, Jacobians, Array]:
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


class LinearTanhReadout(RTRLLayer):
    cell: RTRLCell
    C: Array
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, C: Array, cell: RTRLCell) -> None:
        self.C = C
        self.cell = cell

        self.d_inp = self.cell.input_size
        self.d_out = self.C.shape[0]

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell | None = None,
    ) -> Tuple[State, Jacobians, Array]:
        h_out, jacobians = self.cell.value_and_jacobian(
            state, input, sp_projection_cell
        )
        h_out = h_out + perturbation
        y_out = self.C @ jnp.tanh(h_out)
        y_out = y_out / jnp.sqrt((self.cell.hidden_size))

        return h_out, jacobians, y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C @ jnp.tanh(h_out)
        y_out = y_out / jnp.sqrt((self.cell.hidden_size))

        return h_out, y_out


class StateSpaceReadout(RTRLLayer):
    cell: RTRLCell
    C: Array
    D: Array
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(
        self,
        cell: RTRLCell,
        C: Array,
        D: Array,
    ):

        self.cell = cell
        self.C = C
        self.D = D

        self.d_inp = self.cell.input_size
        self.d_out = self.D.shape[0]

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell | None = None,
    ) -> Tuple[State, Jacobians, Array]:
        """
        Returns h_(t), y_(t)
        """
        # To the RNN Cell
        h_out, jacobians = self.cell.value_and_jacobian(
            state, input, sp_projection_cell
        )
        inmediate_jacobian, dynamics = jacobians

        h_out = h_out + perturbation

        # Project out
        y_out = self.C @ h_out + self.D @ input

        return h_out, (inmediate_jacobian, dynamics), y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C @ h_out + self.D @ input

        return h_out, y_out
