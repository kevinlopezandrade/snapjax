from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State
from snapjax.cells.rnn import RNN
from snapjax.sp_jacrev import sp_jacrev
import jax.tree_util as jtu


def get_internal_weights(key: PRNGKeyArray, N: int, g: float) -> Array:
    key, _ = jrandom.split(key)
    variance = (g**2) / N
    weights = jrandom.normal(key, shape=(N, N))
    weights = jnp.sqrt(variance) * weights

    return weights


def get_random_vectors(key: PRNGKeyArray, out_dim: int, inp_dim: int) -> Array:
    key, _ = jrandom.split(key)
    variance = 1 / out_dim
    weights = jrandom.normal(key, shape=(out_dim, inp_dim))
    weights = jnp.sqrt(variance) * weights

    return weights


class FiringRateRNN(RTRLCell):
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

        h_new = self.W @ jnp.tanh(h) + jnp.sqrt(self.hidden_size) * (self.U @ x)

        return h_new

    @staticmethod
    def init_state(cell: "FiringRateRNN") -> State:
        return RNN.init_state(cell)

    @staticmethod
    def make_zero_jacobians(cell: "FiringRateRNN") -> "FiringRateRNN":
        return RNN.make_zero_jacobians(cell)

    @staticmethod
    def make_sp_pattern(cell: "FiringRateRNN") -> "FiringRateRNN":
        return RNN.make_sp_pattern(cell)


class ContinousRNNLayer(RTRLLayer):
    cell: RTRLCell
    C: Array
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, c: Array, cell: RTRLCell, dt: float) -> None:
        self.C = c
        self.cell = cell
        self.dt = dt

        self.d_inp = self.cell.input_size
        self.d_out = c.shape[0]

    @staticmethod
    def _step(cell: RTRLCell, state: State, input: Array, dt: float):
        h_out = cell.f.__func__(cell, state, input)
        h_out = (1 - dt) * state + (dt * h_out)
        return h_out

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell = None,
    ) -> Tuple[State, Jacobians, Array]:
        # Compute Jacobian and dynamics
        if sp_projection_cell:
            sp_jacobian_fun = sp_jacrev(
                jtu.Partial(self._step, state=state, input=input, dt=self.dt),
                sp_projection_cell,
                transpose=True,
            )
            inmediate_jacobian = sp_jacobian_fun(self.cell)
            dynamics_fun = jax.jacrev(self._step, argnums=1)
            dynamics = dynamics_fun(self.cell, state, input, self.dt)
        else:
            jacobian_func = jax.jacrev(self._step, argnums=(0, 1))
            inmediate_jacobian, dynamics = jacobian_func(
                self.cell, state, input, self.dt
            )

        # Proper discretization.
        h_out = self._step(self.cell, state, input, self.dt) + perturbation
        y_out = self.C @ jnp.tanh(h_out)

        return h_out, (inmediate_jacobian, dynamics), y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        h_out = (1 - self.dt) * state + (self.dt * h_out)
        y_out = self.C @ jnp.tanh(h_out)

        return h_out, y_out
