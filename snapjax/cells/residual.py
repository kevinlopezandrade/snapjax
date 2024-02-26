from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from snapjax.cells.base import RTRLCell, State
from snapjax.cells.utils import snap_n_mask


class LinearResidual(RTRLCell):
    W: Array
    U: Array
    dt: float = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, W: Array, U: Array, dt: float):
        self.W = W
        self.U = U
        self.dt = dt

        self.input_size = U.shape[1]
        self.hidden_size = U.shape[0]

    def f(self, state: State, input: Array) -> State:
        h_out = state + self.dt * (self.W @ state + self.U @ input)

        return h_out

    def make_snap_n_mask(self, n: int) -> Self:
        return jtu.tree_map(lambda leaf: snap_n_mask(leaf, n), self)


class Residual(RTRLCell):
    W: Array
    U: Array
    dt: float = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, W: Array, U: Array, dt: float):
        self.W = W
        self.U = U
        self.dt = dt

        self.input_size = U.shape[1]
        self.hidden_size = U.shape[0]

    def f(self, state: State, input: Array) -> State:
        h_out = state + self.dt * jnp.tanh(self.W @ state + self.U @ input)

        return h_out
