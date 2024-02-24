from typing import Any, Generic, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State
from snapjax.cells.utils import snap_n_mask, snap_n_mask_bcoo
from snapjax.sp_jacrev import Mask, sp_jacrev


class FiringRateRNN(RTRLCell["FiringRateRNN"]):
    W: Array  # mxm
    U: Array  # mxd
    dt: float = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, W: Array, U: Array, dt: float) -> None:
        self.W = W
        self.U = U
        self.dt = dt

        self.input_size = U.shape[1]
        self.hidden_size = W.shape[0]

    def f(self, state: State, input: Array) -> State:
        h = state
        x = input

        h_new = self.W @ jnp.tanh(h) + jnp.sqrt(self.hidden_size) * (self.U @ x)
        h_new = (1 - self.dt) * state + self.dt * h_new

        return h_new

    def make_snap_n_mask(self, n: int):
        mask = jtu.tree_map(lambda leaf: snap_n_mask(leaf, n), self)

        return mask


def mask_matrix(W: Array, mask: Array) -> BCOO:
    indices = jnp.argwhere(mask)
    data = W[indices[:, 0], indices[:, 1]]
    mat = BCOO(
        (data, indices), shape=W.shape, unique_indices=True, indices_sorted=False
    )
    mat = mat.sort_indices()

    return mat


class SparseFiringRateRNN(RTRLCell["SparseFiringRateRNN"]):
    W: BCOO
    U: Array
    dt: float = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        W: Array,
        U: Array,
        dt: float,
        sparsity_fraction: float,
        *,
        key: PRNGKeyArray,
    ):
        mask_hh = jrandom.bernoulli(key, p=(1 - sparsity_fraction), shape=W.shape)

        self.W = mask_matrix(W, mask_hh)
        self.U = U
        self.dt = dt
        self.hidden_size = W.shape[0]
        self.input_size = U.shape[1]

    def f(self, state: State, input: Array) -> State:
        h = state
        x = input

        h_new = self.W @ jnp.tanh(h) + jnp.sqrt(self.hidden_size) * (self.U @ x)
        h_new = (1 - self.dt) * h + self.dt * h_new

        return h_new

    def make_zero_jacobians(self) -> "SparseFiringRateRNN":
        def _get_zero_jacobian(leaf: Array | BCOO):
            """
            The jacobians are still dense arrays.
            """
            if isinstance(leaf, BCOO):
                return jnp.zeros((cell.hidden_size, leaf.nse))
            else:
                return jnp.zeros((cell.hidden_size, *leaf.shape))

        zero_jacobians = jtu.tree_map(
            _get_zero_jacobian,
            cell,
            is_leaf=lambda node: isinstance(node, BCOO),
        )
        return zero_jacobians

    def make_snap_n_mask(self, n: int) -> "SparseFiringRateRNN":
        """
        Only mask the weights that are sparse.
        """

        def _get_mask(leaf: Array | BCOO):
            if isinstance(leaf, BCOO):
                return snap_n_mask_bcoo(leaf, n)
            else:
                return Mask(jnp.ones((self.hidden_size, *leaf.shape)))

        mask = jtu.tree_map(
            _get_mask,
            self,
            is_leaf=lambda node: isinstance(node, BCOO),
        )

        return mask


G = TypeVar("G")


class LinearTanhReadout(RTRLLayer):
    cell: RTRLCell[G]
    C: Array
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, c: Array, cell: RTRLCell[G]) -> None:
        self.C = c
        self.cell = cell

        self.d_inp = self.cell.input_size
        self.d_out = c.shape[0]

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell[G] | None = None,
    ) -> Tuple[State, Jacobians[RTRLCell[G]], Array]:
        h_out, jacobians = self.cell.value_and_jacobian(
            state, input, sp_projection_cell
        )
        h_out = h_out + perturbation
        y_out = self.C @ jnp.tanh(h_out)

        return h_out, jacobians, y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C @ jnp.tanh(h_out)

        return h_out, y_out
