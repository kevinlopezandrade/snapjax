from typing import Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import State
from snapjax.cells.rnn import RNNStandard
from snapjax.cells.utils import snap_n_mask, snap_n_mask_bcoo
from snapjax.sp_jacrev import Mask


class FiringRateRNN(RNNStandard):
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


def sparsify_matrix(W: Array, sparsity_level: float, key: PRNGKeyArray):
    mask = jrandom.bernoulli(key, p=(1 - sparsity_level), shape=W.shape)

    return mask_matrix(W, mask=mask)


class SparseFiringRateRNN(RNNStandard):
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

    def make_snap_n_mask(self, n: int) -> Self:
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
