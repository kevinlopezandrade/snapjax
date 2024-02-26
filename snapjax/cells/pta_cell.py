from typing import List, Self, Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, Scalar

from snapjax.cells.base import RTRLCell, RTRLLayer, State
from snapjax.cells.initializers import pta_matrix
from snapjax.cells.utils import snap_n_mask
from snapjax.sp_jacrev import sp_jacrev


class GLU(eqx.Module):
    W: eqx.nn.Linear
    V: eqx.nn.Linear
    n_dim: int = eqx.field(static=True)

    def __init__(self, n_dim: int, key: PRNGKeyArray):
        w_key, v_key = jax.random.split(key, 2)
        self.n_dim = n_dim

        self.W = eqx.nn.Linear(n_dim, n_dim, use_bias=True, key=w_key)
        self.V = eqx.nn.Linear(n_dim, n_dim, use_bias=True, key=v_key)

    def __call__(self, x: Array) -> Array:
        res = self.W(x) * jax.nn.sigmoid(self.V(x))
        return res


class PTACell(RTRLCell):
    weights_hh: Array
    weights_ih: nn.Linear
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
        if not (hidden_size % 2 == 0):
            raise ValueError("For the PTA Cell the hidden size must be a even number")

        num_rotation_blocks = hidden_size // 2
        hhkey, ihkey = jax.random.split(key, 2)

        # PTA Random Initialization
        hh_subkeys = jax.random.split(hhkey, num_rotation_blocks)
        alfas = [jax.random.uniform(key, minval=0, maxval=10) for key in hh_subkeys]
        thetas = [
            jax.random.uniform(key, minval=0, maxval=jnp.pi) for key in hh_subkeys
        ]

        self.weights_hh = pta_matrix(alfas, thetas)
        self.weights_ih = nn.Linear(
            input_size, hidden_size, use_bias=use_bias, key=ihkey
        )

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias

    def f(
        self,
        state: State,
        input: Array,
    ) -> State:
        h_new = jnp.tanh(self.weights_hh @ state + self.weights_ih(input))

        return h_new

    def make_snap_n_mask(self, n: int) -> Self:
        mask = jtu.tree_map(lambda leaf: snap_n_mask(leaf, n), self)

        return mask
