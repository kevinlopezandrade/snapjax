from typing import Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State
from snapjax.cells.utils import snap_n_mask
from snapjax.sp_jacrev import sp_jacrev


class GRU(RTRLCell):
    """
    Gated recurrent unit from Engel[10] which allows
    I_t being sparse.
    """

    W_iz: nn.Linear
    W_hz: nn.Linear
    W_hr: nn.Linear
    W_ir: nn.Linear
    W_ia: nn.Linear
    W_ha: nn.Linear
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = jrandom.split(key, 6)
        self.W_iz = nn.Linear(input_size, hidden_size, use_bias=False, key=keys[0])
        self.W_hz = nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[1])
        self.W_ir = nn.Linear(input_size, hidden_size, use_bias=False, key=keys[2])
        self.W_hr = nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[3])
        self.W_ia = nn.Linear(input_size, hidden_size, use_bias=False, key=keys[4])
        self.W_ha = nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[5])

        self.input_size = input_size
        self.hidden_size = hidden_size

    def f(self, state: State, input: Array) -> State:
        h = state
        x = input

        z = jax.nn.sigmoid(self.W_iz(x) + self.W_hz(h))
        r = jax.nn.sigmoid(self.W_ir(x) + self.W_hr(h))
        a = jax.nn.tanh(self.W_ia(x) + r * self.W_ha(h))

        h_new = (1 - z) * h + z * a

        return h_new

    def make_snap_n_mask(self, n: int):
        mask = jtu.tree_map(
            lambda leaf: snap_n_mask(leaf, n),
            self,
        )

        return mask


class GRULayer(RTRLLayer):
    cell: GRU
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
        self.cell = GRU(hidden_size, input_size, key=cell_key)
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=c_key)
        self.D = eqx.nn.Linear(input_size, hidden_size, use_bias=False, key=d_key)

        self.d_inp = input_size
        self.d_out = hidden_size

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: GRU = None,
    ) -> Tuple[State, Jacobians, Array]:
        """
        Returns h_(t), y_(t)
        """
        h_out = self.cell.f(state, input) + perturbation

        # Compute Jacobian and dynamics
        if sp_projection_cell:
            sp_jacobian_fun = sp_jacrev(
                jtu.Partial(GRU.f, state=state, input=input),
                sp_projection_cell,
                transpose=True,
            )
            inmediate_jacobian = sp_jacobian_fun(self.cell)
            dynamics_fun = jax.jacrev(GRU.f, argnums=1)
            dynamics = dynamics_fun(self.cell, state, input)
        else:
            jacobian_func = jax.jacrev(GRU.f, argnums=(0, 1))
            inmediate_jacobian, dynamics = jacobian_func(self.cell, state, input)

        # Project out
        y_out = self.C(h_out) + self.D(input)

        return h_out, (inmediate_jacobian, dynamics), y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C(h_out) + self.D(input)

        return h_out, y_out
