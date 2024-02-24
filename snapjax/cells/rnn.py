from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State
from snapjax.cells.utils import snap_n_mask, snap_n_mask_bcoo


class RNN(RTRLCell["RNN"]):
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

    def f(self, state: State, input: Array) -> Array:
        h = state
        x = input

        h_new = jnp.tanh(self.weights_hh(h) + self.weights_ih(x))

        return h_new

    def make_snap_n_mask(self, n: int) -> "RNN":
        """
        Mask every weight.
        """

        def _get_mask(leaf: Array):
            if isinstance(leaf, BCOO):
                return snap_n_mask_bcoo(leaf, n)
            else:
                return snap_n_mask(leaf, n)

        mask = jtu.tree_map(
            _get_mask, self, is_leaf=lambda node: isinstance(node, BCOO)
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
        sp_projection_cell: RNN | None = None,
    ) -> Tuple[State, Jacobians[RNN], Array]:
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
        y_out = self.C(h_out) + self.D(input)

        return h_out, (inmediate_jacobian, dynamics), y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C(h_out) + self.D(input)

        return h_out, y_out
