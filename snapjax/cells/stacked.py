from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import (
    Jacobians,
    RTRLCell,
    RTRLLayer,
    RTRLStacked,
    State,
    is_rtrl_cell,
)
from snapjax.sp_jacrev import sp_projection_tree


class Stacked(RTRLStacked):
    """
    It acts as a unique cell.
    """

    layers: List[RTRLLayer]
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        cls: Type[RTRLLayer],
        num_layers: int,
        cls_kwargs: Optional[Dict[str, Any]] = None,
        *,
        key: PRNGKeyArray,
    ):
        self.num_layers = num_layers
        self.layers = []

        keys = jax.random.split(key, num=num_layers)
        for i in range(num_layers):
            layer = cls(**cls_kwargs, key=keys[i])
            self.layers.append(layer)

    def f(
        self,
        state: Sequence[State],
        input: Array,
        perturbations: Array,
        sp_projection_tree: "Stacked" = None,
    ):
        new_state: List[State] = []
        inmediate_jacobians: List[Jacobians] = []
        out = input

        for i, cell in enumerate(self.layers):
            if sp_projection_tree:
                sp_projection_cell = sp_projection_tree.layers[i].cell
            else:
                sp_projection_cell = None

            layer_state, jacobians, out = cell.f(
                state[i], out, perturbations[i], sp_projection_cell
            )
            new_state.append(layer_state)
            inmediate_jacobians.append(jacobians)

        return tuple(new_state), tuple(inmediate_jacobians), out

    def f_bptt(self, state: Sequence[State], input: Array) -> Tuple[State, Array]:
        new_state: List[State] = []
        for i, cell in enumerate(self.layers):
            layer_state, input = cell.f_bptt(state[i], input)
            new_state.append(layer_state)

        return tuple(new_state), input


class EncoderDecoder(RTRLStacked):
    encoder: nn.Linear
    decoder: nn.Linear
    layers: List[RTRLLayer]
    num_layers: int = eqx.field(static=True)
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, d_inp: int, d_out: int, stacked: Stacked, key: PRNGKeyArray):
        self.d_inp = d_inp
        self.d_out = d_out
        self.layers = stacked.layers
        self.num_layers = stacked.num_layers

        e_key, d_key = jrandom.split(key, 2)
        self.encoder = nn.Linear(d_inp, self.layers[0].d_inp, key=e_key)
        self.decoder = nn.Linear(self.layers[-1].d_out, d_out, key=d_key)

    def f(
        self,
        state: Sequence[State],
        input: Array,
        perturbations: Array,
        sp_projection_tree: RTRLStacked = None,
    ) -> Tuple[Sequence[State], Sequence[Jacobians], Array]:
        input = self.encoder(input)

        states, jacobians, output = Stacked.f(
            self, state, input, perturbations, sp_projection_tree
        )

        output = self.decoder(output)

        return states, jacobians, output
