from typing import Any, Dict, List, Optional, Tuple, Type

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, Layer, RTRLLayer, RTRLStacked, Stacked, State


class StackedCell(RTRLStacked):
    """
    It acts as a unique cell.
    """

    layers: List[Layer]
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
        state: Stacked[State],
        input: Array,
        perturbations: Array,
        sp_projection_tree: "StackedCell" = None,
    ):
        is_sparse = True if sp_projection_tree else False

        def _get_projection_cell(index: int):
            if is_sparse:
                layers = sp_projection_tree.layers
                return layers[index].cell
            else:
                return None

        new_state: List[State] = []
        inmediate_jacobians: List[Jacobians] = []
        out = input
        cell_index = 0
        for layer in self.layers:
            if isinstance(layer, RTRLLayer):
                layer_state, jacobians, out = layer.f(
                    state[cell_index],
                    out,
                    perturbations[cell_index],
                    _get_projection_cell(cell_index),
                )
                new_state.append(layer_state)
                inmediate_jacobians.append(jacobians)

                cell_index += 1
            else:
                out = layer(out)

        return tuple(new_state), tuple(inmediate_jacobians), out

    def f_bptt(
        self, state: Stacked[State], input: Array
    ) -> Tuple[Stacked[State], Array]:
        new_state: List[State] = []
        out = input

        cell_index = 0
        for layer in self.layers:
            if isinstance(layer, RTRLLayer):
                layer_state, out = layer.f_bptt(state[cell_index], out)
                new_state.append(layer_state)
                cell_index += 1
            else:
                out = layer(out)

        return tuple(new_state), out

    @classmethod
    def from_layers(cls, layers: List[Layer]):
        """
        The list of layers has to ensure compatiblity,
        from layers[i-1] and layers[i]
        """
        # Hack since its a dataclass.
        obj = cls.__new__(cls)
        object.__setattr__(obj, "layers", layers)
        object.__setattr__(obj, "num_layers", len(layers))

        return obj


class EncoderDecoder(RTRLStacked):
    encoder: nn.Linear
    decoder: nn.Linear
    layers: List[RTRLLayer]
    num_layers: int = eqx.field(static=True)
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, d_inp: int, d_out: int, stacked: StackedCell, key: PRNGKeyArray):
        self.d_inp = d_inp
        self.d_out = d_out
        self.layers = stacked.layers
        self.num_layers = stacked.num_layers

        e_key, d_key = jrandom.split(key, 2)
        self.encoder = nn.Linear(d_inp, self.layers[0].d_inp, key=e_key)
        self.decoder = nn.Linear(self.layers[-1].d_out, d_out, key=d_key)

    def f(
        self,
        state: Stacked[State],
        input: Array,
        perturbations: Array,
        sp_projection_tree: RTRLStacked = None,
    ) -> Tuple[Stacked[State], Stacked[Jacobians], Array]:
        input = self.encoder(input)

        states, jacobians, output = StackedCell.f(
            self, state, input, perturbations, sp_projection_tree
        )

        output = self.decoder(output)

        return states, jacobians, output

    def f_bptt(self, state: Stacked[State], input: Array) -> Tuple[State, Array]:
        input = self.encoder(input)
        state, output = StackedCell.f_bptt(self, state, input)
        output = self.decoder(output)

        return state, output
