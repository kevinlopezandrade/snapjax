from typing import List, Tuple

import equinox as eqx
from jaxtyping import Array

from snapjax.cells.base import Jacobians, Layer, RTRLLayer, RTRLStacked, Stacked, State


class StackedCell(RTRLStacked):
    """
    It acts as a unique cell.
    """

    layers: List[Layer]
    num_layers: int = eqx.field(static=True)

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.num_layers = len(layers)

    def f(
        self,
        state: Stacked[State],
        input: Array,
        perturbations: Array,
        jacobian_projection: "StackedCell" = None,
    ):
        def _get_projection_cell(index: int):
            if jacobian_projection:
                layers = jacobian_projection.layers
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
