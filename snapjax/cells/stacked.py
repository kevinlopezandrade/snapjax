from typing import List, Self, Tuple

import equinox as eqx
from jaxtyping import Array

from snapjax.cells.base import Layer, RTRLLayer, RTRLStacked, Stacked, State, Traces


class StackedCell(RTRLStacked):
    """
    It acts as a unique cell.
    """

    layers: List[Layer]
    num_layers: int = eqx.field(static=True)
    sparse: bool = eqx.field(static=True)

    def __init__(self, layers: List[Layer], sparse: bool = False):
        self.layers = layers
        self.num_layers = len(layers)
        self.sparse = sparse

    def f_rtrl(
        self,
        state: Stacked[State],
        input: Array,
        perturbations: Stacked[Array],
        jacobian_projection: Self | None = None,
    ):
        def _get_projection_cell(index: int):
            if jacobian_projection:
                layers = jacobian_projection.layers
                return layers[index].cell
            else:
                return None

        new_state: List[State] = []
        inmediate_jacobians: List[Traces] = []
        out = input
        cell_index = 0
        for layer in self.layers:
            if isinstance(layer, RTRLLayer):
                layer_state, jacobians, out = layer.f_rtrl(
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


class SingleCell(RTRLStacked):
    layers: List[Layer]  # To fullfill the general api.
    num_layers: int = eqx.field(static=True)
    sparse: bool = eqx.field(static=True)

    def __init__(self, layer: RTRLLayer, sparse: bool = False):
        self.layers = [layer]
        self.num_layers = 1
        self.sparse = sparse

    @property
    def layer(self):
        return self.layers[0]

    def f_rtrl(
        self,
        state: Stacked[State],
        input: Array,
        perturbations: Stacked[Array],
        jacobian_projection: Self | None = None,
    ):
        def _get_projection_cell():
            if jacobian_projection:
                layers = jacobian_projection.layers
                assert isinstance(layers[0], RTRLLayer)
                return layers[0].cell
            else:
                return None

        assert isinstance(self.layers[0], RTRLLayer)

        new_state, jacobians, out = self.layers[0].f_rtrl(
            state[0], input, perturbations[0], _get_projection_cell()
        )

        return tuple([new_state]), tuple([jacobians]), out

    def f_bptt(
        self, state: Stacked[State], input: Array
    ) -> Tuple[Stacked[State], Array]:

        assert isinstance(self.layers[0], RTRLLayer)

        new_state, out = self.layers[0].f_bptt(state[0], input)

        return tuple([new_state]), out
