from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import equinox as eqx
import jax
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
from snapjax.sp_jacrev import sp_projection_matrices


class Stacked(RTRLStacked):
    """
    It acts as a unique cell.
    """

    layers: List[RTRLLayer]
    num_layers: int = eqx.field(static=True)
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(
        self,
        cls: Type[RTRLLayer],
        num_layers: int,
        d_inp: int,
        d_out: int,
        cls_kwargs: Optional[Dict[str, Any]] = None,
        *,
        key: PRNGKeyArray,
    ):
        self.num_layers = num_layers
        self.layers = []
        self.d_inp = d_inp
        self.d_out = d_out

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

    def get_sp_projection_tree(self):
        """
        Gets the sparse projection tree, from only
        the layers annotated as RTRLCell.
        """
        default = jax.default_backend()
        cpu_device = jax.devices("cpu")[0]

        # Move the RNN to CPU, to avoid creating the
        # explicit jacobians in the GPU.
        cells = eqx.filter(self, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)
        cells = jtu.tree_map(lambda leaf: jax.device_put(leaf, cpu_device), cells)

        with jax.default_device(jax.devices("cpu")[0]):
            sp_tree = jtu.tree_map(
                lambda cell: sp_projection_matrices(cell.make_sp_pattern(cell)),
                cells,
                is_leaf=is_rtrl_cell,
            )

        # Move back to GPU once computed.
        sp_tree = jtu.tree_map(
            lambda leaf: jax.device_put(leaf, jax.devices(default)[0]), sp_tree
        )

        return sp_tree
