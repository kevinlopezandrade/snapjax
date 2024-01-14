from typing import Any, Dict, List, Optional, Sequence, Type

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLLayer, RTRLStacked, State


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
        sparse: bool = False,
    ):
        new_state: List[State] = []
        inmediate_jacobians: List[Jacobians] = []
        out = input

        for i, cell in enumerate(self.layers):
            layer_state, jacobians, out = cell.f(
                state[i], out, perturbations[i], sparse
            )
            new_state.append(layer_state)
            inmediate_jacobians.append(jacobians)

        return tuple(new_state), tuple(inmediate_jacobians), out
