from abc import abstractmethod
from typing import Any, List

import equinox as eqx


class RTRLCell(eqx.Module):
    hidden_size: eqx.AbstractVar[int]
    input_size: eqx.AbstractVar[int]


class RTRLLayer(eqx.Module):
    cell: eqx.AbstractVar[RTRLCell]
    cell_sp_projection: eqx.AbstractVar[RTRLCell]

    @abstractmethod
    def f(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def f_sp(self, *args, **kwargs) -> Any:
        ...


class RTRLStacked(eqx.Module):
    layers: eqx.AbstractVar[List[RTRLLayer]]
    num_layers: eqx.AbstractVar[int]
