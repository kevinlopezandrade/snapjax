from abc import abstractmethod
from typing import Any, List, Sequence, Tuple

import equinox as eqx
from jaxtyping import Array

State = Sequence[Array]
Jacobians = Tuple["RTRLCell", Array]  # I_t, D_t


class RTRLCell(eqx.Module):
    """
    s_(t) = f(s_(t-1), x(t))
    """

    hidden_size: eqx.AbstractVar[int]
    input_size: eqx.AbstractVar[int]

    @abstractmethod
    def f(self, state: State, input: Array) -> State:
        ...

    @staticmethod
    @abstractmethod
    def init_state(cell: "RTRLCell") -> State:
        ...

    @staticmethod
    @abstractmethod
    def make_zero_jacobians(cell: "RTRLCell") -> "RTRLCell":
        ...

    @staticmethod
    @abstractmethod
    def make_sp_pattern(cell: "RTRLCell") -> "RTRLCell":
        ...


class RTRLLayer(eqx.Module):
    """
    s_(t), theta_t, y_(t) = f(s_(t-1), x(t))
    where theta_t = (jacobians, dynamics).
    """

    cell: eqx.AbstractVar[RTRLCell]

    @abstractmethod
    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell = None,
    ) -> Tuple[State, Jacobians, Array]:
        """
        If sp_projection_cell is not None, then the sparse jacobians must be
        returned as transposed jacobians for efficieny in the algorithm.
        """
        ...


class RTRLStacked(eqx.Module):
    """
    s_1:L_(t), theta_1:L_(t), y_L(t) = f_1:L(s_1:L_(t-1), x(t))
    """

    layers: eqx.AbstractVar[List[RTRLLayer]]
    num_layers: eqx.AbstractVar[int]
    d_inp: eqx.AbstractVar[int]
    d_out: eqx.AbstractVar[int]

    @abstractmethod
    def f(
        self,
        state: Sequence[State],
        input: Array,
        perturbations: Array,
        sp_projection_tree: "RTRLStacked" = None,
    ) -> Tuple[Sequence[State], Sequence[Jacobians], Array]:
        ...

    @abstractmethod
    def get_sp_projection_tree(self) -> "RTRLStacked":
        ...


def is_rtrl_cell(node: Any):
    if isinstance(node, RTRLCell):
        return True
    else:
        return False
