from abc import abstractmethod
from typing import Any, List, Sequence, Tuple

import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import Array

from snapjax.sp_jacrev import sp_projection_tree

State = Sequence[Array]
Jacobians = Tuple["RTRLCell", Array]  # I_t, D_t
Stacked = Sequence


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
    d_inp: eqx.AbstractVar[int]
    d_out: eqx.AbstractVar[int]

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

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        raise NotImplementedError("BPTT mode has not been implemented for this Network")


Layer = eqx.Module | RTRLLayer


class RTRLStacked(eqx.Module):
    """
    s_1:L_(t), theta_1:L_(t), y_L(t) = f_1:L(s_1:L_(t-1), x(t))
    """

    layers: eqx.AbstractVar[List[Layer]]
    num_layers: eqx.AbstractVar[int]

    @abstractmethod
    def f(
        self,
        state: Stacked[State],
        input: Array,
        perturbations: Stacked[Array],
        sp_projection_tree: "RTRLStacked" = None,
    ) -> Tuple[Stacked[State], Stacked[Jacobians], Array]:
        ...

    def f_bptt(
        self, state: Stacked[State], input: Array
    ) -> Tuple[Stacked[State], Array]:
        raise NotImplementedError("BPTT mode has not been implemented for this Network")

    def get_sp_projection_tree(self) -> "RTRLStacked":
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
                lambda cell: sp_projection_tree(cell.make_sp_pattern(cell)),
                cells,
                is_leaf=is_rtrl_cell,
            )

        # Move back to GPU once computed.
        sp_tree = jtu.tree_map(
            lambda leaf: jax.device_put(leaf, jax.devices(default)[0]), sp_tree
        )

        return sp_tree


def is_rtrl_cell(node: Any):
    if isinstance(node, RTRLCell):
        return True
    else:
        return False
