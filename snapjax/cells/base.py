from abc import abstractmethod
from typing import Any, List, Self, Sequence, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

State = Sequence[Array] | Array
Stacked = Sequence
Traces = chex.ArrayTree
ArrayTree = chex.ArrayTree


class RTRLCell(eqx.Module):
    """
    s_(t) = f(s_(t-1), x(t))
    """

    hidden_size: eqx.AbstractVar[int]
    input_size: eqx.AbstractVar[int]

    @abstractmethod
    def f(self, state: State, input: Array) -> State: ...

    def f_and_payload(
        self, state: State, input: Array, jacobian_projection: Self | None = None
    ) -> Tuple[State, Traces]:
        """
        If jacobian_projection is passed, it provides the sparse projection
        matrix of the jacobian.
        """
        ...

    def init_state(self) -> State: ...

    def init_traces(self) -> Traces:
        """
        Default method, override for different implementations.
        """
        ...

    def init_perturbation(self) -> Self: ...

    def make_snap_n_mask(self, n: int) -> Self:
        raise NotImplementedError()

    def compute_grads(self, hidden_state_grad: Array, *args) -> Self:
        raise NotImplementedError()

    def update_traces(
        self,
        prev_traces: Traces,
        payloads: Sequence[ArrayTree],
        traces_mask: Traces | None = None,
    ): ...


class RTRLLayer(eqx.Module):
    """
    s_(t), theta_t, y_(t) = f(s_(t-1), x(t))
    where theta_t = (jacobians, dynamics).
    """

    cell: eqx.AbstractVar[RTRLCell]
    d_inp: eqx.AbstractVar[int]
    d_out: eqx.AbstractVar[int]

    @abstractmethod
    def f_rtrl(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell | None = None,
    ) -> Tuple[State, Traces, Array]:
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
    sparse: eqx.AbstractVar[bool]

    @abstractmethod
    def f_rtrl(
        self,
        state: Stacked[State],
        input: Array,
        perturbations: Stacked[Array],
        jacobian_projection: Self | None = None,
    ) -> Tuple[Stacked[State], Stacked[Traces], Array]: ...

    def f_bptt(
        self, state: Stacked[State], input: Array
    ) -> Tuple[Stacked[State], Array]:
        raise NotImplementedError("BPTT mode has not been implemented for this Network")

    def get_snap_n_mask(self, n: int) -> Self:
        """
        Gets the maks for performing snap-n, where n >= 1.
        """
        if n == 0:
            return self.get_identity_mask()

        default = jax.default_backend()
        cpu_device = jax.devices("cpu")[0]

        # Move the RNN to CPU, to avoid creating masks in the GPU.
        cells = eqx.filter(self, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)
        cells = jtu.tree_map(lambda leaf: jax.device_put(leaf, cpu_device), cells)

        with jax.default_device(jax.devices("cpu")[0]):
            jacobian_mask = jtu.tree_map(
                lambda cell: cell.make_snap_n_mask(n),
                cells,
                is_leaf=is_rtrl_cell,
            )

        # Move back to GPU once computed.
        jacobian_mask = jtu.tree_map(
            lambda leaf: jax.device_put(leaf, jax.devices(default)[0]), jacobian_mask
        )

        return jacobian_mask

    def get_identity_mask(self) -> Self:
        default = jax.default_backend()
        cpu_device = jax.devices("cpu")[0]

        # Move the RNN to CPU, to avoid creating masks in the GPU.
        cells = eqx.filter(self, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)
        cells = jtu.tree_map(lambda leaf: jax.device_put(leaf, cpu_device), cells)

        with jax.default_device(jax.devices("cpu")[0]):
            jacobian_mask = jtu.tree_map(
                lambda cell: cell.identity_traces_mask(),
                cells,
                is_leaf=is_rtrl_cell,
            )

        # Move back to GPU once computed.
        jacobian_mask = jtu.tree_map(
            lambda leaf: jax.device_put(leaf, jax.devices(default)[0]), jacobian_mask
        )

        return jacobian_mask

    # Use __ and __ so that equinox does not wrap this method and I
    # can directly pass this to the jax.lax.scan.
    def __forward__(self, state: Stacked[State], input: Array):
        return self.f_bptt(state, input)

    def num_rtrl_layers(self) -> int:
        num = 0
        for layer in self.layers:
            if isinstance(layer, RTRLLayer):
                num += 1

        return num


def is_rtrl_cell(node: Any):
    if isinstance(node, RTRLCell):
        return True
    else:
        return False
