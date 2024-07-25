from typing import Callable, Iterable, List, NamedTuple, Self, Sequence, Tuple

import equinox as eqx
import jax
import jax.experimental
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray, PyTree

from snapjax.algos import update_cell_jacobians
from snapjax.cells.base import RTRLCell, RTRLLayer, State, Traces
from snapjax.cells.utils import snap_n_mask, snap_n_mask_bcoo
from snapjax.sp_jacrev import Mask, sp_jacrev, standard_jacobian


@jax.jit
def update_cell_jacobians_new(
    D_t: Array,
    J_t_prev: RTRLCell,
    U_t: Array | None,
    I_t: RTRLCell,
    traces_mask: RTRLCell | None = None,
):
    if U_t is None:
        return update_cell_jacobians(I_t, D_t, J_t_prev, traces_mask)
    else:
        J_t = jtu.tree_map(
            lambda i_t, j_t_prev: U_t @ standard_jacobian(i_t)
            + D_t @ standard_jacobian(j_t_prev),
            I_t,
            J_t_prev,
        )
        return J_t


class RNNPayload(NamedTuple):
    I_t: Array
    D_t: Array
    U_t: Array


class RNNStandard(RTRLCell):
    def f_and_payload(
        self, state: State, input: Array, jacobian_projection: Self | None = None
    ) -> Tuple[State, RNNPayload]:
        if jacobian_projection:
            sp_jacobian_fun = sp_jacrev(
                self.f.__func__, jacobian_projection, transpose=True, argnums=(0, 1, 2)
            )
            inmediate_jacobian, dynamics, input_dynamics = jax.lax.stop_gradient(
                sp_jacobian_fun(self, state, input)
            )

        else:
            jacobian_func = jax.jacrev(self.f.__func__, argnums=(0, 1, 2))
            inmediate_jacobian, dynamics, input_dynamics = jax.lax.stop_gradient(
                jacobian_func(self, state, input)
            )

        h = self.f(state, input)
        return h, RNNPayload(inmediate_jacobian, dynamics, input_dynamics)

    def init_state(self) -> State:
        """
        Default method, override for different implementations.
        """
        return jnp.zeros(self.hidden_size)

    def init_perturbation(self) -> State:
        return jnp.zeros(self.hidden_size)

    def init_traces(
        self, jacobian_projection_cell: Self | None = None, standarize: bool = True
    ) -> Self:
        """
        Default method, override for different implementations.
        """

        def _get_zero_jacobian(leaf: Array | BCOO):
            """
            The jacobians are still dense arrays.
            """
            if isinstance(leaf, BCOO):
                return jnp.zeros((self.hidden_size, leaf.nse))
            else:
                return standard_jacobian(jnp.zeros((self.hidden_size, *leaf.shape)))

        zero_jacobians = jtu.tree_map(
            _get_zero_jacobian,
            self,
            is_leaf=lambda node: isinstance(node, BCOO),
        )

        return zero_jacobians

    def compute_grads(self, ht_grad: Array, trace: Self):
        """
        Computes the grads for every param in the PyTree.
        """

        def matmul_by_h(ht_grad: Array, jacobian: Array | BCOO):
            if isinstance(jacobian, BCOO):
                return (jacobian @ ht_grad.T).T
            else:
                return ht_grad @ jacobian

        rtrl_cell_grads = jtu.tree_map(
            lambda jacobian: matmul_by_h(ht_grad, jacobian),
            trace,
            is_leaf=lambda node: isinstance(node, BCOO),
        )

        return rtrl_cell_grads

    def update_traces(
        self,
        prev_traces: Sequence[Traces],
        payloads: List[RNNPayload],
        traces_mask: Traces | None = None,
    ):
        # Hack for supporting previous versions.
        if not isinstance(prev_traces, Iterable):
            prev_traces = (prev_traces,)

        new_traces = [None] * len(prev_traces)
        for i, prev_trace in enumerate(prev_traces):
            payload = payloads[i]
            I_t = payload.I_t
            D_t = payload.D_t

            if i == 0:
                new_traces[i] = update_cell_jacobians_new(
                    D_t, prev_trace, None, I_t, traces_mask
                )
            else:
                new_traces[i] = update_cell_jacobians_new(
                    D_t, prev_trace, payload.U_t, new_traces[i - 1]
                )

        return tuple(new_traces)

    def identity_traces_mask(self):
        def _get_mask(leaf: Array | BCOO):
            if isinstance(leaf, BCOO):
                return Mask(jnp.ones((self.hidden_size, leaf.nse)))
            else:
                return Mask(jnp.ones((self.hidden_size, *leaf.shape)))

        mask = jtu.tree_map(
            _get_mask,
            self,
            is_leaf=lambda node: isinstance(node, BCOO),
        )

        return mask

    def make_snap_n_mask(self, n: int) -> Self:
        """
        Mask every weight.
        """

        def _get_mask(leaf: Array):
            if isinstance(leaf, BCOO):
                return snap_n_mask_bcoo(leaf, n)
            else:
                return snap_n_mask(leaf, n)

        mask = jtu.tree_map(
            _get_mask, self, is_leaf=lambda node: isinstance(node, BCOO)
        )

        return mask


class RNN(RNNStandard):
    weights_hh: eqx.nn.Linear
    weights_ih: eqx.nn.Linear
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        hhkey, ihkey, bkey = jax.random.split(key, 3)

        self.weights_hh = eqx.nn.Linear(
            hidden_size, hidden_size, use_bias=use_bias, key=hhkey
        )
        self.weights_ih = eqx.nn.Linear(
            input_size, hidden_size, use_bias=False, key=ihkey
        )

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias

    def f(self, state: State, input: Array) -> Array:
        h = state
        x = input

        h_new = jnp.tanh(self.weights_hh(h) + self.weights_ih(x))

        return h_new


class RNNLayer(RTRLLayer):
    cell: RNN
    C: eqx.nn.Linear
    D: eqx.nn.Linear
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int = 10,
        input_size: int = 10,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        cell_key, c_key, d_key = jax.random.split(key, 3)
        self.cell = RNN(hidden_size, input_size, use_bias=use_bias, key=cell_key)
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=c_key)
        self.D = eqx.nn.Linear(input_size, hidden_size, use_bias=False, key=d_key)

        self.d_inp = input_size
        self.d_out = hidden_size

    def f_rtrl(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RNN | None = None,
    ) -> Tuple[State, Traces, Array]:
        """
        Returns h_(t), y_(t)
        """
        # To the RNN Cell
        h_out, payload = self.cell.f_and_payload(state, input, sp_projection_cell)

        h_out = h_out + perturbation

        # Project out
        y_out = self.C(h_out) + self.D(input)

        return h_out, payload, y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)
        y_out = self.C(h_out) + self.D(input)

        return h_out, y_out


class RNNGeneral(RNNStandard):
    W: Array | BCOO
    U: Array | BCOO
    b: Array | BCOO
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    activation_function: Callable = eqx.field(static=True)

    def __init__(
        self,
        W: Array | BCOO,
        U: Array | BCOO,
        b: Array | BCOO | None = None,
        activation_function=jax.nn.tanh,
    ):
        self.W = W
        self.U = U
        self.b = b
        self.activation_function = activation_function

        self.input_size = U.shape[1]
        self.hidden_size = W.shape[0]

    def f(self, state: State, input: Array) -> Array:
        if self.b:
            h_new = self.W @ self.activation_function(state) + self.U @ input + self.b
        else:
            h_new = self.W @ self.activation_function(state) + self.U @ input

        return h_new
