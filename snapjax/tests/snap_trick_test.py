import time

import equinox as eqx
import jax.random as jrandom
from jax import config
from jax.lax import stop_gradient
from jax.random import PRNGKey
from jaxtyping import Array

from snapjax.cells.base import RTRLCell, State
from snapjax.cells.initializers import glorot_weights, normal_channels
from snapjax.cells.readout import LinearTanhReadout
from snapjax.cells.stacked import StackedCell
from snapjax.cells.utils import snap_n_mask
from snapjax.losses import masked_quadratic
from snapjax.sp_jacrev import make_jacobian_projection

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.tree_util as jtu

from snapjax.algos import rtrl
from snapjax.bptt import bptt
from snapjax.tests.utils import get_random_mask, get_random_sequence

ATOL = 1e-12
RTOL = 0.0


def diag_matrix(W: Array):
    return jnp.diag(jnp.diag(W))


class RNNDiag(RTRLCell):
    W: Array
    diag: Array
    U: Array
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, W: Array, U: Array):
        self.W = W
        self.U = U
        self.diag = diag_matrix(W)

        self.input_size = W.shape[1]
        self.hidden_size = W.shape[0]

    def f(self, state: State, input: Array) -> State:
        h_a = self.diag @ state + stop_gradient((self.W - self.diag) @ state)
        h_u = self.U @ input
        h = h_a + h_u
        return h

    def make_snap_n_mask(self, n: int):
        mask = jtu.tree_map(lambda leaf: snap_n_mask(leaf, n), self)
        return mask


class RNN(RTRLCell):
    W: Array
    U: Array
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, W: Array, U: Array):
        self.W = W
        self.U = U

        self.input_size = W.shape[1]
        self.hidden_size = W.shape[0]

    def f(self, state: State, input: Array) -> State:
        h_a = self.W @ state
        h_u = self.U @ input
        h = h_a + h_u
        return h

    def make_snap_n_mask(self, n: int):
        mask = jtu.tree_map(lambda leaf: snap_n_mask(leaf, n), self)
        return mask


def test_snap_trick():
    """
    BPTT and SNAP-1 must agree in the acc_grads if we use stop_grad
    in combination with BPTT.
    """
    T = 150
    N = 128

    seed = int(time.time() * 1000)
    w_key, u_key, c_key, data_key = jrandom.split(PRNGKey(seed), 4)
    W_0 = glorot_weights(w_key, inp_dim=N, out_dim=N)
    C = normal_channels(c_key, out_dim=N, inp_dim=N)
    U = normal_channels(u_key, out_dim=N, inp_dim=N)
    cell_stop_grad = RNNDiag(W_0, U)
    rnn_stop_grad = LinearTanhReadout(C, cell=cell_stop_grad)
    rnn_stop_grad = StackedCell([rnn_stop_grad])

    cell = RNN(W_0, U)
    rnn = LinearTanhReadout(C, cell=cell)
    rnn = StackedCell([rnn])
    jacobian_mask = rnn.get_snap_n_mask(1)
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    inputs = get_random_sequence(T, rnn_stop_grad)
    targets = get_random_sequence(T, rnn_stop_grad)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        rnn,
        inputs,
        targets,
        mask,
        jacobian_mask,
        jacobian_projection=jacobian_projection,
        use_scan=False,
        loss_func=masked_quadratic,
    )

    loss_bptt, acc_grads_bptt, _ = bptt(
        rnn_stop_grad, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_bptt, atol=ATOL, rtol=RTOL)

    assert jnp.allclose(
        acc_grads.layers[0].cell.W,
        acc_grads_bptt.layers[0].cell.diag,
        atol=ATOL,
        rtol=RTOL,
    )

    assert jnp.allclose(
        acc_grads.layers[0].cell.U,
        acc_grads_bptt.layers[0].cell.U,
        atol=ATOL,
        rtol=RTOL,
    )

    assert jnp.allclose(
        acc_grads.layers[0].C,
        acc_grads_bptt.layers[0].C,
        atol=ATOL,
        rtol=RTOL,
    )
