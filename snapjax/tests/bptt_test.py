import time
from typing import Any, Callable

import equinox as eqx
from jax import config

from snapjax.cells.lru import LRU, LRULayer
from snapjax.cells.stacked import StackedCell
from snapjax.losses import masked_quadratic
from snapjax.sp_jacrev import make_jacobian_projection

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from snapjax.algos import rtrl, rtrl_exact
from snapjax.bptt import bptt
from snapjax.cells.utils import densify_jacobian_mask
from snapjax.tests.utils import (
    get_random_mask,
    get_random_sequence,
    get_stacked_rnn,
    get_stacked_rnn_exact,
    replace_rnn_with_diagonals,
)

ATOL = 1e-12
RTOL = 0.0


def test_no_snap_one_layer():
    """
    RTRL and BPTT should agree in the computed accumulated gradients,
    if there is only layer in the stacked rnn.
    """
    T = 50
    model = get_stacked_rnn(1, 256, 256)
    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )
    loss_bptt, acc_grads_bptt, _ = bptt(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_bptt)

    print("Comparing BPTT and RTRL with one layer")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_no_snap_multiple_layers_approx():
    """
    RTRL and BPTT where the stacked rnn contains multiple layers,
    should agree at least in the last layer.
    """
    T = 50
    model = get_stacked_rnn(4, 10, 10)
    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )
    loss_bptt, acc_grads_bptt, _ = bptt(
        model, inputs, targets, mask, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_bptt)

    acc_grads = acc_grads.layers[-1]
    acc_grads_bptt = acc_grads_bptt.layers[-1]

    print("Comparing BPTT and RTRL with multiple layers")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_no_snap_multiple_layers_exact():
    """
    RTRL and BPTT where the stacked rnn contains multiple layers,
    should agree in all the layers.
    """
    T = 50
    model = get_stacked_rnn_exact(4, 10, 10)
    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl_exact(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )
    loss_bptt, acc_grads_bptt, _ = bptt(
        model, inputs, targets, mask, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_bptt)

    acc_grads = acc_grads.layers
    acc_grads_bptt = acc_grads_bptt.layers

    print("Comparing BPTT and RTRL with multiple layers")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_scan_unrolled():
    T = 50
    model = get_stacked_rnn(4, 10, 10)
    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        model, inputs, targets, mask, use_scan=True, loss_func=masked_quadratic
    )
    loss_no_scan, acc_grads_no_scan, _ = rtrl(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_no_scan)

    print("Comparing RTRL with jax.lax.scan and without scan")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads),
        jtu.tree_leaves_with_path(acc_grads_no_scan),
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_bptt_snap_diagonal():
    """
    SNAP and BPTT must match if there is one layer
    and the recurrent matrices are diagonal.
    """
    T = 100
    model = get_stacked_rnn(1, 20, 20)
    model = replace_rnn_with_diagonals(model)
    jacobian_mask = model.get_snap_n_mask(1)
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        model,
        inputs,
        targets,
        mask,
        jacobian_mask=jacobian_mask,
        jacobian_projection=jacobian_projection,
        use_scan=False,
        loss_func=masked_quadratic,
    )
    loss_bptt, acc_grads_bptt, _ = bptt(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_bptt)

    print("Comparing BPTT and SNAP with one layer")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_bptt_snap_2_dense_rnn():
    """
    SNAP-2 on a Dense RNN is equivalent to BPTT, since there is no
    sparsity introduced by the mask.
    """
    T = 100
    model = get_stacked_rnn(1, 20, 20)
    jacobian_mask = model.get_snap_n_mask(2)
    jacobian_mask = densify_jacobian_mask(jacobian_mask)
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        model,
        inputs,
        targets,
        mask,
        jacobian_mask=jacobian_mask,
        jacobian_projection=jacobian_projection,
        use_scan=False,
        loss_func=masked_quadratic,
    )
    loss_bptt, acc_grads_bptt, _ = bptt(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_bptt)

    print("Comparing BPTT and SNAP with one layer")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_bptt_snap_diagonal_sparse():
    """
    SNAP and BPTT must match if there is one layer
    and the recurrent matrices are parametrized as diagonal.
    """
    T = 100
    model = get_stacked_rnn(1, 20, 20, sparse=True)
    model = replace_rnn_with_diagonals(model, sparse_weights=True)
    jacobian_mask = model.get_snap_n_mask(10)
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        model,
        inputs,
        targets,
        mask,
        jacobian_mask=jacobian_mask,
        jacobian_projection=jacobian_projection,
        use_scan=False,
        loss_func=masked_quadratic,
    )
    loss_bptt, acc_grads_bptt, _ = bptt(
        model,
        inputs,
        targets,
        mask,
        use_scan=False,
        loss_func=masked_quadratic,
    )

    assert jnp.allclose(loss, loss_bptt)

    print("Comparing BPTT and SNAP with one layer")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_lru():
    seed = int(time.time() * 1000)
    key = jrandom.PRNGKey(seed)
    cell_key, readout_key, mlp_key = jrandom.split(key, 3)

    T = 100
    H = 100
    cell = LRU(hidden_size=H, input_size=H, key=cell_key)
    lru_layer = LRULayer(cell=cell, d_out=H, key=readout_key)
    mlp = eqx.nn.Linear(in_features=H, out_features=H, use_bias=False, key=mlp_key)

    class Call(eqx.Module):
        f: Callable = eqx.field(static=True)

        @jax.jit
        def __call__(self, input) -> Any:
            return self.f(input)

    model = StackedCell([lru_layer, Call(jax.nn.gelu), mlp])

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = get_random_mask(T)

    loss, acc_grads, _ = rtrl(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )
    loss_bptt, acc_grads_bptt, _ = bptt(
        model, inputs, targets, mask, use_scan=False, loss_func=masked_quadratic
    )

    assert jnp.allclose(loss, loss_bptt)

    print("Comparing BPTT and RTRL with one layer for LRU")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed
