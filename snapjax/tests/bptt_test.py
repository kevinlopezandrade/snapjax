import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from utils import get_random_sequence, get_stacked_rnn

from snapjax.algos import bptt, rtrl

ATOL = 5e-04
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

    loss, acc_grads, _ = rtrl(
        model, inputs, targets, use_snap_1=False, sparse=False, use_scan=False
    )
    loss_bptt, acc_grads_bptt = bptt(model, inputs, targets, use_scan=False)

    assert jnp.allclose(loss, loss_bptt)

    print("Comparing BPTT and SNAP with one layer")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed


def test_no_snap_mutliple_layers():
    """
    RTRL and BPTT where the stacked rnn contains multiple layers,
    should agree at least in the last layer.
    """
    T = 50
    model = get_stacked_rnn(4, 10, 10)
    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)

    loss, acc_grads, _ = rtrl(
        model, inputs, targets, use_snap_1=False, sparse=False, use_scan=False
    )
    loss_bptt, acc_grads_bptt = bptt(model, inputs, targets)

    assert jnp.allclose(loss, loss_bptt)

    acc_grads = acc_grads.layers[-1]
    acc_grads_bptt = acc_grads_bptt.layers[-1]

    print("Comparing BPTT and SNAP with multiple layers")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
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

    loss, acc_grads, _ = rtrl(
        model, inputs, targets, use_snap_1=False, sparse=False, use_scan=True
    )
    loss_no_scan, acc_grads_no_scan, _ = rtrl(
        model, inputs, targets, use_snap_1=False, sparse=False, use_scan=False
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
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed
