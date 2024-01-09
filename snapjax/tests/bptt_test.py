import jax.numpy as jnp
import jax.tree_util as jtu
from utils import get_random_input_sequence, get_stacked_rnn

from snapjax.algos import bptt, rtrl


def test_no_snap_one_layer():
    """
    RTRL and BPTT should agree in the computed accumulated gradients,
    if there is only layer in the stacked rnn.
    """
    T = 100
    model = get_stacked_rnn(1, 20, 20, sparse=False)
    inputs = get_random_input_sequence(T, model)
    targets = get_random_input_sequence(T, model)

    loss, acc_grads, _ = rtrl(model, inputs, targets, use_snap_1=False, use_scan=True)
    loss_bptt, acc_grads_bptt = bptt(model, inputs, targets)

    assert jnp.allclose(loss, loss_bptt)

    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if jnp.allclose(leaf_a, leaf_b):
            print("Match", jtu.keystr(key_a))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("Max difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("Mean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            assert False


def test_no_snap_mutliple_layers():
    """
    RTRL and BPTT where the stacked rnn contains multiple layers,
    should agree at least in the last layer.
    """
    T = 100
    model = get_stacked_rnn(10, 20, 20, sparse=False)
    inputs = get_random_input_sequence(T, model)
    targets = get_random_input_sequence(T, model)

    loss, acc_grads, _ = rtrl(model, inputs, targets, use_snap_1=False, use_scan=True)
    loss_bptt, acc_grads_bptt = bptt(model, inputs, targets)

    assert jnp.allclose(loss, loss_bptt)

    acc_grads = acc_grads.layers[-1]
    acc_grads_bptt = acc_grads_bptt.layers[-1]

    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
    ):
        if not jnp.allclose(leaf_a, leaf_b):
            print("Don't Match", jtu.keystr(key_a))
            print("Max difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("Mean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            assert False
