import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from snapjax.algos import rtrl
from snapjax.tests.utils import get_random_batch, get_stacked_rnn

ATOL = 1e-05 # It has to be numerically stability.
RTOL = 0.0


def test_batch_no_sp_no_scan():
    T = 50
    B = 8
    model = get_stacked_rnn(4, 20, 20)
    batch_inp = get_random_batch(B, T, model)
    batch_out = get_random_batch(B, T, model)

    mapped_rtrl = jax.vmap(
        jtu.Partial(rtrl, model, use_snap_1=True, sparse=False, use_scan=False)
    )
    acc_loss_batch, acc_grads_batch, _ = mapped_rtrl(batch_inp, batch_out)

    passed = True
    for i in range(B):
        inp = batch_inp[i]
        out = batch_out[i]

        acc_loss, acc_grads, _ = rtrl(
            model, inp, out, use_snap_1=True, sparse=False, use_scan=False
        )

        assert jnp.allclose(acc_loss_batch[i], acc_loss)

        # Flatten the acc_grads tree.
        for (key_a, leaf_a), (key_b, leaf_b) in zip(
            jtu.tree_leaves_with_path(acc_grads_batch),
            jtu.tree_leaves_with_path(acc_grads),
        ):
            if jnp.allclose(leaf_a[i], leaf_b, atol=ATOL, rtol=RTOL):
                print("Match", jtu.keystr(key_a))
                print("\tMax difference: ", jnp.max(jnp.abs(leaf_a[i] - leaf_b)))
                print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a[i] - leaf_b)))
            else:
                print("Don't Match", jtu.keystr(key_a))
                print("\tMax difference: ", jnp.max(jnp.abs(leaf_a[i] - leaf_b)))
                print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a[i] - leaf_b)))
                passed = False

    assert passed


def test_batch_sp_no_scan():
    T = 50
    B = 8
    model = get_stacked_rnn(4, 20, 20)
    batch_inp = get_random_batch(B, T, model)
    batch_out = get_random_batch(B, T, model)

    mapped_rtrl = jax.vmap(
        jtu.Partial(rtrl, model, use_snap_1=True, sparse=True, use_scan=False)
    )
    acc_loss_batch, acc_grads_batch, _ = mapped_rtrl(batch_inp, batch_out)

    passed = True
    for i in range(B):
        inp = batch_inp[i]
        out = batch_out[i]

        acc_loss, acc_grads, _ = rtrl(
            model, inp, out, use_snap_1=True, sparse=True, use_scan=False
        )

        assert jnp.allclose(acc_loss_batch[i], acc_loss)

        # Flatten the acc_grads tree.
        for (key_a, leaf_a), (key_b, leaf_b) in zip(
            jtu.tree_leaves_with_path(acc_grads_batch),
            jtu.tree_leaves_with_path(acc_grads),
        ):
            if jnp.allclose(leaf_a[i], leaf_b, atol=ATOL, rtol=RTOL):
                print("Match", jtu.keystr(key_a))
            else:
                print("Don't Match", jtu.keystr(key_a))
                print("\tMax difference: ", jnp.max(jnp.abs(leaf_a[i] - leaf_b)))
                print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a[i] - leaf_b)))
                passed = False

    assert passed


def test_batch_sp_scan():
    T = 50
    B = 8
    model = get_stacked_rnn(4, 20, 20)
    batch_inp = get_random_batch(B, T, model)
    batch_out = get_random_batch(B, T, model)

    mapped_rtrl = jax.vmap(
        jtu.Partial(rtrl, model, use_snap_1=True, sparse=True, use_scan=True)
    )
    acc_loss_batch, acc_grads_batch, _ = mapped_rtrl(batch_inp, batch_out)

    passed = True
    for i in range(B):
        inp = batch_inp[i]
        out = batch_out[i]

        acc_loss, acc_grads, _ = rtrl(
            model, inp, out, use_snap_1=True, sparse=True, use_scan=True
        )

        assert jnp.allclose(acc_loss_batch[i], acc_loss)

        # Flatten the acc_grads tree.
        for (key_a, leaf_a), (key_b, leaf_b) in zip(
            jtu.tree_leaves_with_path(acc_grads_batch),
            jtu.tree_leaves_with_path(acc_grads),
        ):
            if jnp.allclose(leaf_a[i], leaf_b, atol=ATOL, rtol=RTOL):
                print("Match", jtu.keystr(key_a))
            else:
                print("Don't Match", jtu.keystr(key_a))
                print("\tMax difference: ", jnp.max(jnp.abs(leaf_a[i] - leaf_b)))
                print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a[i] - leaf_b)))
                passed = False

    assert passed
