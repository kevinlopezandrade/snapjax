import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from utils import get_random_sequence, get_stacked_rnn

from snapjax.algos import rtrl

ATOL = 1e-07
RTOL = 0.0


def test_jacobians():
    T = 50
    model = get_stacked_rnn(4, 20, 20, sparse=True, seed=7)
    model_no_sp = get_stacked_rnn(4, 20, 20, sparse=False, seed=7)

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)

    loss, acc_grads, _ = rtrl(model, inputs, targets, use_snap_1=True, use_scan=False)
    loss_no_sp, acc_grads_no_sp, _ = rtrl(
        model_no_sp, inputs, targets, use_snap_1=True, use_scan=False
    )

    assert jnp.allclose(loss, loss_no_sp)

    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads),
        jtu.tree_leaves_with_path(acc_grads_no_sp),
    ):
        if jnp.allclose(leaf_a, leaf_b, atol=ATOL, rtol=RTOL):
            print("Match", jtu.keystr(key_a))
        else:
            print("Don't Match", jtu.keystr(key_a))
            print("\tMax difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
            print("\tMean difference: ", jnp.mean(jnp.abs(leaf_a - leaf_b)))
            passed = False

    assert passed
