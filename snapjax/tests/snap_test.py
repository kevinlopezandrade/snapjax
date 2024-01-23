from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.tree_util as jtu

from snapjax.algos import rtrl
from snapjax.tests.utils import get_random_sequence, get_stacked_rnn

ATOL = 1e-12
RTOL = 0.0


def test_jacobians():
    T = 50
    model = get_stacked_rnn(4, 20, 20)
    sp_projection_tree = model.get_sp_projection_tree()

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)

    loss, acc_grads = rtrl(
        model,
        inputs,
        targets,
        use_snap_1=True,
        sp_projection_tree=sp_projection_tree,
        use_scan=False,
    )
    loss_no_sp, acc_grads_no_sp = rtrl(
        model, inputs, targets, use_snap_1=True, use_scan=False
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
