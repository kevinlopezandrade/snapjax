from jax import config

from snapjax.losses import masked_quadratic

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.tree_util as jtu

from snapjax.algos import rtrl
from snapjax.cells.utils import densify_jacobian_mask
from snapjax.sp_jacrev import make_jacobian_projection
from snapjax.tests.utils import get_random_mask, get_random_sequence, get_stacked_rnn

ATOL = 1e-12
RTOL = 0.0


def test_jacobians():
    T = 50
    model = get_stacked_rnn(4, 20, 20)
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

    # Make the jacobian mask dense.
    jacobian_mask = densify_jacobian_mask(jacobian_mask)

    loss_no_sp, acc_grads_no_sp, _ = rtrl(
        model,
        inputs,
        targets,
        mask,
        jacobian_mask=jacobian_mask,
        use_scan=False,
        loss_func=masked_quadratic,
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
