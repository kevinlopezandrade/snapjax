import jax.tree_util as jtu
import optax as opt
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.tree_util as jtu

from snapjax.algos import (  # TODO: This is an ugly workaround.
    forward_rtrl,
    make_init_state,
)
from snapjax.losses import mean_squared_loss
from snapjax.rtrl.base import RTRLApprox
from snapjax.tests.utils import get_random_mask, get_random_sequence, get_stacked_rnn
from snapjax.utils import apply_update

ATOL = 1e-12
RTOL = 0.0


def test_online_updates():
    T = 50
    model = get_stacked_rnn(1, 256, 256)
    inp = get_random_sequence(T, model)
    target = get_random_sequence(T, model)
    mask = get_random_mask(T)

    optimizer = opt.adam(learning_rate=0.1)
    optim_state = optimizer.init(model)

    update_func = jtu.Partial(apply_update, optimizer=optimizer)
    algo = RTRLApprox(loss_func=mean_squared_loss, update_func=update_func)

    jacobians = algo.init_traces(model)
    h = make_init_state(model)
    model_no_scan = model
    for i in range(T):
        h, grads, jacobians, loss, y_hat = algo.step(
            model_no_scan,
            jacobians,
            h,
            inp[i],
            target[i],
            mask[i],
        )

        # Update online
        if mask[i]:
            model_no_scan, optim_state = apply_update(
                model_no_scan, grads, optim_state, optimizer
            )

    optimizer = opt.adam(learning_rate=0.1)
    optim_state = optimizer.init(model)
    update_func = jtu.Partial(apply_update, optimizer=optimizer)
    algo = RTRLApprox(loss_func=mean_squared_loss, update_func=update_func)

    model_scan = model
    _, model_scan, opt_state_scan = algo.rtrl_online(
        model_scan, inp, target, mask, optim_state
    )

    print("Comparing RTRL and RTRL scan")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(model_no_scan),
        jtu.tree_leaves_with_path(model_scan),
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
