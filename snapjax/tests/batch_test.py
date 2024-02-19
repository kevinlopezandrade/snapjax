from jax import config
from jax.experimental.sparse import BCOO

from snapjax.losses import masked_quadratic

config.update("jax_enable_x64", True)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from snapjax.algos import rtrl
from snapjax.cells.utils import densify_jacobian_mask
from snapjax.sp_jacrev import make_jacobian_projection
from snapjax.tests.utils import get_random_batch, get_random_mask_batch, get_stacked_rnn

ATOL = 1e-12
RTOL = 0.0


def test_batch_no_sp_no_scan():
    T = 50
    B = 8
    model = get_stacked_rnn(4, 20, 20)
    jacobian_mask = model.get_snap_n_mask(1)
    # Make the jacobian mask dense.
    jacobian_mask = densify_jacobian_mask(jacobian_mask)

    batch_inp = get_random_batch(B, T, model)
    batch_out = get_random_batch(B, T, model)
    batch_mask = get_random_mask_batch(B, T)

    mapped_rtrl = jax.vmap(
        jtu.Partial(
            rtrl,
            model,
            jacobian_mask=jacobian_mask,
            use_scan=False,
            loss_func=masked_quadratic,
        )
    )
    acc_loss_batch, acc_grads_batch, _ = mapped_rtrl(batch_inp, batch_out, batch_mask)

    passed = True
    for i in range(B):
        inp = batch_inp[i]
        out = batch_out[i]
        mask = batch_mask[i]

        acc_loss, acc_grads, _ = rtrl(
            model,
            inp,
            out,
            mask,
            jacobian_mask=jacobian_mask,
            use_scan=False,
            loss_func=masked_quadratic,
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


def test_batch_sp_no_scan():
    T = 50
    B = 8
    model = get_stacked_rnn(4, 20, 20)
    jacobian_mask = model.get_snap_n_mask(1)
    jacobian_projection = make_jacobian_projection(jacobian_mask)
    batch_inp = get_random_batch(B, T, model)
    batch_out = get_random_batch(B, T, model)
    batch_mask = get_random_mask_batch(B, T)

    mapped_rtrl = jax.vmap(
        jtu.Partial(
            rtrl,
            model,
            jacobian_mask=jacobian_mask,
            jacobian_projection=jacobian_projection,
            use_scan=False,
            loss_func=masked_quadratic,
        )
    )
    acc_loss_batch, acc_grads_batch, _ = mapped_rtrl(batch_inp, batch_out, batch_mask)

    passed = True
    for i in range(B):
        inp = batch_inp[i]
        out = batch_out[i]
        mask = batch_mask[i]

        acc_loss, acc_grads, _ = rtrl(
            model,
            inp,
            out,
            mask,
            jacobian_mask=jacobian_mask,
            jacobian_projection=jacobian_projection,
            use_scan=False,
            loss_func=masked_quadratic,
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
    jacobian_mask = model.get_snap_n_mask(1)
    jacobian_projection = make_jacobian_projection(jacobian_mask)
    batch_inp = get_random_batch(B, T, model)
    batch_out = get_random_batch(B, T, model)
    batch_mask = get_random_mask_batch(B, T)

    mapped_rtrl = jax.vmap(
        jtu.Partial(
            rtrl,
            model,
            jacobian_mask=jacobian_mask,
            jacobian_projection=jacobian_projection,
            use_scan=True,
            loss_func=masked_quadratic,
        )
    )
    acc_loss_batch, acc_grads_batch, _ = mapped_rtrl(batch_inp, batch_out, batch_mask)

    passed = True
    for i in range(B):
        inp = batch_inp[i]
        out = batch_out[i]
        mask = batch_mask[i]

        acc_loss, acc_grads, _ = rtrl(
            model,
            inp,
            out,
            mask,
            jacobian_mask=jacobian_mask,
            jacobian_projection=jacobian_projection,
            use_scan=True,
            loss_func=masked_quadratic,
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
