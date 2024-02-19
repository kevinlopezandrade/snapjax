import jax.numpy as jnp
import jax.tree_util as jtu
from jax import config

from snapjax.algos import rtrl
from snapjax.bptt import bptt
from snapjax.cells.utils import densify_jacobian_mask
from snapjax.losses import masked_quadratic
from snapjax.sp_jacrev import make_jacobian_projection
from snapjax.tests.utils import (
    get_random_mask,
    get_random_sequence,
    get_sparse_continous_rnn,
    is_subset_pytree,
    make_dense_identity_mask,
    make_dense_jacobian_projection,
)

config.update("jax_enable_x64", True)

ATOL = 1e-12
RTOL = 0.0


def test_rtrl_bptt_sparse_weights():
    T = 100
    model = get_sparse_continous_rnn(100, 100, sparsity_fraction=0.98)
    jacobian_mask = model.get_snap_n_mask(1)
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    # Don't mask the jacobians..
    jacobian_mask = densify_jacobian_mask(jacobian_mask)
    jacobian_mask = make_dense_identity_mask(jacobian_mask)

    # Forzes to compute the full gradients, not the compressed ones.
    jacobian_projection = make_dense_jacobian_projection(jacobian_projection)

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
        sparse_model=True,
    )

    assert jnp.allclose(loss, loss_bptt)

    print("Comparing BPTT and RTRL with sparse weight matrices.")
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


def test_snap_n_indices():
    model = get_sparse_continous_rnn(100, 100, sparsity_fraction=0.90)
    mask_snap_1 = model.get_snap_n_mask(1)
    mask_snap_2 = model.get_snap_n_mask(2)

    assert is_subset_pytree(mask_snap_1, mask_snap_2)


def test_dense_sparse_snap_n():
    T = 150
    model = get_sparse_continous_rnn(100, 100, sparsity_fraction=0.90)
    jacobian_mask = model.get_snap_n_mask(2)
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = jnp.ones((targets.shape[0],))

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

    jacobian_mask_dense = densify_jacobian_mask(jacobian_mask)
    jacobian_projection_dense = make_dense_jacobian_projection(jacobian_projection)
    loss_no_sp, acc_grads_no_sp, _ = rtrl(
        model,
        inputs,
        targets,
        mask,
        jacobian_mask=jacobian_mask_dense,
        jacobian_projection=jacobian_projection_dense,
        use_scan=False,
        loss_func=masked_quadratic,
    )

    assert jnp.allclose(loss, loss_no_sp)

    print("Comparing BPTT and RTRL with sparse weight matrices.")
    passed = True
    for (key_a, leaf_a), (key_b, leaf_b) in zip(
        jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_no_sp)
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
