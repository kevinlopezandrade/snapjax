import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax import config
from jaxtyping import PRNGKeyArray

from snapjax.algos import rtrl, rtrl_exact
from snapjax.bptt import bptt
from snapjax.cells.initializers import (
    normal_channels,
    normal_weights,
    sparse_lecun_matrix,
)
from snapjax.cells.readout import IdentityLayer, LinearTanhReadout
from snapjax.cells.rnn import RNNGeneral
from snapjax.cells.stacked import StackedCell
from snapjax.cells.utils import densify_jacobian_mask, make_dense_identity_mask
from snapjax.losses import masked_quadratic
from snapjax.sp_jacrev import make_jacobian_projection
from snapjax.tests.utils import (
    get_random_mask,
    get_random_sequence,
    get_sparse_continous_rnn,
    is_subset_pytree,
    make_dense_jacobian_projection,
)

config.update("jax_enable_x64", True)

ATOL = 1e-12
RTOL = 0.0


def make_sparse_rnn_layer(
    inp_dim: int,
    h_dim: int,
    key: PRNGKeyArray,
    sparsity_level: float,
    sparse_u: bool = True,
    g: float = 0.9,
):
    W_key, U_key = jrandom.split(key, 2)
    if sparsity_level > 0.5:
        W = sparse_lecun_matrix(W_key, N=h_dim, sparsity_level=sparsity_level)

        if inp_dim != h_dim:
            U = normal_channels(U_key, h_dim, inp_dim)
        else:
            U = sparse_lecun_matrix(U_key, N=h_dim, sparsity_level=sparsity_level)
    else:
        W = normal_weights(W_key, N=h_dim, g=g)

        if inp_dim != h_dim:
            U = normal_channels(U_key, h_dim, inp_dim)
        else:
            U = normal_weights(U_key, N=h_dim, g=g)

    return RNNGeneral(W=W, U=U)


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


def test_dense_sparse_snap_2():
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


def test_dense_sparse_snap_3():
    T = 150
    model = get_sparse_continous_rnn(100, 100, sparsity_fraction=0.99)
    jacobian_mask = model.get_snap_n_mask(3)
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


def test_multilayer_sparse():
    T = 50
    L = 8
    H = 32
    keys = jrandom.split(jax.random.PRNGKey(20), L)
    layers = []
    for key in keys[:-1]:
        layer = make_sparse_rnn_layer(inp_dim=H, h_dim=H, sparsity_level=0.99, key=key)

        layers.append(IdentityLayer(layer))

    last_layer_key, readout_key = jrandom.split(keys[-1])
    output_layer = make_sparse_rnn_layer(
        inp_dim=H, h_dim=H, key=last_layer_key, sparsity_level=0.6
    )
    C = normal_weights(key=readout_key, N=H, g=0.9)
    output_layer = LinearTanhReadout(cell=output_layer, C=C)
    layers.append(output_layer)

    model = StackedCell(layers, sparse=True)
    jacobian_mask = model.get_identity_mask()
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    inputs = get_random_sequence(T, model)
    targets = get_random_sequence(T, model)
    mask = jnp.ones((targets.shape[0],))

    loss, acc_grads, _ = rtrl_exact(
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
