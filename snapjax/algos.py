from typing import Any, Callable, List, Tuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import config
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree

from snapjax.cells.base import RTRLCell, RTRLLayer, RTRLStacked
from snapjax.sp_jacrev import SparseProjection
from snapjax.spp_primitives.primitives import spp_csr_matmul_jax

config.update("jax_numpy_rank_promotion", "raise")


def is_rtrl_cell(node: Any):
    if isinstance(node, RTRLCell):
        return True
    else:
        return False


@jax.jit
def dense_coo_product_jax(D: Array, J: BCOO, sp: Array):
    J = J.transpose()

    # TO CSR
    J = BCSR.from_bcoo(J)
    D = D.T
    data = spp_csr_matmul_jax(J.data, J.indices, J.indptr, D, sp)

    return BCOO(
        (data, sp), shape=J.shape, indices_sorted=True, unique_indices=True
    ).transpose()


@jax.jit
def sparse_matching_addition(A: BCOO, B: BCOO) -> BCOO:
    # Assumes A and B have the same sparsity
    # pattern and that their indices are ordered. (Not strictly necessary)
    # then addition is just adding the A.data + B.data
    res = A.data + B.data

    return BCOO(
        (res, A.indices), indices_sorted=True, unique_indices=True, shape=A.shape
    )


def _make_zeros_jacobians_bcco(model: RTRLStacked):
    # Make jacobians only for RTRLCells
    zero_jacobians = eqx.filter(
        model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell
    )

    for i in range(model.num_layers):
        sp_projection = model.layers[i].cell_sp_projection
        zeros_jac = jtu.tree_map(
            lambda sp: BCOO(
                (jnp.zeros(sp.sparse_def.nse), sp.sparse_def.indices),
                shape=sp.sparse_def.shape,
                indices_sorted=True,
                unique_indices=True,
            ),
            sp_projection,
            is_leaf=lambda node: isinstance(node, SparseProjection),
        )
        zero_jacobians = eqx.tree_at(
            lambda model: model.layers[i].cell, zero_jacobians, zeros_jac
        )

    zero_jacobians = cast(RTRLStacked, zero_jacobians)

    return zero_jacobians


def _make_zeros_jacobians(model: RTRLStacked):
    # Make jacobians only for RTRLCells
    zero_jacobians = eqx.filter(
        model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell
    )

    def zeros_in_leaf(leaf):
        return jnp.zeros(shape=(model.hidden_size, *leaf.shape))

    zero_jacobians = jax.tree_map(zeros_in_leaf, zero_jacobians)
    zero_jacobians = cast(RTRLStacked, zero_jacobians)

    return zero_jacobians


def make_zeros_jacobians(model: RTRLStacked, sparse: bool = False):
    if sparse:
        return _make_zeros_jacobians_bcco(model)
    else:
        return _make_zeros_jacobians(model)


def make_zeros_grads(model: RTRLStacked):
    def zeros_in_leaf(leaf):
        return jnp.zeros(shape=leaf.shape)

    zero_grads = jax.tree_map(zeros_in_leaf, model)

    return zero_grads


def step_loss(
    model_spatial: RTRLStacked,
    model_rtrl: RTRLStacked,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    y_t: Array,
):
    print("Compiling step_loss")
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, y_hat, inmediate_jacobians = model.f(h_prev, x_t, perturbations)

    diff = (y_t - y_hat) ** 2

    return jnp.sum(diff), (h_t, y_hat, inmediate_jacobians)


@eqx.filter_jit
def update_cell_jacobians(
    I_t: RTRLCell,
    dynamics: Array,
    J_t_prev: RTRLCell,
    use_snap_1: bool = False,
    sparse: bool = False,
):
    print("Compiling update_cell_jacobians")
    # RTRL
    if not sparse:
        J_t = jax.tree_map(
            lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
            I_t,
            J_t_prev,
        )
        if use_snap_1:
            # The real and theoretical one.
            mask = jax.tree_map(
                lambda matrix: (jnp.abs(matrix) > 0.0).astype(jnp.float32),
                I_t,
            )
            J_t = jax.tree_map(lambda mask, j_t: mask * j_t, mask, J_t)
            return J_t
        else:
            return J_t
    else:
        # Using sparse already implies using the snap-1 algorithm.
        def _update_rtrl_bcco(i_t: BCOO, j_t_prev: BCOO) -> BCOO:
            prod = dense_coo_product_jax(dynamics, j_t_prev, i_t.indices)
            return sparse_matching_addition(i_t, prod)

        J_t = jax.tree_map(
            lambda i_t, j_t_prev: _update_rtrl_bcco(i_t, j_t_prev),
            I_t,
            J_t_prev,
            is_leaf=lambda node: isinstance(node, BCOO),
        )

        return J_t


def update_jacobians_rtrl(
    jacobians_prev: RTRLStacked,
    inmediate_jacobians: List[Tuple[RTRLCell, Array]],
    use_snap_1: bool = False,
    sparse: bool = False,
):
    print("Compiling update_jacobians_rtrl")
    # Jax will do loop unrolling here, but number of layers is not that big
    # so it will be fine.
    jacobians = jacobians_prev
    for i in range(jacobians_prev.num_layers):
        I_t, D_t = inmediate_jacobians[i]
        J_t_prev = jacobians.layers[i].cell
        J_t = update_cell_jacobians(
            I_t, D_t, J_t_prev, use_snap_1=use_snap_1, sparse=sparse
        )

        jacobians = eqx.tree_at(lambda model: model.layers[i].cell, jacobians, J_t)

    return jacobians


def update_rtrl_cells_grads(
    grads: RTRLStacked,
    hidden_states_grads: List[Array],
    jacobians: RTRLStacked,
    sparse: bool = False,
):
    print("Compiling update_cells_grads")

    def _leaf_function(sparse):
        if sparse:
            _is_leaf_func = lambda node: isinstance(node, BCOO)
            return _is_leaf_func
        else:
            return None

    for i in range(grads.num_layers):
        ht_grad = hidden_states_grads[i]
        rtrl_cell_grads = jax.tree_map(
            lambda jacobian: ht_grad @ jacobian,
            jacobians.layers[i].cell,
            is_leaf=_leaf_function(sparse),
        )
        grads = eqx.tree_at(
            lambda grads: grads.layers[i].cell,
            grads,
            rtrl_cell_grads,
            is_leaf=lambda x: x is None,
        )

    return grads


# This should be a pure function.
def forward_rtrl(
    theta_spatial: RTRLStacked,
    theta_rtrl: RTRLStacked,
    acc_grads: RTRLStacked,
    jacobians_prev: RTRLStacked,
    h_prev: Array,
    input: Array,
    target: Array,
    use_snap_1: bool = False,
    sparse: bool = False,
):
    print("Compiling forward_rtrl")
    step_loss_and_grad = jax.value_and_grad(step_loss, argnums=(0, 4), has_aux=True)
    perturbations = jnp.zeros(shape=(theta_rtrl.num_layers, theta_rtrl.hidden_size))

    (loss_t, aux), (grads) = step_loss_and_grad(
        theta_spatial,
        theta_rtrl,
        h_prev,
        input,
        perturbations,
        target,
    )

    h_t, y_hat, inmediate_jacobians = aux
    spatial_grads, hidden_states_grads = grads

    jacobians = update_jacobians_rtrl(
        jacobians_prev, inmediate_jacobians, use_snap_1=use_snap_1, sparse=sparse
    )

    grads = update_rtrl_cells_grads(
        spatial_grads, hidden_states_grads, jacobians, sparse=sparse
    )

    # Reshape is needed in the addition since I flattened the arrays
    # to use the new primitive with a BCOO matrix of only 2 dimensions.
    # For non-sparse reshapes does not do anything.
    acc_grads = jax.tree_map(
        lambda acc_grad, grad: acc_grad + grad.reshape(acc_grad.shape), acc_grads, grads
    )

    return h_t, acc_grads, jacobians, loss_t, y_hat


def rtrl(
    model: RTRLStacked,
    inputs: Array,
    targets: Array,
    use_snap_1: bool = False,
    sparse: bool = True,
    use_scan: bool = True,
):
    model_rtrl, model_spatial = eqx.partition(
        model,
        lambda leaf: is_rtrl_cell(leaf),
        is_leaf=is_rtrl_cell,
    )

    def forward_repack(carry, data):
        print("Compiling forward_repack")
        input, target = data
        h_prev, acc_grads, jacobians_prev, acc_loss = carry

        out = forward_rtrl(
            model_spatial,
            model_rtrl,
            acc_grads,
            jacobians_prev,
            h_prev,
            input,
            target,
            use_snap_1=use_snap_1,
            sparse=sparse,
        )
        h_t, acc_grads, jacobians_t, loss_t, y_hat = out
        acc_loss = acc_loss + loss_t

        return (h_t, acc_grads, jacobians_t, acc_loss), y_hat

    h_init = jnp.zeros(shape=(model.num_layers, model.hidden_size))
    acc_grads = make_zeros_grads(model)
    zero_jacobians = make_zeros_jacobians(model, sparse)
    acc_loss = 0.0

    if use_scan:
        carry_T, _ = jax.lax.scan(
            forward_repack,
            init=(h_init, acc_grads, zero_jacobians, acc_loss),
            xs=(inputs, targets),
        )
    else:
        carry = (h_init, acc_grads, zero_jacobians, acc_loss)
        y_hats = []
        for i in range(inputs.shape[0]):
            carry, y_hat = forward_repack(carry, (inputs[i], targets[i]))
            y_hats.append(y_hat)

        y_hats = jnp.stack(y_hats)
        carry_T = carry

    h_T, acc_grads, jacobians_T, acc_loss = carry_T

    return acc_loss, acc_grads, jacobians_T


def forward_sequence(model: RTRLStacked, inputs: Array):
    hidden_state = jnp.zeros(shape=(model.num_layers, model.hidden_size))
    perturbations = jnp.zeros(shape=(model.num_layers, model.hidden_size))

    def f_repack(h: Array, input: Array):
        h, out, _ = model(h, input, perturbations)
        return h, out

    # Ful forward pass over the sequence
    _, out = jax.lax.scan(
        lambda carry, input: f_repack(carry, input),
        init=hidden_state,
        xs=inputs,
    )

    return out


def loss_func(model: RTRLStacked, inputs: Array, targets: Array):
    pred = forward_sequence(model, inputs)
    errors = jnp.sum((pred - targets) ** 2, axis=1)
    return jnp.sum(errors)


def bptt(model: RTRLStacked, inputs: Array, targets: Array):
    loss, grads = eqx.filter_value_and_grad(loss_func)(model, inputs, targets)

    return loss, grads
