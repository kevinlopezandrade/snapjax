from functools import partial
from typing import Any, List, Sequence, Tuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import config
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array

from snapjax.cells.base import RTRLCell, RTRLStacked, State
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
    orig_shape = J.shape
    J = J.transpose()

    # TO CSR
    J = BCSR.from_bcoo(J)
    D = D.T
    # Transpose also the sparsity pattern.
    sp_T = sp.at[:, 0].set(sp[:, 1])
    sp_T = sp_T.at[:, 1].set(sp[:, 0])

    data = spp_csr_matmul_jax(J.data, J.indices, J.indptr, D, sp_T)

    # No need to do the transpose since, sp is already in the shape we want.
    return BCOO((data, sp), shape=orig_shape, indices_sorted=True, unique_indices=True)


@jax.jit
def sparse_matching_addition(A: BCOO, B: BCOO) -> BCOO:
    # Assumes A and B have the same sparsity
    # pattern and that their indices are ordered.
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
                indices_sorted=True,  # This is guaranteed in if sparse_def
                unique_indices=True,  # is properly constructed.
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
    cells = eqx.filter(model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)

    def _cell_zero_jacobian(cell: RTRLCell):
        return type(cell).make_zero_jacobians(cell)

    zero_jacobians = jtu.tree_map(
        lambda cell: _cell_zero_jacobian(cell),
        cells,
        is_leaf=lambda leaf: is_rtrl_cell(leaf),
    )

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


@partial(jax.jit, static_argnums=6)
def step_loss(
    model_spatial: RTRLStacked,
    model_rtrl: RTRLStacked,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    y_t: Array,
    sparse: bool = False,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, inmediate_jacobians, y_hat = model.f(h_prev, x_t, perturbations, sparse)

    diff = (y_t - y_hat) ** 2

    return jnp.sum(diff), (h_t, y_hat, inmediate_jacobians)


@jax.jit
def mask_by_it(i_t: BCOO, j_t: BCOO):
    mask = (jnp.abs(i_t.data) > 0.0).astype(jnp.float32)

    # It does introduce explicit zeros in the BCOO
    # but is fine for our puposes.
    return BCOO(
        (j_t.data * mask, j_t.indices),
        shape=j_t.shape,
        indices_sorted=True,
        unique_indices=True,
    )


@jax.jit
def sparse_update(I_t: RTRLCell, dynamics: Array, J_t_prev: RTRLCell):
    def _update_rtrl_bcco(i_t: BCOO, j_t_prev: BCOO) -> BCOO:
        prod = dense_coo_product_jax(dynamics, j_t_prev, j_t_prev.indices)
        return sparse_matching_addition(i_t, prod)

    J_t = jax.tree_map(
        lambda i_t, j_t_prev: mask_by_it(i_t, _update_rtrl_bcco(i_t, j_t_prev)),
        I_t,
        J_t_prev,
        is_leaf=lambda node: isinstance(node, BCOO),
    )
    return J_t


@partial(jax.jit, static_argnums=(3, 4))
def update_cell_jacobians(
    I_t: RTRLCell,
    dynamics: Array,
    J_t_prev: RTRLCell,
    use_snap_1: bool = False,
    sparse: bool = False,
):
    # RTRL
    if use_snap_1:
        if sparse:
            J_t = sparse_update(I_t, dynamics, J_t_prev)
            return J_t
        else:
            # The theoretical one from the paper.
            J_t = jax.tree_map(
                lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
                I_t,
                J_t_prev,
            )
            mask = jax.tree_map(
                lambda matrix: (jnp.abs(matrix) > 0.0).astype(jnp.float32),
                I_t,
            )
            J_t = jax.tree_map(lambda mask, j_t: mask * j_t, mask, J_t)
            return J_t

    else:
        J_t = jax.tree_map(
            lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
            I_t,
            J_t_prev,
        )
        return J_t


@partial(jax.jit, static_argnums=(2, 3))
def update_jacobians_rtrl(
    jacobians_prev: RTRLStacked,
    inmediate_jacobians: List[Tuple[RTRLCell, Array]],
    use_snap_1: bool = False,
    sparse: bool = False,
):
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


@partial(jax.jit, static_argnums=3)
def update_rtrl_cells_grads(
    grads: RTRLStacked,
    hidden_states_grads: List[Array],
    jacobians: RTRLStacked,
    sparse: bool = False,
):
    def _leaf_function(is_sparse):
        if is_sparse:
            _is_leaf_func = lambda node: isinstance(node, BCOO)
            return _is_leaf_func
        else:
            return None

    for i in range(grads.num_layers):
        ht_grad = hidden_states_grads[i]
        rtrl_cell_jac = jacobians.layers[i].cell
        rtrl_cell_grads = jax.tree_map(
            lambda jacobian: ht_grad @ jacobian,
            rtrl_cell_jac,
            is_leaf=_leaf_function(sparse),
        )
        grads = eqx.tree_at(
            lambda grads: grads.layers[i].cell,
            grads,
            rtrl_cell_grads,
            is_leaf=lambda x: x is None,
        )

    return grads


def make_perturbations(model: RTRLStacked):
    L = model.num_layers
    perturbations = [None] * L
    for i in range(L):
        cell = model.layers[i].cell
        perturbations[i] = jnp.zeros(cell.hidden_size)

    return tuple(perturbations)


# This should be a pure function.
def forward_rtrl(
    model: RTRLStacked,
    acc_grads: RTRLStacked,
    jacobians_prev: RTRLStacked,
    h_prev: Array,
    input: Array,
    target: Array,
    use_snap_1: bool = False,
    sparse: bool = False,
):
    theta_rtrl, theta_spatial = eqx.partition(
        model,
        lambda leaf: is_rtrl_cell(leaf),
        is_leaf=is_rtrl_cell,
    )
    step_loss_and_grad = jax.value_and_grad(step_loss, argnums=(0, 4), has_aux=True)
    perturbations = make_perturbations(theta_rtrl)

    (loss_t, aux), (grads) = step_loss_and_grad(
        theta_spatial, theta_rtrl, h_prev, input, perturbations, target, sparse=sparse
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
        lambda acc_grad, grad: acc_grad + grad.reshape(acc_grad.shape),
        acc_grads,
        grads,
    )

    return h_t, acc_grads, jacobians, loss_t, y_hat


def init_state(model: RTRLStacked):
    states: Sequence[State] = []
    for layer in model.layers:
        cell = layer.cell
        states.append(type(cell).init_state(cell))

    return tuple(states)


def rtrl(
    model: RTRLStacked,
    inputs: Array,
    targets: Array,
    use_scan: bool = True,
    use_snap_1: bool = False,
    sparse: bool = False,
):
    def forward_repack(carry, data):
        input, target = data
        h_prev, acc_grads, jacobians_prev, acc_loss = carry

        out = forward_rtrl(
            model,
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

    h_init: Sequence[State] = init_state(model)
    acc_grads: RTRLStacked = make_zeros_grads(model)
    zero_jacobians: RTRLStacked = make_zeros_jacobians(model, sparse=sparse)
    acc_loss = 0.0

    if use_scan:
        carry_T, _ = jax.lax.scan(
            forward_repack,
            init=(h_init, acc_grads, zero_jacobians, acc_loss),
            xs=(inputs, targets),
        )
    else:
        carry = (h_init, acc_grads, zero_jacobians, acc_loss)
        T = inputs.shape[0]
        y_hats = [None] * T
        for i in range(inputs.shape[0]):
            carry, y_hat = forward_repack(carry, (inputs[i], targets[i]))
            y_hats[i] = y_hat

        y_hats = jnp.stack(y_hats)
        carry_T = carry

    h_T, acc_grads, jacobians_T, acc_loss = carry_T

    return acc_loss, acc_grads, jacobians_T


def forward_sequence(model: RTRLStacked, inputs: Array, use_scan: bool = True):
    hidden_state = init_state(model)
    perturbations = jnp.zeros(shape=(model.num_layers, model.d_inp))

    def f_repack(state: Sequence[State], input: Array):
        h, _, out = model.f(state, input, perturbations)
        return h, out

    # Ful forward pass over the sequence
    # Add use_scan to test if they actually differ now.
    if use_scan:
        _, out = jax.lax.scan(
            lambda carry, input: f_repack(carry, input),
            init=hidden_state,
            xs=inputs,
        )
    else:
        carry = hidden_state
        y_hats = []
        for i in range(inputs.shape[0]):
            carry, y_hat = f_repack(carry, inputs[i])
            y_hats.append(y_hat)

        y_hats = jnp.stack(y_hats)
        out = y_hats

    return out


def loss_func(model: RTRLStacked, inputs: Array, targets: Array, use_scan: bool = True):
    pred = forward_sequence(model, inputs, use_scan)
    errors = jnp.sum((pred - targets) ** 2, axis=1)
    return jnp.sum(errors)


def bptt(model: RTRLStacked, inputs: Array, targets: Array, use_scan: bool = True):
    loss, grads = jax.value_and_grad(loss_func, argnums=0)(
        model, inputs, targets, use_scan
    )

    return loss, grads
