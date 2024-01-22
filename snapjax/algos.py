from functools import partial
from typing import Any, Callable, List, Sequence, Tuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import config
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Scalar

from snapjax.cells.base import RTRLCell, RTRLStacked, State, is_rtrl_cell
from snapjax.sp_jacrev import SparseProjection
from snapjax.spp_primitives.primitives import spp_csr_matmul

config.update("jax_numpy_rank_promotion", "raise")


def l2_loss(y: Array, y_hat: Array):
    return jnp.sum((y - y_hat) ** 2)


@jax.jit
def dense_coo_product(D: Array, J_T: BCOO, sp_T: Array):
    orig_shape = J_T.shape

    # TO CSR
    J_T = BCSR.from_bcoo(J_T)
    D_T = D.T

    data = spp_csr_matmul(J_T.data, J_T.indices, J_T.indptr, D_T, sp_T)

    return BCOO((data, sp_T), shape=J_T.shape, indices_sorted=True, unique_indices=True)


@jax.jit
def sparse_matching_addition(A: BCOO, B: BCOO) -> BCOO:
    # Assumes A and B have the same sparsity
    # pattern and that their indices are ordered.
    # then addition is just adding the A.data + B.data
    res = A.data + B.data

    return BCOO(
        (res, A.indices), indices_sorted=True, unique_indices=True, shape=A.shape
    )


def _make_zeros_jacobians_bcco(sp_projection_tree: RTRLStacked):
    # Jacobians are saved as the tranpose jacobians.
    def _sparse_jacobian(leaf: SparseProjection):
        zeros = jnp.zeros(leaf.sparse_def.nse)
        indices = leaf.sparse_def.indices_csc[:, ::-1]  # For the transpose.
        structure = (zeros, indices)
        return BCOO(
            structure,
            shape=leaf.sparse_def.shape[::-1],  # For the transpose.
            # This is guaranteed since csc arse sorted by colum and we transpose.
            indices_sorted=True,
            unique_indices=True,
        )

    zero_jacobians = jtu.tree_map(
        lambda leaf: _sparse_jacobian(leaf),
        sp_projection_tree,
        is_leaf=lambda node: isinstance(node, SparseProjection),
    )

    return zero_jacobians


def _make_zeros_jacobians(model: RTRLStacked):
    cells = eqx.filter(model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)

    def _cell_zero_jacobian(cell: RTRLCell):
        return cell.make_zero_jacobians(cell)

    zero_jacobians = jtu.tree_map(
        lambda cell: _cell_zero_jacobian(cell),
        cells,
        is_leaf=lambda leaf: is_rtrl_cell(leaf),
    )

    return zero_jacobians


def make_zeros_jacobians(model: RTRLStacked, sp_projection_tree: RTRLStacked = None):
    if sp_projection_tree:
        return _make_zeros_jacobians_bcco(sp_projection_tree)
    else:
        return _make_zeros_jacobians(model)


def make_zeros_grads(model: RTRLStacked):
    def zeros_in_leaf(leaf):
        return jnp.zeros(shape=leaf.shape)

    zero_grads = jax.tree_map(zeros_in_leaf, model)

    return zero_grads


@partial(jax.jit, static_argnames=["loss"])
def step_loss(
    model_spatial: RTRLStacked,
    model_rtrl: RTRLStacked,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    y_t: Array,
    sp_projection_tree: RTRLStacked = None,
    loss: Callable[[Array, Array], Array] = l2_loss,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, inmediate_jacobians, y_hat = model.f(
        h_prev, x_t, perturbations, sp_projection_tree
    )

    res = loss(y_t, y_hat)

    return res, (h_t, y_hat, inmediate_jacobians)


@jax.jit
def mask_by_it(i_t: BCOO, j_t: BCOO):
    mask = (jnp.abs(i_t.data) > 0.0).astype(i_t.data.dtype)

    # It does introduce explicit zeros in the BCOO
    # but is fine for our puposes.
    return BCOO(
        (j_t.data * mask, j_t.indices),
        shape=j_t.shape,
        indices_sorted=True,
        unique_indices=True,
    )


@jax.jit
def sparse_update(I_t_T: RTRLCell, dynamics: Array, J_t_prev_T: RTRLCell):
    """
    Returns the I + D*J(t_1) tranposed.
    """

    def _update_rtrl_bcco(i_t: BCOO, j_t_prev_T: BCOO) -> BCOO:
        prod = dense_coo_product(dynamics, j_t_prev_T, j_t_prev_T.indices)
        return sparse_matching_addition(i_t, prod)

    J_t_T = jax.tree_map(
        lambda i_t_T, j_t_prev_T: mask_by_it(
            i_t_T, _update_rtrl_bcco(i_t_T, j_t_prev_T)
        ),
        I_t_T,
        J_t_prev_T,
        is_leaf=lambda node: isinstance(node, BCOO),
    )
    return J_t_T


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
            J_t_T = sparse_update(I_t, dynamics, J_t_prev)
            return J_t_T
        else:
            # The theoretical one from the paper.
            J_t = jax.tree_map(
                lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
                I_t,
                J_t_prev,
            )
            mask = jax.tree_map(
                lambda matrix: (jnp.abs(matrix) > 0.0).astype(matrix.dtype),
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
        if sparse:
            matmul_by_h = jtu.Partial(lambda a, b: (b @ a.T).T, ht_grad)
        else:
            matmul_by_h = jtu.Partial(lambda a, b: a @ b, ht_grad)

        rtrl_cell_jac = jacobians.layers[i].cell
        rtrl_cell_grads = jax.tree_map(
            lambda jacobian: matmul_by_h(jacobian),
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
    jacobians_prev: RTRLStacked,
    h_prev: Sequence[State],
    input: Array,
    target: Array,
    sp_projection_tree: RTRLStacked = None,
    loss: Callable[[Array, Array], Array] = l2_loss,
    use_snap_1: bool = False,
):
    theta_rtrl, theta_spatial = eqx.partition(
        model,
        lambda leaf: is_rtrl_cell(leaf),
        is_leaf=is_rtrl_cell,
    )
    step_loss_and_grad = jax.value_and_grad(
        jtu.Partial(step_loss, loss=loss), argnums=(0, 4), has_aux=True
    )
    perturbations = make_perturbations(theta_rtrl)

    (loss_t, aux), (grads) = step_loss_and_grad(
        theta_spatial,
        theta_rtrl,
        h_prev,
        input,
        perturbations,
        target,
        sp_projection_tree=sp_projection_tree,
    )

    h_t, y_hat, inmediate_jacobians = aux
    spatial_grads, hidden_states_grads = grads

    sparse = True if sp_projection_tree else False
    jacobians = update_jacobians_rtrl(
        jacobians_prev, inmediate_jacobians, use_snap_1=use_snap_1, sparse=sparse
    )

    grads = update_rtrl_cells_grads(
        spatial_grads, hidden_states_grads, jacobians, sparse=sparse
    )

    # Reshape the flattened gradients of the RTRL cells.
    if sparse:
        grads = jtu.tree_map(lambda mat, grad: grad.reshape(mat.shape), model, grads)

    return h_t, grads, jacobians, loss_t, y_hat


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
    sp_projection_tree: RTRLStacked = None,
    loss: Callable[[Array, Array], Scalar] = l2_loss,
    use_scan: bool = True,
    use_snap_1: bool = False,
):
    def forward_repack(carry, data):
        input, target = data
        h_prev, acc_grads, jacobians_prev, acc_loss = carry

        out = forward_rtrl(
            model,
            jacobians_prev,
            h_prev,
            input,
            target,
            sp_projection_tree=sp_projection_tree,
            use_snap_1=use_snap_1,
            loss=loss,
        )
        h_t, grads, jacobians_t, loss_t, y_hat = out
        acc_loss = acc_loss + loss_t
        acc_grads = jtu.tree_map(
            lambda acc_grads, grads: acc_grads + grads, acc_grads, grads
        )

        return (h_t, acc_grads, jacobians_t, acc_loss), y_hat

    h_init: Sequence[State] = init_state(model)
    acc_grads: RTRLStacked = make_zeros_grads(model)
    zero_jacobians: RTRLStacked = make_zeros_jacobians(
        model, sp_projection_tree=sp_projection_tree
    )
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
