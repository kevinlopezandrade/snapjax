from functools import partial
from typing import Callable, List, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import config
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Scalar

from snapjax.cells.base import RTRLCell, RTRLLayer, RTRLStacked, State, is_rtrl_cell
from snapjax.losses import l2
from snapjax.sp_jacrev import SparseProjection
from snapjax.spp_primitives.primitives import spp_csr_matmul

config.update("jax_numpy_rank_promotion", "raise")


def make_zeros_jacobians_sp(sp_projection_tree: RTRLStacked):
    # Jacobians are saved as the tranpose jacobians.
    def _sparse_jacobian(leaf: SparseProjection):
        zeros = jnp.zeros(leaf.sparse_def.nse)
        indices = leaf.sparse_def.indices_csc[:, ::-1]  # For the transpose.
        structure = (zeros, indices)
        return BCOO(
            structure,
            shape=leaf.sparse_def.shape[::-1],  # For the transpose.
            # This is guaranteed since csc is sorted by colum and we transpose.
            indices_sorted=True,
            unique_indices=True,
        )

    zero_jacobians = jtu.tree_map(
        lambda leaf: _sparse_jacobian(leaf),
        sp_projection_tree,
        is_leaf=lambda node: isinstance(node, SparseProjection),
    )

    return zero_jacobians


def make_zeros_jacobians(model: RTRLStacked):
    cells = eqx.filter(model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)

    def _cell_zero_jacobian(cell: RTRLCell):
        return cell.make_zero_jacobians(cell)

    zero_jacobians = jtu.tree_map(
        lambda cell: _cell_zero_jacobian(cell),
        cells,
        is_leaf=lambda leaf: is_rtrl_cell(leaf),
    )

    return zero_jacobians


def make_zeros_grads(model: RTRLStacked):
    def zeros_in_leaf(leaf):
        return jnp.zeros(shape=leaf.shape)

    zero_grads = jax.tree_map(zeros_in_leaf, model)

    return zero_grads


def make_perturbations(model: RTRLStacked):
    perturbations = []
    cell_index = 0
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = model.layers[cell_index].cell
            perturbations.append(jnp.zeros(cell.hidden_size))
            cell_index += 1

    return tuple(perturbations)


@jax.jit
def dense_coo_product(D: Array, J_T: BCOO, sp_T: Array):
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
def sparse_update(I_t: RTRLCell, dynamics: Array, J_t_prev: RTRLCell):
    """
    Note that I_t and J_t_prev must be actually the transposed ones.
    Returns the I + D*J(t_1) tranposed.
    """

    def _update_rtrl_bcco(i_t: BCOO, j_t_prev: BCOO) -> BCOO:
        prod = dense_coo_product(dynamics, j_t_prev, j_t_prev.indices)
        return sparse_matching_addition(i_t, prod)

    J_t = jax.tree_map(
        lambda i_t, j_t_prev: mask_by_it(i_t, _update_rtrl_bcco(i_t, j_t_prev)),
        I_t,
        J_t_prev,
        is_leaf=lambda node: isinstance(node, BCOO),
    )
    return J_t


@partial(jax.jit, static_argnums=(4, 5))
def update_cell_jacobians(
    I_t: RTRLCell,
    dynamics: Array,
    J_t_prev: RTRLCell,
    jacobian_cell_mask: RTRLCell | None = None,
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

    # Used for snap-n
    elif jacobian_cell_mask:
        J_t = jax.tree_map(
            lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev, I_t, J_t_prev
        )
        J_t = jax.tree_map(lambda mask, j_t: mask * j_t, jacobian_cell_mask, J_t)

        return J_t
    else:
        J_t = jax.tree_map(
            lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
            I_t,
            J_t_prev,
        )
        return J_t


@partial(jax.jit, static_argnums=(3, 4))
def update_jacobians_rtrl(
    jacobians_prev: RTRLStacked,
    inmediate_jacobians: List[Tuple[RTRLCell, Array]],
    jacobian_mask: RTRLStacked | None = None,
    use_snap_1: bool = False,
    sparse: bool = False,
):
    # Jax will do loop unrolling here, but number of layers is not that big
    # so it will be fine.
    jacobians = jacobians_prev
    cell_index = 0
    for layer in jacobians_prev.layers:
        if isinstance(layer, RTRLLayer):
            I_t, D_t = inmediate_jacobians[cell_index]
            J_t_prev = jacobians.layers[cell_index].cell
            if jacobian_mask:
                jacobian_cell_mask = jacobian_mask.layers[cell_index].cell
            else:
                jacobian_cell_mask = None

            J_t = update_cell_jacobians(
                I_t,
                D_t,
                J_t_prev,
                use_snap_1=use_snap_1,
                sparse=sparse,
                jacobian_cell_mask=jacobian_cell_mask,
            )

            jacobians = eqx.tree_at(
                lambda model: model.layers[cell_index].cell, jacobians, J_t
            )

            cell_index += 1

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

    cell_index = 0
    for layer in grads.layers:
        if isinstance(layer, RTRLLayer):
            ht_grad = hidden_states_grads[cell_index]
            if sparse:
                matmul_by_h = jtu.Partial(lambda a, b: (b @ a.T).T, ht_grad)
            else:
                matmul_by_h = jtu.Partial(lambda a, b: a @ b, ht_grad)

            rtrl_cell_jac = jacobians.layers[cell_index].cell
            rtrl_cell_grads = jax.tree_map(
                lambda jacobian: matmul_by_h(jacobian),
                rtrl_cell_jac,
                is_leaf=_leaf_function(sparse),
            )
            grads = eqx.tree_at(
                lambda grads: grads.layers[cell_index].cell,
                grads,
                rtrl_cell_grads,
                is_leaf=lambda x: x is None,
            )

            cell_index += 1

    return grads


@partial(jax.jit, static_argnames=["loss_func"])
def step_loss(
    model_spatial: RTRLStacked,
    model_rtrl: RTRLStacked,
    h_prev: Array,
    x_t: Array,
    perturbations: Array,
    y_t: Array,
    mask: float,
    sp_projection_tree: RTRLStacked | None = None,
    loss_func: Callable[[Array, Array, float], Array] = l2,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, inmediate_jacobians, y_hat = model.f(
        h_prev, x_t, perturbations, sp_projection_tree
    )

    res = loss_func(y_t, y_hat, mask)

    return res, (h_t, y_hat, inmediate_jacobians)


# This should be a pure function.
def forward_rtrl(
    model: RTRLStacked,
    jacobians_prev: RTRLStacked,
    h_prev: Sequence[State],
    input: Array,
    target: Array,
    mask: float = 1.0,
    sp_projection_tree: RTRLStacked | None = None,
    jacobian_mask: RTRLStacked | None = None,
    loss_func: Callable[[Array, Array, float], Array] = l2,
    use_snap_1: bool = False,
):
    theta_rtrl, theta_spatial = eqx.partition(
        model,
        lambda leaf: is_rtrl_cell(leaf),
        is_leaf=is_rtrl_cell,
    )
    step_loss_and_grad = jax.value_and_grad(
        jtu.Partial(step_loss, loss_func=loss_func), argnums=(0, 4), has_aux=True
    )
    perturbations = make_perturbations(theta_rtrl)

    (loss_t, aux), (grads) = step_loss_and_grad(
        theta_spatial,
        theta_rtrl,
        h_prev,
        input,
        perturbations,
        target,
        mask,
        sp_projection_tree=sp_projection_tree,
    )

    h_t, y_hat, inmediate_jacobians = aux
    spatial_grads, hidden_states_grads = grads

    sparse = True if sp_projection_tree else False
    jacobians = update_jacobians_rtrl(
        jacobians_prev,
        inmediate_jacobians,
        use_snap_1=use_snap_1,
        sparse=sparse,
        jacobian_mask=jacobian_mask,
    )

    grads = update_rtrl_cells_grads(
        spatial_grads, hidden_states_grads, jacobians, sparse=sparse
    )

    # Reshape the flattened gradients of the RTRL cells.
    if sparse:
        grads = jtu.tree_map(lambda mat, grad: grad.reshape(mat.shape), model, grads)

    return h_t, grads, jacobians, loss_t, y_hat


def make_init_state(model: RTRLStacked):
    states: Sequence[State] = []
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = layer.cell
            states.append(type(cell).init_state(cell))

    return tuple(states)


def rtrl(
    model: RTRLStacked,
    inputs: Array,
    targets: Array,
    mask: Array | None = None,
    sp_projection_tree: RTRLStacked | None = None,
    jacobian_mask: RTRLStacked | None = None,
    loss_func: Callable[[Array, Array, float], Scalar] = l2,
    use_scan: bool = True,
    use_snap_1: bool = False,
):
    if mask is not None:
        n_non_masked = mask.sum()
        _loss_func = lambda y, y_hat, mask: (1 / n_non_masked) * loss_func(
            y, y_hat, mask
        )
    else:
        mask = jnp.ones(targets.shape[0])
        _loss_func = loss_func

    def forward_repack(carry, data):
        input, target, mask = data
        h_prev, acc_grads, jacobians_prev, acc_loss = carry

        out = forward_rtrl(
            model,
            jacobians_prev,
            h_prev,
            input,
            target,
            mask,
            sp_projection_tree=sp_projection_tree,
            jacobian_mask=jacobian_mask,
            use_snap_1=use_snap_1,
            loss_func=_loss_func,
        )
        h_t, grads, jacobians_t, loss_t, y_hat = out
        acc_loss = acc_loss + loss_t
        acc_grads = jtu.tree_map(
            lambda acc_grads, grads: acc_grads + grads, acc_grads, grads
        )

        return (h_t, acc_grads, jacobians_t, acc_loss), y_hat

    h_init: Sequence[State] = make_init_state(model)
    acc_grads: RTRLStacked = make_zeros_grads(model)

    if sp_projection_tree:
        zero_jacobians = make_zeros_jacobians_sp(sp_projection_tree)
    else:
        zero_jacobians = make_zeros_jacobians(model)

    acc_loss = 0.0

    if use_scan:
        carry_T, _ = jax.lax.scan(
            forward_repack,
            init=(h_init, acc_grads, zero_jacobians, acc_loss),
            xs=(inputs, targets, mask),
        )
    else:
        carry = (h_init, acc_grads, zero_jacobians, acc_loss)
        T = inputs.shape[0]
        y_hats = [None] * T
        for i in range(inputs.shape[0]):
            carry, y_hat = forward_repack(carry, (inputs[i], targets[i], mask[i]))
            y_hats[i] = y_hat

        y_hats = jnp.stack(y_hats)
        carry_T = carry

    h_T, acc_grads, jacobians_T, acc_loss = carry_T

    return acc_loss, acc_grads
