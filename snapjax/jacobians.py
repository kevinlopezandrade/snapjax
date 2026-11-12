"""Sparse and dense trace updates shared by recurrent cells."""

import jax
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array

from snapjax.cells.base import RTRLCell
from snapjax.sp_jacrev import Mask, SparseMask, standard_jacobian
from snapjax.spp_primitives.primitives import spp_csr_matmul


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
def compute_masked_jacobian(
    jacobian_mask: RTRLCell, I_t: RTRLCell, dynamics: Array, J_t_prev: RTRLCell
):
    """
    Note that I_t and J_t_prev must be actually the transposed ones if they are in BCOO.
    Returns the I + D*J(t_1) tranposed if in BCOO.
    """

    def _update(i_t: BCOO | Array, j_t_prev: BCOO | Array, j_mask: Mask | SparseMask):
        if isinstance(j_mask, SparseMask):
            # j_t_prev comes as the transpose jacobian.
            # NOTE: Actually we can ignore the j_mask, since
            # the compressed was computed using j_mask as its
            # sparsity pattern. And we avoid transposing and sorting.
            # We leave it just for consistency in the api.
            prod = dense_coo_product(dynamics, j_t_prev, j_t_prev.indices)
            res = sparse_matching_addition(i_t, prod)
            return res
        else:
            return standard_jacobian(j_mask.jacobian_mask) * (
                standard_jacobian(i_t) + dynamics @ standard_jacobian(j_t_prev)
            )

    J_t = jtu.tree_map(
        lambda i_t, j_t_prev, j_mask: _update(i_t, j_t_prev, j_mask),
        I_t,
        J_t_prev,
        jacobian_mask,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    return J_t


@jax.jit
def update_cell_jacobians(
    I_t: RTRLCell,
    dynamics: Array,
    J_t_prev: RTRLCell,
    jacobian_cell_mask: RTRLCell | None = None,
):
    # Apply masks to the jacobians.
    if jacobian_cell_mask:
        J_t = compute_masked_jacobian(jacobian_cell_mask, I_t, dynamics, J_t_prev)
        return J_t
    else:
        J_t = jtu.tree_map(
            lambda i_t, j_t_prev: standard_jacobian(i_t)
            + dynamics @ standard_jacobian(j_t_prev),
            I_t,
            J_t_prev,
        )
        return J_t
