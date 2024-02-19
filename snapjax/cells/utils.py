from typing import TypeVar

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array

from snapjax.cells.base import RTRLStacked
from snapjax.sp_jacrev import Mask, SparseMask

_T = TypeVar("_T")


def construct_snap_n_mask_bcoo(W: BCOO, n: int) -> SparseMask:
    """
    Given a sparse connectivity matrix, compute its influence on future hidden
    states, to later mask the jacobian. This only works properly if a weight in
    h(t) = f(W, h(t-1), x(t)), affects only one output unit in just one timestep,
    if it affects more as in the GRU, it will not work properly. The variant of the
    GRU we use, fits this condition so we are safe here.
    """
    h = jnp.ones(W.shape[1])
    _W = BCOO(
        (jnp.ones(W.nse), W.indices),
        shape=W.shape,
        indices_sorted=W.indices_sorted,
        unique_indices=W.unique_indices,
    )

    def f(W: BCOO, h: Array, n: int):
        for i in range(n):
            h = W @ jnp.ones(h.shape[0]) + (W @ h)

        return h

    gradient = jsparse.jacrev(f)(_W, h, n).data
    mask = (gradient > 0.0).astype(jnp.float32)
    mask = BCOO.fromdense(mask)

    return SparseMask(mask.indices, mask.shape, orig_shape=mask.shape)


def construct_snap_n_mask(W: Array, n: int) -> SparseMask:
    if W.ndim == 2:
        _W = jnp.ones(W.shape)
        h = jnp.ones(W.shape[1])

        def f(W: Array, h: Array):
            for i in range(n):
                h = W @ h

            return h

        mask = (jax.jacrev(f)(_W, h) > 0.0).astype(jnp.float32)
        orig_shape = mask.shape

        mask = mask.reshape(_W.shape[0], -1)
        mask = BCOO.fromdense(mask)
        return SparseMask(mask.indices, mask.shape, orig_shape=orig_shape)
    else:
        # NOTE: If only one dimension, assume its the bias.
        _W = jnp.ones((W.shape[0], W.shape[0]))
        h = jnp.ones(_W.shape[1])
        b = jnp.ones(W.shape)

        def f(W: Array, h: Array, b: Array):
            for i in range(n):
                h = W @ h + b

            return h

        mask = (jax.jacrev(f, argnums=2)(_W, h, b) > 0.0).astype(jnp.float32)
        mask = BCOO.fromdense(mask)
        return SparseMask(mask.indices, mask.shape, orig_shape=mask.shape)


def _sparse_mask_to_mask(mask: SparseMask | Mask) -> Mask:
    if isinstance(mask, Mask):
        return mask

    indices = mask.indices
    data = jnp.ones(indices.shape[0])
    dense_mask = BCOO((data, indices), shape=mask.shape, unique_indices=True).todense()
    dense_mask = Mask(dense_mask.reshape(mask.orig_shape))

    return dense_mask


def densify_jacobian_mask(mask: _T) -> _T:
    """
    Given a mask PyTree, convert all the SparseMasks to Mask.
    """

    mask = jtu.tree_map(
        lambda leaf: _sparse_mask_to_mask(leaf),
        mask,
        is_leaf=lambda node: isinstance(node, (SparseMask, Mask)),
    )

    return mask
