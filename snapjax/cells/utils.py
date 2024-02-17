from logging import debug

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Array

from snapjax.sp_jacrev import SparseMask


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

    def f(W: BCOO, h: Array):
        for i in range(n):
            h = W @ h

        return h

    mask = (jsparse.jacrev(f)(_W, h).data > 0.0).astype(jnp.float32)
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

        mask = mask.reshape(h.shape[0], -1)
        mask = BCOO.fromdense(mask)
        return SparseMask(mask.indices, mask.shape, orig_shape=orig_shape)
    else:
        mask = jnp.eye(W.shape[0])
        mask = BCOO.fromdense(mask)
        return SparseMask(mask.indices, mask.shape, orig_shape=mask.shape)
