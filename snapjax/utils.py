from functools import partial
from typing import Any

import jax
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from optax._src.base import GradientTransformation

from snapjax.cells.base import RTRLStacked


@jax.jit
def sparse_aware_update(model: RTRLStacked, updates: RTRLStacked):
    """
    For models that contain weights as BCOO arrays.
    """

    def _update(weight: BCOO | Array, update: Array):
        if isinstance(weight, BCOO):
            return BCOO(
                (weight.data + update, weight.indices),
                shape=weight.shape,
                indices_sorted=weight.indices_sorted,
                unique_indices=weight.unique_indices,
            )
        else:
            return weight + update

    model = jtu.tree_map(
        _update,
        model,
        updates,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    return model


@partial(jax.jit, static_argnames=["optimizer"])
def apply_update(model: RTRLStacked, grads: RTRLStacked, state: Any, optimizer: Any):
    updates, state = optimizer.update(grads, state)
    model = sparse_aware_update(model, updates)
    return model, state


@partial(jax.jit, static_argnames=["optimizer"])
def batch_apply_update(
    model: RTRLStacked, batch_grads: RTRLStacked, state: Any, optimizer: Any
):
    grads = jtu.tree_map(lambda mat: mat.mean(axis=0), batch_grads)
    updates, state = optimizer.update(grads, state)
    model = sparse_aware_update(model, updates)
    return model, state


def sparse_aware_init(model: RTRLStacked, optimizer: GradientTransformation):
    optim_state = optimizer.init(model)
    optim_state = jtu.tree_map(
        lambda node: node.data if isinstance(node, BCOO) else node,
        optim_state,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    return optim_state
