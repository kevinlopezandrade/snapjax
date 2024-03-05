from typing import Callable, cast

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, Scalar

from snapjax.algos import make_init_state
from snapjax.cells.base import RTRLStacked
from snapjax.losses import l2


def forward_sequence(model: RTRLStacked, inputs: Array, use_scan: bool = True):
    hidden_state = make_init_state(model)

    # Ful forward pass over the sequence.
    if use_scan:
        h_T, out = jax.lax.scan(
            lambda carry, input: model.f_bptt(carry, input),
            init=hidden_state,
            xs=inputs,
        )
    else:
        carry = hidden_state
        y_hats = []
        for i in range(inputs.shape[0]):
            carry, y_hat = model.f_bptt(carry, inputs[i])
            y_hats.append(y_hat)

        y_hats = jnp.stack(y_hats)
        out = y_hats

    return out


def bptt(
    model: RTRLStacked,
    inputs: Array,
    targets: Array,
    mask: Array | None = None,
    jacobian_mask: RTRLStacked | None = None,  # Ignored for consistency with API RTRL.
    jacobian_projection: RTRLStacked | None = None,  # Ignored
    loss_func: Callable[[Array, Array, float], Scalar] = l2,
    use_scan: bool = True,
    sparse_model: bool = False,
):
    if mask is not None:
        factor = mask.sum()
    else:
        mask = jnp.ones(targets.shape[0])
        factor = 1

    def _loss(model: RTRLStacked, inputs: Array, targets: Array):
        preds = forward_sequence(model, inputs, use_scan=use_scan)
        losses = (1 / factor) * jnp.sum(jax.vmap(loss_func)(targets, preds, mask))
        return losses, preds

    if not sparse_model:
        (acc_loss, preds), acc_grads = jax.value_and_grad(_loss, has_aux=True)(
            model, inputs, targets
        )
    else:
        (acc_loss, preds), acc_grads = jsparse.value_and_grad(
            _loss, argnums=(0,), has_aux=True
        )(model, inputs, targets)
        acc_grads = jtu.tree_unflatten(
            jtu.tree_structure(model, is_leaf=lambda node: isinstance(node, BCOO)),
            acc_grads,
        )
        acc_grads = jtu.tree_map(
            lambda node: node.data if isinstance(node, BCOO) else node,
            acc_grads,
            is_leaf=lambda node: isinstance(node, BCOO),
        )

    acc_loss = cast(float, acc_loss)
    acc_grads = cast(RTRLStacked, acc_grads)
    preds = cast(Array, preds)
    return acc_loss, acc_grads, preds
