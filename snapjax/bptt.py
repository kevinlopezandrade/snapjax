from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Scalar

from snapjax.algos import make_init_state, make_perturbations
from snapjax.cells.base import RTRLStacked, State


def l2_loss(inp, pred):
    return jnp.sum((inp - pred) ** 2)


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
    loss_func: Callable[[Array, Array], Scalar] = l2_loss,
    use_scan: bool = True,
):
    def _loss(model: RTRLStacked, inputs: Array, targets: Array):
        preds = forward_sequence(model, inputs, use_scan=use_scan)
        losses = jnp.sum(jax.vmap(loss_func)(targets, preds))
        return losses, preds

    (acc_loss, preds), acc_grads = jax.value_and_grad(_loss, has_aux=True)(
        model, inputs, targets
    )

    return acc_loss, acc_grads
