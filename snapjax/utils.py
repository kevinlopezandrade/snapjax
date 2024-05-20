from functools import partial
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray
from optax._src.base import GradientTransformation
from tqdm import tqdm

from snapjax.algos import (
    forward_rtrl,
    make_init_state,
    make_zeros_grads,
    make_zeros_jacobians_sp,
)
from snapjax.cells.base import RTRLStacked
from snapjax.dataloaders.base import bgen
from snapjax.losses import l_infinity, mean_squared_loss
from snapjax.mlops import Algorithm, get_algorithm
from snapjax.sp_jacrev import make_jacobian_projection


@jax.jit
def sparse_aware_update(model: RTRLStacked, updates: RTRLStacked):
    """
    For models that contain weights as BCOO arrays.
    """

    def _update(weight: BCOO | Array, update: Array | None):
        if isinstance(weight, BCOO):
            return BCOO(
                (weight.data + update, weight.indices),
                shape=weight.shape,
                indices_sorted=weight.indices_sorted,
                unique_indices=weight.unique_indices,
            )
        elif eqx.is_array_like(weight):
            return weight + update
        else:
            return weight

    model = jtu.tree_map(
        _update,
        model,
        updates,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    return model


@partial(jax.jit, static_argnums=3)
def apply_update(
    model: RTRLStacked, grads: RTRLStacked, state: Any, optimizer: Any
) -> Tuple[RTRLStacked, Any]:
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


def train(
    N: int,
    bs: int,
    rnn: RTRLStacked,
    generator: Any,
    algorithm: Algorithm,
    optimizer: Any,
    algorithm_params: Dict[str, Any] | None = None,
    return_models: bool = False,
):
    if algorithm_params is not None:
        _gradient = jtu.Partial(
            get_algorithm(algorithm), loss_func=mean_squared_loss, **algorithm_params
        )
    else:
        _gradient = jtu.Partial(get_algorithm(algorithm), loss_func=mean_squared_loss)

    gradient = jax.jit(jax.vmap(_gradient, in_axes=(None, 0, 0, 0, None, None)))

    jacobian_mask = None
    jacobian_projection = None

    if algorithm == Algorithm.RTRL:
        jacobian_mask = rnn.get_snap_n_mask(1)
        jacobian_projection = make_jacobian_projection(jacobian_mask)

    optim_state = optimizer.init(rnn)

    eval_loss = jax.vmap(l_infinity)  # For a sequence
    eval_loss = jax.vmap(eval_loss)  # For a batch of sequences
    eval_loss = jax.jit(eval_loss)

    errors = []
    models = [rnn]

    for inp, target, mask in tqdm(bgen(generator)(N=N, bs=bs), total=N):
        acc_loss, acc_grads, preds = gradient(
            rnn, inp, target, mask, jacobian_mask, jacobian_projection
        )
        rnn, optim_state = batch_apply_update(rnn, acc_grads, optim_state, optimizer)
        losses = eval_loss(target, preds, mask)
        max_loss = jnp.max(losses)

        errors.append(max_loss)
        if return_models:
            models.append(rnn)

    return errors, models if return_models else rnn


def train_online(
    N: int,
    rnn: RTRLStacked,
    generator: Any,
    optimizer: Any,
    return_models: bool = False,
    flip_coin: bool = False,
    key: PRNGKeyArray | None = None,
):
    _gradient = jtu.Partial(forward_rtrl, loss_func=mean_squared_loss)
    gradient = jax.jit(_gradient)

    jacobian_mask = rnn.get_snap_n_mask(1)
    jacobian_projection = make_jacobian_projection(jacobian_mask)

    optim_state = optimizer.init(rnn)

    eval_loss = jax.vmap(l_infinity)  # For a sequence
    eval_loss = jax.jit(eval_loss)

    errors = []
    models = [rnn]
    for inp, target, mask in tqdm(generator(N=N), total=N):
        T = inp.shape[0]
        jacobian = make_zeros_jacobians_sp(jacobian_projection)
        h = make_init_state(rnn)
        acc_grads = make_zeros_grads(rnn)
        preds = []
        for i in range(T):
            h, grads, jacobian, loss, pred = gradient(
                rnn,
                jacobian,
                h,
                inp[i],
                target[i],
                mask[i],
                jacobian_mask,
                jacobian_projection,
            )

            acc_grads = jtu.tree_map(
                lambda acc_grads, grads: acc_grads + grads, acc_grads, grads
            )

            if flip_coin:
                update = jrandom.choice(key, jnp.array([True, False]))
                key, _ = jrandom.split(key)
            else:
                update = mask[i] == 1.0

            if update:
                rnn, optim_state = apply_update(rnn, acc_grads, optim_state, optimizer)

            preds.append(pred)

        preds = jnp.stack(preds)
        losses = eval_loss(target, preds, mask)
        max_loss = jnp.max(losses)

        if return_models:
            models.append(rnn)

        errors.append(max_loss)

    return errors, models if return_models else rnn
