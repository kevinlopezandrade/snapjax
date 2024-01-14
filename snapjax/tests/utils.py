import time

import jax.numpy as jnp
import jax.random as jrandom

from snapjax.cells.rnn import RNNLayer
from snapjax.cells.stacked import Stacked


def get_stacked_rnn(
    num_layers: int,
    hidden_size: int,
    input_size: int,
    seed: int | None = None,
):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    layer_args = {
        "hidden_size": hidden_size,
        "input_size": hidden_size,
    }

    theta = Stacked(
        RNNLayer,
        d_inp=input_size,
        d_out=input_size,
        num_layers=num_layers,
        cls_kwargs=layer_args,
        key=jrandom.PRNGKey(key),
    )

    return theta


def get_random_sequence(T: int, model: Stacked, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    inputs = jrandom.normal(
        jrandom.PRNGKey(key), shape=(T, model.d_inp), dtype=jnp.float32
    )

    return inputs


def get_random_batch(N: int, T: int, model: Stacked, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    batch_inputs = jrandom.normal(
        jrandom.PRNGKey(key), shape=(N, T, model.d_inp), dtype=jnp.float32
    )

    return batch_inputs
