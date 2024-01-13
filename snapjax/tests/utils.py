import time

import jax.numpy as jnp
import jax.random as jrandom

from snapjax.cells.rnn import StackedRNN


def get_stacked_rnn(
    num_layers: int,
    hidden_size: int,
    input_size: int,
    sparse: bool,
    seed: int | None = None,
):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    theta = StackedRNN(
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        sparse=sparse,
        key=jrandom.PRNGKey(key),
    )

    return theta


def get_random_sequence(T: int, model: StackedRNN, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    inputs = jrandom.normal(
        jrandom.PRNGKey(key), shape=(T, model.input_size), dtype=jnp.float32
    )

    return inputs


def get_random_batch(N: int, T: int, model: StackedRNN, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    batch_inputs = jrandom.normal(
        jrandom.PRNGKey(key), shape=(N, T, model.input_size), dtype=jnp.float32
    )

    return batch_inputs
