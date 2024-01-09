import time

import jax.numpy as jnp
import jax.random as jrandom

from snapjax.cells.rnn import StackedRNN


def get_stacked_rnn(num_layers: int, hidden_size: int, input_size: int, sparse: bool):
    key = int(time.time() * 1000)
    theta = StackedRNN(
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        sparse=sparse,
        key=jrandom.PRNGKey(key),
    )

    return theta


def get_random_input_sequence(T: int, model: StackedRNN):
    random_num = int(time.time() * 1000)
    key = jrandom.PRNGKey(random_num)
    inputs = jrandom.normal(key, shape=(T, model.input_size), dtype=jnp.float32)

    return inputs
