import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from snapjax.cells.base import is_rtrl_cell
from snapjax.cells.rnn import RNN, RNNLayer
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
        "input_size": input_size,
    }

    theta = Stacked(
        RNNLayer,
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

    inputs = jrandom.normal(jrandom.PRNGKey(key), shape=(T, model.layers[0].d_inp))

    return inputs


def get_random_batch(N: int, T: int, model: Stacked, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    batch_inputs = jrandom.normal(
        jrandom.PRNGKey(key), shape=(N, T, model.layers[0].d_inp)
    )

    return batch_inputs


def replace_rnn_with_diagonals(model: Stacked):
    # Replace by diagonal matrices.
    # Ugly but works.
    leafs, tree_def = jtu.tree_flatten(model, is_leaf=is_rtrl_cell)
    key = jrandom.PRNGKey(7)
    for i, leaf in enumerate(leafs):
        if isinstance(leaf, RNN):
            key, *subkeys = jrandom.split(key, 3)
            I = jnp.eye(leaf.hidden_size)

            diag_hh = jrandom.normal(subkeys[0], (leaf.hidden_size,))
            cell = eqx.tree_at(
                lambda leaf: leaf.weights_hh,
                leaf,
                jnp.fill_diagonal(I, diag_hh, inplace=False),
            )

            diag_ih = jrandom.normal(subkeys[1], (leaf.hidden_size,))
            cell = eqx.tree_at(
                lambda cell: cell.weights_ih,
                cell,
                jnp.fill_diagonal(I, diag_ih, inplace=False),
            )

            leafs[i] = cell

    model = jtu.tree_unflatten(tree_def, leafs)
    return model
