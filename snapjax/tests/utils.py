import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array

from snapjax.cells.base import RTRLStacked, is_rtrl_cell
from snapjax.cells.continous_rnn import (
    ContinousRNNLayer,
    SparseFiringRateRNN,
    get_random_vectors,
)
from snapjax.cells.rnn import RNN, RNNLayer, glorot_weights
from snapjax.cells.stacked import StackedCell
from snapjax.sp_jacrev import DenseProjection, Mask, SparseMask, SparseProjection


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

    layers = []
    keys = jrandom.split(jrandom.PRNGKey(key), num_layers)
    for i in range(num_layers):
        layer = RNNLayer(hidden_size=hidden_size, input_size=input_size, key=keys[i])
        layers.append(layer)

    theta = StackedCell(layers)

    return theta


def get_sparse_continous_rnn(
    hidden_size: int, input_size: int, sparsity_fraction: float, seed: int | None = None
):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    w_key, u_key, c_key, sp_key = jrandom.split(jrandom.PRNGKey(key), 4)
    W_0 = glorot_weights(w_key, inp_dim=hidden_size, out_dim=hidden_size)
    U = get_random_vectors(u_key, inp_dim=input_size, out_dim=hidden_size)
    C = get_random_vectors(c_key, inp_dim=hidden_size, out_dim=input_size)

    cell = SparseFiringRateRNN(W_0, U, sparsity_fraction=sparsity_fraction, key=sp_key)
    rnn = ContinousRNNLayer(C, cell=cell, dt=0.5)
    rnn = StackedCell([rnn])

    return rnn


def get_random_sequence(T: int, model: StackedCell, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    inputs = jrandom.normal(jrandom.PRNGKey(key), shape=(T, model.layers[0].d_inp))

    return inputs


def get_random_batch(N: int, T: int, model: StackedCell, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    batch_inputs = jrandom.normal(
        jrandom.PRNGKey(key), shape=(N, T, model.layers[0].d_inp)
    )

    return batch_inputs


def get_random_mask(T: int, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    mask = jrandom.bernoulli(jrandom.PRNGKey(key), p=0.5, shape=(T,)) * 1.0

    return mask


def get_random_mask_batch(N: int, T: int, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    mask = jrandom.bernoulli(
        jrandom.PRNGKey(key),
        p=0.4,
        shape=(
            N,
            T,
        ),
    )

    mask = mask * 1.0

    return mask


def replace_rnn_with_diagonals(model: StackedCell):
    # Replace by diagonal matrices.
    # Ugly but works.
    # Hack for the nn.RNN
    class MatrixCallable(eqx.Module):
        W: Array

        def __call__(self, x):
            return self.W @ x

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
                MatrixCallable(jnp.fill_diagonal(I, diag_hh, inplace=False)),
            )

            diag_ih = jrandom.normal(subkeys[1], (leaf.hidden_size,))
            cell = eqx.tree_at(
                lambda cell: cell.weights_ih,
                cell,
                MatrixCallable(jnp.fill_diagonal(I, diag_ih, inplace=False)),
            )

            leafs[i] = cell

    model = jtu.tree_unflatten(tree_def, leafs)
    return model


def make_dense_identity_mask(jacobian_mask: RTRLStacked) -> RTRLStacked:
    def _convert(leaf: Mask):
        mask = jnp.ones(leaf.mask.shape)
        return Mask(mask)

    jacobian_mask = jtu.tree_map(
        _convert, jacobian_mask, is_leaf=lambda node: isinstance(node, Mask)
    )

    return jacobian_mask


def make_dense_jacobian_projection(jacobian_projection: RTRLStacked) -> RTRLStacked:
    def _convert(leaf: DenseProjection | SparseProjection):
        if isinstance(leaf, DenseProjection):
            return leaf
        else:
            projection_matrix = jnp.eye(leaf.projection_matrix.shape[1])
            return DenseProjection(projection_matrix, shape=leaf.sparse_def.shape)

    jacobian_projection = jtu.tree_map(
        _convert,
        jacobian_projection,
        is_leaf=lambda node: isinstance(node, (DenseProjection, SparseProjection)),
    )

    return jacobian_projection


def mask_to_set(mask: SparseMask):
    indices = mask.indices
    return set(frozenset(row.tolist()) for row in indices)


def is_subset_pytree(A: RTRLStacked, B: RTRLStacked):
    A = jtu.tree_map(
        lambda leaf: mask_to_set(leaf) if isinstance(leaf, SparseMask) else frozenset(),
        A,
        is_leaf=lambda node: isinstance(node, (SparseMask, Mask)),
    )

    B = jtu.tree_map(
        lambda leaf: mask_to_set(leaf) if isinstance(leaf, SparseMask) else frozenset(),
        B,
        is_leaf=lambda node: isinstance(node, (SparseMask, Mask)),
    )

    is_subset = jtu.tree_map(
        lambda a, b: a.issubset(b),
        A,
        B,
    )

    return jtu.tree_all(is_subset)
