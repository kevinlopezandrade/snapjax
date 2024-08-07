import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import RTRLStacked, is_rtrl_cell
from snapjax.cells.continous_rnn import SparseFiringRateRNN
from snapjax.cells.initializers import (
    glorot_weights,
    normal_channels,
    normal_weights,
    sparse_lecun_matrix,
)
from snapjax.cells.readout import IdentityLayer, LinearTanhReadout
from snapjax.cells.rnn import RNN, RNNGeneral, RNNLayer
from snapjax.cells.stacked import StackedCell
from snapjax.sp_jacrev import DenseProjection, Mask, SparseMask, SparseProjection


def get_stacked_rnn(
    num_layers: int,
    hidden_size: int,
    input_size: int,
    seed: int | None = None,
    sparse: bool = False,
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

    theta = StackedCell(layers, sparse=sparse)

    return theta


def get_stacked_rnn_exact(
    num_layers: int,
    hidden_size: int,
    input_size: int,
    seed: int | None = None,
    sparse: bool = False,
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
        cell = RNN(hidden_size=hidden_size, input_size=input_size, key=keys[i])
        layer = IdentityLayer(cell)
        layers.append(layer)

    theta = StackedCell(layers, sparse=sparse)

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
    U = normal_channels(u_key, inp_dim=input_size, out_dim=hidden_size)
    C = normal_channels(c_key, inp_dim=hidden_size, out_dim=input_size)

    cell = SparseFiringRateRNN(
        W_0, U, dt=0.5, sparsity_fraction=sparsity_fraction, key=sp_key
    )
    rnn = LinearTanhReadout(C, cell=cell)
    rnn = StackedCell([rnn], sparse=True)

    return rnn


def get_random_sequence(T: int, model: StackedCell, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    inputs = jrandom.normal(jrandom.PRNGKey(key), shape=(T, model.layers[0].d_inp))

    return inputs


def get_random_batch(B: int, T: int, model: StackedCell, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    batch_inputs = jrandom.normal(
        jrandom.PRNGKey(key), shape=(B, T, model.layers[0].d_inp)
    )

    return batch_inputs


def get_random_mask(T: int, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    mask = jrandom.bernoulli(jrandom.PRNGKey(key), p=0.5, shape=(T,)) * 1.0

    return mask


def get_random_mask_batch(B: int, T: int, seed: int | None = None):
    if seed is None:
        key = int(time.time() * 1000)
    else:
        key = seed

    mask = jrandom.bernoulli(
        jrandom.PRNGKey(key),
        p=0.4,
        shape=(
            B,
            T,
        ),
    )

    mask = mask * 1.0

    return mask


def replace_rnn_with_diagonals(model: StackedCell, sparse_weights: bool = False):
    # Replace by diagonal matrices.
    # Ugly but works.
    # Hack for the nn.RNN
    class MatrixCallable(eqx.Module):
        W: Array | BCOO

        def __init__(self, W):
            if sparse_weights:
                self.W = BCOO.fromdense(W)
            else:
                self.W = W

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


def make_dense_jacobian_projection(jacobian_projection: RTRLStacked) -> RTRLStacked:
    def _convert(leaf: DenseProjection | SparseProjection):
        if isinstance(leaf, DenseProjection):
            return leaf
        else:
            projection_matrix = jnp.eye(leaf.projection_matrix.shape[1])
            return DenseProjection(
                projection_matrix, jacobian_shape=leaf.sparse_def.jacobian_shape
            )

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


def make_sparse_rnn_layer(
    inp_dim: int,
    h_dim: int,
    key: PRNGKeyArray,
    sparsity_level: float,
    sparse_u: bool = True,
    g: float = 0.9,
):
    W_key, U_key = jrandom.split(key, 2)
    if sparsity_level > 0.5:
        W = sparse_lecun_matrix(W_key, N=h_dim, sparsity_level=sparsity_level)

        if inp_dim != h_dim:
            U = normal_channels(U_key, h_dim, inp_dim)
        else:
            U = sparse_lecun_matrix(U_key, N=h_dim, sparsity_level=sparsity_level)
    else:
        W = normal_weights(W_key, N=h_dim, g=g)

        if inp_dim != h_dim:
            U = normal_channels(U_key, h_dim, inp_dim)
        else:
            U = normal_weights(U_key, N=h_dim, g=g)

    return RNNGeneral(W=W, U=U)
