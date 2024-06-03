from collections.abc import Callable
from typing import Literal, Optional, TypeVar, Union

import equinox
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import networkx
import numpy as np
import scipy.sparse
from equinox import field, filter_vmap, is_array
from equinox.nn import Linear
from jax.experimental.sparse import BCOO
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from snapjax.sp_jacrev import Mask, SparseMask

_T = TypeVar("_T")


def sparse_mask_to_mask(mask: SparseMask | Mask) -> Mask:
    if isinstance(mask, Mask):
        return mask

    indices = mask.indices
    data = jnp.ones(indices.shape[0])
    dense_mask = BCOO(
        (data, indices), shape=mask.jacobian_shape, unique_indices=True
    ).todense()
    dense_mask = Mask(dense_mask.reshape(mask.orig_jacobian_shape))

    return dense_mask


def make_dense_identity_mask(jacobian_mask: _T) -> _T:
    def _convert(leaf: Mask):
        mask = jnp.ones(leaf.jacobian_mask.shape)
        return Mask(mask)

    jacobian_mask = jtu.tree_map(
        _convert, jacobian_mask, is_leaf=lambda node: isinstance(node, Mask)
    )

    return jacobian_mask


def densify_jacobian_mask(mask: _T) -> _T:
    """
    Given a mask PyTree, convert all the SparseMasks to Mask.
    """

    mask = jtu.tree_map(
        lambda leaf: sparse_mask_to_mask(leaf),
        mask,
        is_leaf=lambda node: isinstance(node, (SparseMask, Mask)),
    )

    return mask


def n_step_graph(G: networkx.DiGraph):
    paths_per_node = {}
    for node in G.nodes:
        paths_per_node[node] = networkx.single_target_shortest_path(G, node)

    n_step_matrix = np.zeros((len(G.nodes), len(G.edges)))
    sorted_edges = sorted(G.edges, key=lambda item: item[1])

    for node in G.nodes:
        row = n_step_matrix[node]
        paths = paths_per_node[node]
        for i, (u, v) in enumerate(sorted_edges):
            if v in paths.keys():
                row[i] = 1 + (len(paths[v]) - 1)

    return n_step_matrix


def _build_sparse_mask(adj_matrix: ArrayLike, n: int, raw: bool = False):
    # Tranpose because: (i, j)  = 1 in adj_matrix means, (i -> j)
    # but in a NN it means j -> i.
    if isinstance(adj_matrix, scipy.sparse.coo_array):
        graph = networkx.from_scipy_sparse_array(
            adj_matrix.T, create_using=networkx.DiGraph
        )
        orig_shape = None
    else:
        graph = networkx.from_numpy_array(adj_matrix.T, create_using=networkx.DiGraph)
        orig_shape = (adj_matrix.shape[0], *adj_matrix.shape)

    n_step_matrix = n_step_graph(graph)
    mask = (n_step_matrix <= n).astype(np.float32)

    jacobian_mask = n_step_matrix * mask
    jacobian_mask_sp = BCOO.from_scipy_sparse(scipy.sparse.coo_array(jacobian_mask))

    if not raw:
        return SparseMask(
            jacobian_mask_sp.indices,
            shape=jacobian_mask_sp.shape,
            orig_shape=orig_shape if orig_shape else jacobian_mask_sp.shape,
        )
    else:
        return jacobian_mask


def snap_n_mask_bcoo(W: BCOO, n: int) -> SparseMask:
    """
    Given a sparse connectivity matrix, compute its influence on future hidden
    states, to later mask the jacobian. This only works properly if a weight in
    h(t) = f(W, h(t-1), x(t)), affects only one output unit in just one timestep,
    if it affects more as in the GRU, it will not work properly. The variant of the
    GRU we use, fits this condition so we are safe here.
    """
    rows = W.indices[:, 0]
    cols = W.indices[:, 1]
    adj_matrix = scipy.sparse.coo_array((np.ones(W.nse), (rows, cols)), shape=(W.shape))

    return _build_sparse_mask(adj_matrix, n)


def snap_n_mask(W: Array, n: int):
    if n == 1:
        if W.ndim == 1:
            h = W.shape[0]
            indices = np.zeros((h, 2), dtype=np.int32)
            for i in range(h):
                indices[i] = jnp.array([i, i])

            return SparseMask(
                indices=jnp.array(indices), shape=(h, h), orig_shape=(h, h)
            )

        else:
            inp = W.shape[1]
            h = W.shape[0]
            indices = np.zeros((h * inp, 2), dtype=np.int32)
            for i in range(h):
                indices[(i * inp) : (i + 1) * inp, 0] = np.repeat(i, inp)
                indices[(i * inp) : (i + 1) * inp, 1] = np.arange(inp) + (i * inp)

            indices = jnp.array(indices)
            return SparseMask(
                indices=indices, shape=(h, h * inp), orig_shape=(h, h, inp)
            )

    else:
        return snap_n_mask_dijkstra(W, n)


def snap_n_mask_dijkstra(W: Array, n: int, dense: bool = True):
    if dense:
        is_one_dim = False
        if W.ndim == 1:
            W = W.reshape(W.shape[0], 1)
            is_one_dim = True

        if W.shape[0] == W.shape[1]:
            adj_matrix = np.ones(W.shape)
            return _build_sparse_mask(adj_matrix, n)
        else:
            # Assume its encoding matrix, from n_inp to hidden dimension.
            # And build the entire adjency matrix of recurrent neurons and
            # input neurons..
            assert W.shape[0] >= W.shape[1]

            adj_matrix = np.ones((W.shape[0] + W.shape[1], W.shape[0] + W.shape[1]))
            adj_matrix[W.shape[0] :, :] = 0.0

            mask = _build_sparse_mask(adj_matrix, n, raw=True)

            indices_to_recover = []
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    index = (i, W.shape[0] + j)
                    index = adj_matrix.shape[1] * index[0] + index[1]
                    indices_to_recover.append(index)

            mask = mask[: W.shape[0], indices_to_recover]

            jacobian_mask_sp = BCOO.from_scipy_sparse(scipy.sparse.coo_array(mask))
            return SparseMask(
                jacobian_mask_sp.indices,
                shape=jacobian_mask_sp.shape,
                orig_shape=(
                    (W.shape[0], *W.shape)
                    if not is_one_dim
                    else (W.shape[0], W.shape[0])
                ),
            )
    else:
        raise NotImplementedError("Not implemented yet")


# Equinox MLP adapated to use static fields in
# activation functions. All credits to the equinox
# developers.
_identity = lambda x: x
_relu = jax.nn.relu


class MLP(equinox.Module, strict=True):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: tuple[Linear, ...]
    activation: Callable = field(static=True)
    final_activation: Callable = field(static=True)
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: int = field(static=True)
    depth: int = field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        activation: Callable = _relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Linear(in_size, out_size, use_final_bias, key=keys[0]))
        else:
            layers.append(Linear(in_size, width_size, use_bias, key=keys[0]))
            for i in range(depth - 1):
                layers.append(Linear(width_size, width_size, use_bias, key=keys[i + 1]))
            layers.append(Linear(width_size, out_size, use_final_bias, key=keys[-1]))
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = filter_vmap(
            filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            layer_activation = jtu.tree_map(
                lambda x: x[i] if is_array(x) else x, self.activation
            )
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x
