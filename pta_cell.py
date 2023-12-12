from typing import Any, List, Optional, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree, Scalar


@jax.jit
def scaled_rotation_matrix(
    theta: Float32[Array, ""], alfa: Float32[Array, ""]
) -> Float32[Array, "2 2"]:
    matrix = jnp.zeros((2, 2), dtype=jnp.float32)

    matrix = matrix.at[0, 0].set(jnp.cos(theta))
    matrix = matrix.at[0, 1].set(-jnp.sin(theta))
    matrix = matrix.at[1, 0].set(jnp.sin(theta))
    matrix = matrix.at[1, 1].set(jnp.cos(theta))

    return alfa * matrix


def pta_matrix(alfas: List[Scalar], thetas: List[Scalar]) -> Float32[Array, "..."]:
    N = len(thetas)
    matrix = jnp.zeros((N * 2, N * 2))

    for i, alfa, theta in zip(range(N), alfas, thetas):
        rot = scaled_rotation_matrix(alfa, theta)
        matrix = matrix.at[i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2].set(rot)

    return matrix


class PTACell(eqx.Module):
    weights_hh: Float32[Array, "n_hidden n_hidden"]
    weights_ih: Float32[Array, "n_hidden input_dim"]
    bias: Optional[Float32[Array, "n_hidden"]]
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        if not (hidden_size % 2 == 0):
            raise ValueError("For the PTA Cell the hidden size must be a even number")

        num_rotation_blocks = hidden_size // 2
        hhkey, ihkey, bkey = jax.random.split(key, 3)

        # PTA Random Initialization
        hh_subkeys = jax.random.split(hhkey, num_rotation_blocks)
        alfas = [jax.random.uniform(key, minval=0, maxval=10) for key in hh_subkeys]
        thetas = [
            jax.random.uniform(key, minval=0, maxval=jnp.pi) for key in hh_subkeys
        ]

        self.weights_hh = pta_matrix(alfas, thetas)

        # Use Glorot Initialization for the IH Matrix.
        lim = jnp.sqrt(1 / hidden_size)
        self.weights_ih = jax.random.uniform(
            ihkey, shape=(hidden_size, input_size), minval=-lim, maxval=lim
        )

        if use_bias:
            self.bias = jax.random.uniform(
                bkey, (hidden_size,), minval=-lim, maxval=lim
            )
        else:
            self.bias = None

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias

    @jax.jit
    def f(
        self, h: Float32[Array, "hidden_size"], x: Float32[Array, "input_size"]
    ) -> Float32[Array, "hidden_size"]:
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0

        h = jnp.tanh(self.weights_hh @ h + self.weights_ih @ x + bias)

        return h

    def __call__(
        self, h: Float32[Array, "hidden_size"], x: Float32[Array, "input_size"]
    ) -> Float32[Array, "hidden_size"]:
        return self.f(h, x)


def zero_influence_pytree(cell: PTACell) -> PTACell:
    def zeros_jacobian(leaf):
        return jnp.zeros(shape=(cell.hidden_size, *leaf.shape))

    return jax.tree_map(zeros_jacobian, cell)


@jax.jit
def forward_rtrl(theta: PTACell, carry, input):
    """
    Not an efficient implementation.
    """
    h, prev_influence = carry
    h_new = PTACell.f(theta, h, input)
    jacobian = jax.jacrev(PTACell.f, argnums=(0, 1))
    inmediate_jacobian, dynamics = jacobian(theta, h, input)

    influence = jax.tree_map(
        lambda i_t, j_t_prev: i_t + dynamics @ j_t_prev,
        inmediate_jacobian,
        prev_influence,
    )

    # According to the paper, only entries that are not
    # zero in the immediate jacobian are kept
    onestep_infl_mask = jax.tree_map(
        lambda t: (jnp.abs(t) > 0.0).astype(jnp.float32), inmediate_jacobian
    )

    new_infl = jax.tree_map(
        lambda matrix, mask: matrix * mask, influence, onestep_infl_mask
    )

    return (h_new, new_infl), h_new


class GLU(eqx.Module):
    W: eqx.nn.Linear
    V: eqx.nn.Linear
    n_dim: int = eqx.field(static=True)

    def __init__(self, n_dim: int, key: PRNGKeyArray):
        w_key, v_key = jax.random.split(key, 2)
        self.n_dim = n_dim

        self.W = eqx.nn.Linear(n_dim, n_dim, use_bias=True, key=w_key)
        self.V = eqx.nn.Linear(n_dim, n_dim, use_bias=True, key=v_key)

    @jax.jit
    def __call__(self, x: Float32[Array, "n_dim"]) -> Float32[Array, "n_dim"]:
        res = self.W(x) * jax.nn.sigmoid(self.V(x))
        return res


class PTALayerRTRL(eqx.Module):
    cell: PTACell
    C: eqx.nn.Linear
    D: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    glu: GLU

    def __init__(
        self, hidden_size: int, input_size: int, use_bias: bool, key: PRNGKeyArray
    ):
        cell_key, c_key, d_key, glu_key = jax.random.split(key, 4)
        self.cell = PTACell(hidden_size, input_size, use_bias=use_bias, key=cell_key)
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=c_key)
        self.D = eqx.nn.Linear(hidden_size, input_size, use_bias=False, key=d_key)
        self.layer_norm = eqx.nn.LayerNorm(input_size)
        self.glu = GLU(hidden_size, key=glu_key)

    @jax.jit
    def __call__(self, h_prev: Float32[Array, "ndim"], input: Float32[Array, "ndim"]):
        """
        Applies layer norm to input.
        computes y(t) = g(h(t), input) where ht = PTA(h(t-1), input)
        And applies glu to y(t) and adds skip connection.

        Returns h_out, y_out
        """
        # Layer Norm First
        x = jax.vmap(self.layer_norm)(input)

        # To the PTA Cell
        h_out = self.cell(h_prev, x)

        # Project out
        y_out = self.C(h_out) + self.D(x)

        # Apply GLU
        y_out = self.glu(y_out)

        # Skip Connection
        y_out = y_out + input

        return h_out, y_out


class PTALayer(eqx.Module):
    cell: PTACell
    C: eqx.nn.Linear
    D: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    glu: GLU

    def __init__(
        self, hidden_size: int, input_size: int, use_bias: bool, key: PRNGKeyArray
    ):
        cell_key, c_key, d_key, glu_key = jax.random.split(key, 4)
        self.cell = PTACell(hidden_size, input_size, use_bias=use_bias, key=cell_key)
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=c_key)
        self.D = eqx.nn.Linear(hidden_size, input_size, use_bias=False, key=d_key)
        self.layer_norm = eqx.nn.LayerNorm(input_size)
        self.glu = GLU(hidden_size, key=glu_key)

    @jax.jit
    def __call__(self, inputs: Float32[Array, "T ndim"]):
        """
        Recieves a sequence.
        """
        # Layer Norm First
        x = jax.vmap(self.layer_norm)(inputs)

        # To the PTA Cell
        prev_influence = zero_influence_pytree(self.cell)
        h = jnp.zeros(shape=(self.cell.hidden_size,))

        carry_out, h_out = jax.lax.scan(
            jax.tree_util.Partial(forward_rtrl, self.cell),
            init=(h, prev_influence),
            xs=x,
        )

        # Sequence out
        y_out = jax.vmap(self.C)(h_out) + jax.vmap(self.D)(x)

        # Apply GLU
        y_out = jax.vmap(self.glu)(y_out)

        # Skip Connection
        y_out = y_out + inputs

        return y_out


class Stacked(eqx.Module):
    layers: List[PTALayer]
    encoder: eqx.nn.Linear

    def __init__(
        self, key: PRNGKeyArray, num_layers: int, hidden_size: int, input_size: int
    ):
        self.layers = []
        keys = jax.random.split(key, num=num_layers + 1)
        for i in range(num_layers):
            layer = PTALayer(hidden_size, hidden_size, use_bias=True, key=keys[i])
            self.layers.append(layer)

        self.encoder = eqx.nn.Linear(
            in_features=input_size, out_features=hidden_size, key=keys[-1]
        )

    def __call__(self, inputs: Float32[Array, "T ndim"]):
        x = jax.vmap(self.encoder)(inputs)

        h_out = x
        for layer in self.layers:
            h_out = layer(h_out)

        return h_out
