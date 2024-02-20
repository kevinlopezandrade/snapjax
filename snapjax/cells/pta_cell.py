from typing import List, Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, Scalar

from snapjax.cells.base import RTRLCell, RTRLLayer, State
from snapjax.cells.utils import snap_n_mask
from snapjax.sp_jacrev import sp_jacrev


def scaled_rotation_matrix(theta: Scalar, alfa: Scalar) -> Array:
    matrix = jnp.zeros((2, 2), dtype=jnp.float32)

    matrix = matrix.at[0, 0].set(jnp.cos(theta))
    matrix = matrix.at[0, 1].set(-jnp.sin(theta))
    matrix = matrix.at[1, 0].set(jnp.sin(theta))
    matrix = matrix.at[1, 1].set(jnp.cos(theta))

    return alfa * matrix


def pta_matrix(alfas: List[Scalar], thetas: List[Scalar]) -> Array:
    blocks: List[Array] = []
    for alfa, theta in zip(alfas, thetas):
        rot = scaled_rotation_matrix(alfa, theta)
        blocks.append(rot)

    matrix = jax.scipy.linalg.block_diag(*blocks)

    return matrix


def pta_weights(key: PRNGKeyArray, inp_dim: int, out_dim: int):
    if not (out_dim % 2 == 0):
        raise ValueError("PTA Initialization requires even number of out dim")
    if inp_dim != out_dim:
        raise ValueError("PTA Initialization only for square matrices.")

    num_rotation_blocks = out_dim // 2
    alfas_key, thetas_key = jax.random.split(key, 2)

    # PTA Random Initialization
    alfas_keys = jax.random.split(alfas_key, num_rotation_blocks)
    alfas = [jax.random.uniform(key, minval=0, maxval=10) for key in alfas_keys]

    thetas_keys = jax.random.split(thetas_key, num_rotation_blocks)
    thetas = [jax.random.uniform(key, minval=0, maxval=jnp.pi) for key in thetas_keys]

    weights = pta_matrix(alfas, thetas)

    return weights


class GLU(eqx.Module):
    W: eqx.nn.Linear
    V: eqx.nn.Linear
    n_dim: int = eqx.field(static=True)

    def __init__(self, n_dim: int, key: PRNGKeyArray):
        w_key, v_key = jax.random.split(key, 2)
        self.n_dim = n_dim

        self.W = eqx.nn.Linear(n_dim, n_dim, use_bias=True, key=w_key)
        self.V = eqx.nn.Linear(n_dim, n_dim, use_bias=True, key=v_key)

    def __call__(self, x: Array) -> Array:
        res = self.W(x) * jax.nn.sigmoid(self.V(x))
        return res


class PTACell(RTRLCell):
    weights_hh: Array
    weights_ih: nn.Linear
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
        hhkey, ihkey = jax.random.split(key, 2)

        # PTA Random Initialization
        hh_subkeys = jax.random.split(hhkey, num_rotation_blocks)
        alfas = [jax.random.uniform(key, minval=0, maxval=10) for key in hh_subkeys]
        thetas = [
            jax.random.uniform(key, minval=0, maxval=jnp.pi) for key in hh_subkeys
        ]

        self.weights_hh = pta_matrix(alfas, thetas)
        self.weights_ih = nn.Linear(
            input_size, hidden_size, use_bias=use_bias, key=ihkey
        )

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_bias = use_bias

    def f(
        self,
        state: State,
        input: Array,
    ) -> State:
        h_new = jnp.tanh(self.weights_hh @ state + self.weights_ih(input))

        return h_new

    def make_snap_n_mask(self: RTRLCell, n: int) -> RTRLCell:
        mask = jtu.tree_map(lambda leaf: snap_n_mask(leaf, n), self)

        return mask


class PTALayer(RTRLLayer):
    cell: PTACell
    C: eqx.nn.Linear
    D: eqx.nn.Linear
    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3)
        self.cell = PTACell(
            hidden_size=hidden_size,
            input_size=input_size,
            use_bias=use_bias,
            key=keys[0],
        )
        self.C = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[1])
        self.D = eqx.nn.Linear(input_size, hidden_size, use_bias=False, key=keys[2])

        self.d_inp = input_size
        self.d_out = hidden_size

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: PTACell = None,
    ):
        # Compute Jacobians
        if sp_projection_cell:
            f_partial = jtu.Partial(PTACell.f, state=state, input=input)
            jacobian_func = sp_jacrev(f_partial, sp_projection_cell, transpose=True)
            inmediate_jacobian = jacobian_func(self.cell)
            dynamics_fun = jax.jacrev(PTACell.f, argnums=1)
            dynamics = dynamics_fun(self.cell, state, input)
        else:
            jacobian_func = jax.jacrev(PTACell.f, argnums=(0, 1))
            inmediate_jacobian, dynamics = jacobian_func(self.cell, state, input)

        h_out = self.cell.f(state, input) + perturbation

        # Project out
        y_out = self.C(h_out) + self.D(input)

        return h_out, (inmediate_jacobian, dynamics), y_out

    def f_bptt(self, state: State, input: Array) -> Tuple[State, Array]:
        h_out = self.cell.f(state, input)

        # Project out
        y_out = self.C(h_out) + self.D(input)

        return h_out, y_out
