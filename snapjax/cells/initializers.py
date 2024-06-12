from typing import List

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PRNGKeyArray, Scalar


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


def normal_weights(key: PRNGKeyArray, N: int, g: float) -> Array:
    key, _ = jrandom.split(key)
    variance = (g**2) / N
    weights = jrandom.normal(key, shape=(N, N))
    weights = jnp.sqrt(variance) * weights

    return weights


def normal_channels(key: PRNGKeyArray, out_dim: int, inp_dim: int) -> Array:
    key, _ = jrandom.split(key)
    variance = 1 / out_dim
    weights = jrandom.normal(key, shape=(out_dim, inp_dim))
    weights = jnp.sqrt(variance) * weights

    return weights


def glorot_weights(key: PRNGKeyArray, out_dim: int, inp_dim: int):
    lim = 1 / jnp.sqrt(inp_dim)
    weights = jrandom.uniform(key, shape=(out_dim, inp_dim), minval=-lim, maxval=lim)
    return weights


def sparse_lecun_matrix(key: PRNGKeyArray, N: int, sparsity_level: float):
    """
    Samples a a random matrix where every row is sampled from a different
    normal distribution so that the ouptut standard deviation is one at
    every output neuron, taking into the effective number of inputs given
    by the sparsity pattern. So basically Lecun row by row, where every row
    has different number of 'effective' inputs.
    """

    def _init_row(mask: Array, key: PRNGKeyArray):
        m = jnp.sum(mask)
        variance = 1 / m
        weights = jrandom.normal(key, shape=mask.shape)
        weights = jnp.sqrt(variance) * weights

        return weights * mask

    mask_key, weights_key = jrandom.split(key)
    mask = jrandom.bernoulli(mask_key, p=(1 - sparsity_level), shape=(N, N))

    rows_keys = jrandom.split(weights_key, N)
    W = jax.vmap(_init_row)(mask, rows_keys)

    return BCOO.fromdense(W)
