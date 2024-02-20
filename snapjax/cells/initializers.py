from typing import List

import jax
import jax.numpy as jnp
import jax.random as jrandom
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
