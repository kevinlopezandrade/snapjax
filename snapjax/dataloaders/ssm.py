from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax.scipy.linalg import expm
from jaxtyping import Array, ArrayLike, PRNGKeyArray
from numpy import convolve
from numpy.linalg import matrix_power
from scipy.linalg import logm


# K_conv and causal_convolution are taken
# from https://srush.github.io/annotated-s4/
def K_conv(Ab, Bb, Cb, L):
    return jnp.array([(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)])


@jax.jit
def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return jnp.fft.irfft(out)[: u.shape[0]]


def check_log_existence(M: ArrayLike):
    val = logm(M)

    return np.allclose(np.imag(val), 0)


def _gen_ssm(key: PRNGKeyArray, N: int, h: int, T: int):
    with jax.default_device(jax.devices("cpu")[0]):
        key, c_key, b_key, matrix_key = jrandom.split(key, 4)
        c = jrandom.normal(c_key, shape=(h,))
        b = jrandom.normal(b_key, shape=(h,))

        # Generate recurrence matrix.
        d_key, p_key = jrandom.split(matrix_key, 2)
        D = jnp.diag(jrandom.uniform(d_key, shape=(h,), minval=-1, maxval=0))
        P = jrandom.normal(p_key, shape=(h, h))
        P_inv = jnp.linalg.inv(P)
        W = P @ expm(D) @ P_inv

        assert check_log_existence(W)

        kernel = K_conv(W, b, c, T)

        keys = jrandom.split(key, N)

        mask = jnp.ones(T)
        for key in keys:
            inp = jrandom.normal(key, shape=(T,))
            out = causal_convolution(inp, kernel)
            inp = inp.reshape(inp.shape[0], 1)
            out = out.reshape(out.shape[0], 1)

            yield inp, out, mask


def gen_ssm(key: PRNGKeyArray, N: int, **params):
    for inp, out, mask in _gen_ssm(key, N=N, **params):
        yield inp, out, mask
