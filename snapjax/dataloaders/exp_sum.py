import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import scipy
from jaxtyping import PRNGKeyArray
from numpy.typing import NDArray


def exp_sum_sequence_scipy(
    key: PRNGKeyArray,
    dt: float,
    T: float,
    c: NDArray[np.float32],
    b: NDArray[np.float32],
    W: NDArray[np.float32],
):
    times = np.arange(0, T + dt, step=dt)

    # x(s)
    variance = 1
    mean = 0
    x_discrete = np.sqrt(variance) * jrandom.normal(key, shape=times.shape) + mean

    def p(t):
        return c.T @ scipy.linalg.expm(W * t) @ b

    p_discrete = np.array([p(time) for time in times])

    y = np.convolve(p_discrete, x_discrete, mode="full") * dt
    y = y[: len(times)]

    mask = np.zeros(times.shape[0])
    mask[-1] = 1.0

    # test = np.trapz(p_discrete * x_discrete[::-1], dx=dt)

    return x_discrete, y, mask


def _convolution_with_white_noise(
    key: PRNGKeyArray, dt: float, signal: NDArray, mean: int = 0, variance: float = 1.0
):
    x_discrete = jnp.sqrt(variance) * jrandom.normal(key, shape=signal.shape) + mean
    y = jnp.convolve(signal, x_discrete, mode="full") * dt
    y = y[: len(signal)]
    return x_discrete, y


def gen_exp_sum(
    key: PRNGKeyArray, N: int, m: int, dt: float, T: float, mask_target: bool
):
    with jax.default_device(jax.devices("cpu")[0]):
        key, c_key, b_key, matrix_key = jrandom.split(key, 4)

        # Kernel params
        c = jrandom.normal(c_key, shape=(m,))
        b = jrandom.normal(b_key, shape=(m,))

        variance = 1 / m
        Z = jrandom.normal(matrix_key, shape=(m, m))
        Z = jnp.sqrt(variance) * Z
        W = -1 * jnp.eye(Z.shape[0]) - (Z.T @ Z)

        # Time steps
        times = jnp.arange(0, T + dt, step=dt)

        # Kernel
        @jax.jit
        def p(t):
            return c.T @ jax.scipy.linalg.expm(W * t) @ b

        p_discrete = jax.vmap(p)(times)

        # Mask
        if mask_target:
            mask = jnp.zeros(len(times))
            mask = mask.at[-1].set(1.0)
        else:
            mask = jnp.ones(len(times))

        keys = jrandom.split(key, N)
        for key in keys:
            inp, out = _convolution_with_white_noise(key, dt=dt, signal=p_discrete)
            inp = inp.reshape(inp.shape[0], 1)
            out = out.reshape(out.shape[0], 1)
            yield inp, out, mask


def gen_batch_exp_sum(
    key: PRNGKeyArray,
    N: int,
    bs: int,
    m: int,
    dt: float,
    T: float,
    mask_target: bool = True,
):
    TOTAL = N * bs

    batch_inp = []
    batch_out = []
    batch_mask = []

    for i, (inp, out, mask) in enumerate(
        gen_exp_sum(key, TOTAL, m=m, dt=dt, T=T, mask_target=mask_target), 1
    ):
        batch_inp.append(inp)
        batch_out.append(out)
        batch_mask.append(mask)
        if i % bs == 0:
            yield jnp.array(batch_inp), jnp.array(batch_out), jnp.array(batch_mask)
            batch_inp = []
            batch_out = []
            batch_mask = []
