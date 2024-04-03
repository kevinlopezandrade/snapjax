from typing import Any, Generator, Tuple

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def bgen(generator: Generator[Tuple[Array, Array, Array], Any, None]):

    def _batched_version(N: int | None = None, bs: int | None = None, **params):
        TOTAL = N * bs

        b_inp = []
        b_out = []
        b_mask = []

        for i, (inp, out, mask) in enumerate(generator(N=TOTAL, **params), 1):
            b_inp.append(inp)
            b_out.append(out)
            b_mask.append(mask)

            if i % bs == 0:
                yield jnp.array(b_inp), jnp.array(b_out), jnp.array(b_mask)
                b_inp = []
                b_out = []
                b_mask = []

    return _batched_version
