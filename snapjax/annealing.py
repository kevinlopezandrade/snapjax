from typing import Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np


def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
    exponent: float = 1.0,
):
    def schedule(count):
        count = np.minimum(count, decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * count / decay_steps))
        decayed = (1 - alpha) * cosine_decay**exponent + alpha
        return init_value * decayed

    return schedule


class Annealer(Protocol):
    @staticmethod
    def init(*args, **kwargs) -> Callable[[int], float]: ...


class CosineCoin:
    @staticmethod
    def init(end_step: int):
        return cosine_decay_schedule(1, end_step)


class ComplementCosineCoin:
    @staticmethod
    def init(end_step: int):
        aux_func = cosine_decay_schedule(1, end_step)
        return lambda step: 1 - aux_func(step)


class FlipCoin:
    @staticmethod
    def init(end_step: int):
        f = lambda step: 0.5
        f_batch = jax.vmap(f)

        def _schedule(step):
            if jnp.isscalar(step):
                return f(step)
            else:
                return f_batch(step)

        return _schedule


class Always:
    @staticmethod
    def init(end_step: int):
        f = lambda step: 1.0
        f_batch = jax.vmap(f)

        def _schedule(step):
            if jnp.isscalar(step):
                return f(step)
            else:
                return f_batch(step)

        return _schedule


class Never:
    @staticmethod
    def init(end_step: int):
        f = lambda step: 0.0
        f_batch = jax.vmap(f)

        def _schedule(step):
            if jnp.isscalar(step):
                return f(step)
            else:
                return f_batch(step)

        return _schedule


POLICIES = {
    "FlipCoin": FlipCoin,
    "ComplementCosineCoin": ComplementCosineCoin,
    "CosineCoin": CosineCoin,
    "Always": Always,
    "Never": Never,
}
