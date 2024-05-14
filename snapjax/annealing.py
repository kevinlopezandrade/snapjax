from typing import Callable, Protocol

import jax
import jax.numpy as jnp
import optax
from jax._src.dtypes import is_python_scalar


class Annealer(Protocol):
    @staticmethod
    def init(*args, **kwargs) -> Callable[[int], float]: ...


class CosineCoin:
    @staticmethod
    def init(end_step: int):
        return optax.cosine_decay_schedule(1, end_step)


class ComplementCosineCoin:
    @staticmethod
    def init(end_step: int):
        aux_func = optax.cosine_decay_schedule(1, end_step)
        return lambda step: 1 - aux_func(step)


class FlipCoin:
    @staticmethod
    def init(end_step: int):
        f = lambda step: 0.5
        f_batch = jax.vmap(f)

        @jax.jit
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

        @jax.jit
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

        @jax.jit
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
