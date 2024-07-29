from functools import partial
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from optax._src.base import GradientTransformation

from snapjax.cells.base import RTRLStacked


@jax.jit
def sparse_aware_update(model: RTRLStacked, updates: RTRLStacked):
    """
    For models that contain weights as BCOO arrays.
    """

    def _update(weight: BCOO | Array, update: Array | None):
        if isinstance(weight, BCOO):
            return BCOO(
                (weight.data + update, weight.indices),
                shape=weight.shape,
                indices_sorted=weight.indices_sorted,
                unique_indices=weight.unique_indices,
            )
        elif eqx.is_array_like(weight):
            return weight + update
        else:
            return weight

    model = jtu.tree_map(
        _update,
        model,
        updates,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    return model


@partial(jax.jit, static_argnums=(3, 4))
def apply_update(
    model: RTRLStacked,
    grads: RTRLStacked,
    state: Any,
    optimizer: Any,
    return_updates: bool = False,
) -> Tuple[RTRLStacked, Any]:
    updates, state = optimizer.update(grads, state)
    model = sparse_aware_update(model, updates)
    if return_updates:
        return model, state, updates
    else:
        return model, state


@partial(jax.jit, static_argnames=["optimizer"])
def batch_apply_update(
    model: RTRLStacked, batch_grads: RTRLStacked, state: Any, optimizer: Any
):
    grads = jtu.tree_map(lambda mat: mat.mean(axis=0), batch_grads)
    updates, state = optimizer.update(grads, state)
    model = sparse_aware_update(model, updates)
    return model, state


def sparse_aware_init(model: RTRLStacked, optimizer: GradientTransformation):
    optim_state = optimizer.init(model)
    optim_state = jtu.tree_map(
        lambda node: node.data if isinstance(node, BCOO) else node,
        optim_state,
        is_leaf=lambda node: isinstance(node, BCOO),
    )

    return optim_state


class Welford:
    """Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self, shape):
        self.k = 0
        self.M = jnp.zeros(shape)
        self.S = jnp.zeros(shape)
        self.crossings = 0
        self.prev = None

    def update(self, x, crossings: bool = False):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * (1.0 / self.k)
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

        if crossings:
            if self.prev is not None:
                if jnp.sign(x) != jnp.sign(self.prev):
                    self.crossings += 1

            self.prev = x

    def __call__(self, x, crossings: bool = False):
        self.update(x, crossings=crossings)

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        if self.k == 1:
            return 0
        return jnp.sqrt(self.S / (self.k - 1))
