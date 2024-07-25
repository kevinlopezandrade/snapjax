import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.experimental.sparse import BCOO

from snapjax.sp_jacrev import SparseMask, make_jacobian_projection, sp_jacrev


def test_matrix_jacobian_sparse():

    def f(W, h):
        res = W @ h
        res = res.at[0].set(res[0] + W[1, 1] * 5.0)
        return res

    seed = int(time.time() * 1000)
    w_key, h_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    W = jax.random.normal(w_key, (4, 4))
    h = jax.random.normal(h_key, (4,))

    jacobian = jax.jacobian(f)(W, h)

    sp = BCOO.fromdense((jnp.abs(jacobian) > 0.0).astype(jnp.float32).reshape(4, -1))
    sp = SparseMask(sp.indices, sp.shape, jacobian.shape)
    projection = make_jacobian_projection(sp)

    partial = jax.tree_util.Partial(f, h=h)
    sp_jacobian_func = sp_jacrev(partial, projection)
    sp_jacobian = sp_jacobian_func(W)
    # print(sp_jacobian.todense().reshape(4, 4, 4))

    assert jnp.allclose(jacobian.reshape(jacobian.shape[0], -1), sp_jacobian.todense())
