import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as onp
import pytest
from jax.experimental.sparse.bcoo import BCOO

from snapjax.sp_jacrev import (
    Mask,
    SparseMask,
    make_jacobian_projection,
    sp_jacrev,
)

_SIZE = 50

# This units tests are test taken from
# the sparsejac repo: github.com/mfschubert/sparsejac/
# I just adapted them to make it work with pytest, by
# just deleting the self argument for the test.


class sparsejac:
    @staticmethod
    def jacrev(fn, sparsity):
        jac_fun = sp_jacrev(fn, make_jacobian_projection(sparsity), transpose=True)

        def f(x):
            tree = jac_fun(x)
            # Transpose since I return the tranposed jacobian
            # matrix.
            return jtu.tree_map(
                lambda jacobian: jacobian.transpose(),
                tree,
                is_leaf=lambda leaf: isinstance(leaf, BCOO),
            )

        return f


def test_diagonal():
    fn = lambda x: x**2
    sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    actual = sparsejac.jacrev(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())


def test_diagonal_jit():
    fn = lambda x: x**2
    sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    jacfn = sparsejac.jacrev(fn, sparsity)
    jacfn = jax.jit(jacfn)
    actual = jacfn(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())


def test_diagonal_shuffled():
    fn = lambda x: jax.random.permutation(jax.random.PRNGKey(0), x**2)
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    expected = jax.jacrev(fn)(x)
    sparsity = jsparse.BCOO.fromdense(expected != 0)
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    actual = sparsejac.jacrev(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())


def test_dense():
    fn = lambda x: jnp.stack((jnp.sum(x), jnp.sum(x) ** 2, jnp.sum(x) ** 3))
    sparsity = jsparse.BCOO.fromdense(jnp.ones((3, _SIZE)))
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    actual = sparsejac.jacrev(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())


def test_convolutional_1d():
    fn = lambda x: jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="valid")
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    i, j = jnp.meshgrid(jnp.arange(_SIZE - 2), jnp.arange(_SIZE), indexing="ij")
    sparsity = (i == j) | ((i + 1) == j) | ((i + 2) == j)
    sparsity = jsparse.BCOO.fromdense(sparsity)
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    actual = sparsejac.jacrev(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())


def test_convolutional_1d_nonlinear():
    fn = lambda x: jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="valid") ** 2
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    i, j = jnp.meshgrid(jnp.arange(_SIZE - 2), jnp.arange(_SIZE), indexing="ij")
    sparsity = (i == j) | ((i + 1) == j) | ((i + 2) == j)
    sparsity = jsparse.BCOO.fromdense(sparsity)
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    actual = sparsejac.jacrev(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())


def test_convolutional_2d():
    shape_2d = (20, 20)

    def fn(x_flat):
        x = jnp.reshape(x_flat, shape_2d)
        result = jax.scipy.signal.convolve2d(x, jnp.ones((3, 3)), mode="valid")
        return result.flatten()

    x_flat = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(shape_2d[0] * shape_2d[1],)
    )
    expected = jax.jacrev(fn)(x_flat)
    sparsity = jsparse.BCOO.fromdense(expected != 0)
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    actual = sparsejac.jacrev(fn, sparsity)(x_flat)
    onp.testing.assert_array_equal(expected, actual.todense())


def test_convolutional_2d_nonlinear():
    shape_2d = (20, 20)

    def fn(x_flat):
        x = jnp.reshape(x_flat, shape_2d)
        result = jax.scipy.signal.convolve2d(x, jnp.ones((3, 3)), mode="valid")
        return result.flatten() ** 2

    x_flat = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(shape_2d[0] * shape_2d[1],)
    )
    expected = jax.jacrev(fn)(x_flat)
    sparsity = jsparse.BCOO.fromdense(expected != 0)
    sparsity = SparseMask(sparsity.indices, sparsity.shape)
    actual = sparsejac.jacrev(fn, sparsity)(x_flat)
    onp.testing.assert_array_equal(expected, actual.todense())


@pytest.mark.parametrize("sparse_type", [jsparse.BCOO, jsparse.BCSR])
@pytest.mark.parametrize("argnums", [(2, 0), (-1, 0)])
@pytest.mark.parametrize("compiled", [False, True])
def test_sparse_parameter_tree_and_selected_arguments(sparse_type, argnums, compiled):
    weights = sparse_type.fromdense(jnp.array([[1.0, 2.0], [0.0, -3.0]]))
    bias = jnp.array([0.2, -0.5])
    x = jnp.array([0.7, -0.1])
    config = {"scale": 1.4}

    def dense_reference(data, bias, x):
        dense = jnp.array([[data[0], data[1]], [0.0, data[2]]])
        return jnp.tanh(config["scale"] * (dense @ x) + bias)

    expected_w, expected_bias, expected_x = jax.jacrev(
        dense_reference, argnums=(0, 1, 2)
    )(weights.data, bias, x)
    pattern = jsparse.BCOO.fromdense(expected_w != 0)
    projection = make_jacobian_projection({
        "weights": SparseMask(pattern.indices, pattern.shape),
        "bias": Mask(jnp.ones((2, 2))),
    })

    def fun(x, config, params):
        return jnp.tanh(config["scale"] * (params["weights"] @ x) + params["bias"])

    jacfun = sp_jacrev(fun, projection, argnums=argnums)
    if compiled:
        jacfun = jax.jit(jacfun)
    actual_params, actual_x = jacfun(x, config, {"weights": weights, "bias": bias})
    onp.testing.assert_allclose(actual_params["weights"].todense(), expected_w, atol=1e-14)
    onp.testing.assert_allclose(actual_params["bias"], expected_bias, atol=1e-14)
    onp.testing.assert_allclose(actual_x, expected_x, atol=1e-14)
