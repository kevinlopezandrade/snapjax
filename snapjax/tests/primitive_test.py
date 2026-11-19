import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from snapjax.spp_primitives.primitives import spp_csr_matmul


def example(dtype):
    # Rectangular input, empty row, irregular row lengths, non-sorted output.
    a = np.array([[0, 2, 0, -1], [0, 0, 0, 0], [3, 0, 4, 5]], dtype=dtype)
    b = np.arange(20, dtype=dtype).reshape(4, 5) / 7
    csr = csr_matrix(a)
    sp = np.array([[2, 4], [0, 1], [1, 3], [2, 0], [0, 1]], dtype=np.int32)
    args = tuple(jnp.asarray(x) for x in (csr.data, csr.indices, csr.indptr, b, sp))
    expected = (a @ b)[sp[:, 0], sp[:, 1]]
    return args, expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("compiled", [False, True])
def test_sampled_product(dtype, compiled):
    args, expected = example(dtype)
    call = jax.jit(spp_csr_matmul) if compiled else spp_csr_matmul
    result = call(*args)
    assert result.dtype == dtype
    assert result.devices() == {jax.devices()[0]}
    np.testing.assert_allclose(result, expected, rtol=2e-7 if dtype == np.float32 else 1e-14)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_and_padded_patterns(dtype):
    args, _ = example(dtype)
    empty = jnp.empty((0, 2), dtype=jnp.int32)
    assert jax.jit(spp_csr_matmul)(*args[:4], empty).shape == (0,)
    padded = jnp.array([[3, 0], [0, 5], [-1, 0], [0, -1]], dtype=jnp.int32)
    np.testing.assert_array_equal(jax.jit(spp_csr_matmul)(*args[:4], padded), 0)
    np.testing.assert_array_equal(
        jax.jit(spp_csr_matmul)(
            jnp.empty(0, dtype=dtype), jnp.empty(0, dtype=jnp.int32),
            jnp.zeros(4, dtype=jnp.int32), args[3], args[4],
        ),
        0,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axes", [
    (None, None, None, 0, None),
    (0, None, None, None, None),
    (0, None, None, 0, None),
    (1, None, None, 2, None),
    (0, 0, 0, 0, 0),
])
def test_batching(dtype, axes):
    args, _ = example(dtype)
    other = list(args)
    other[0] = other[0] * 2
    other[3] = other[3] + 1
    if axes[1] is not None:
        other[1] = (other[1] + 1) % other[3].shape[0]
        other[2] = jnp.array([0, 1, 2, 5], dtype=jnp.int32)
        other[4] = other[4][::-1]
    batched = tuple(
        arg if axis is None else jnp.stack([arg, changed], axis=axis)
        for arg, changed, axis in zip(args, other, axes)
    )
    expected = []
    for batch in range(2):
        single = [arg if axis is None else jnp.take(arg, batch, axis=axis)
                  for arg, axis in zip(batched, axes)]
        data, cols, indptr, b, sp = map(np.asarray, single)
        a = csr_matrix((data, cols, indptr), shape=(3, 4)).toarray()
        expected.append((a @ b)[sp[:, 0], sp[:, 1]])
    result = jax.jit(jax.vmap(spp_csr_matmul, in_axes=axes))(*batched)
    np.testing.assert_allclose(result, expected, rtol=2e-7 if dtype == np.float32 else 1e-14)


def test_nested_and_empty_batching():
    args, expected = example(np.float64)
    call = lambda rhs: spp_csr_matmul(*args[:3], rhs, args[4])
    nested = jnp.broadcast_to(args[3], (2, 3, *args[3].shape))
    result = jax.jit(jax.vmap(jax.vmap(call)))(nested)
    np.testing.assert_allclose(result, np.broadcast_to(expected, (2, 3, len(expected))))
    empty = jnp.empty((0, *args[3].shape), dtype=args[3].dtype)
    assert jax.jit(jax.vmap(call))(empty).shape == (0, len(expected))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_blocks_and_transposed_rhs(dtype):
    rng = np.random.default_rng(17)
    a = rng.normal(size=(67, 51)).astype(dtype)
    a[rng.random(a.shape) < 0.8] = 0
    a[4] = 0
    b_transposed = rng.normal(size=(29, 51)).astype(dtype)
    csr = csr_matrix(a)
    pattern = np.column_stack((rng.integers(0, 67, 1027), rng.integers(0, 29, 1027))).astype(np.int32)
    expected = (a @ b_transposed.T)[pattern[:, 0], pattern[:, 1]]
    pattern = np.concatenate((pattern, [[67, 0], [0, 29], [-1, 0], [0, -1]])).astype(np.int32)

    @jax.jit
    def run(data, cols, pointers, rhs, sp):
        return spp_csr_matmul(data, cols, pointers, rhs.T, sp)

    result = run(*(jnp.asarray(v) for v in
                   (csr.data, csr.indices, csr.indptr, b_transposed, pattern)))
    np.testing.assert_allclose(result[:1027], expected,
        rtol=2e-5 if dtype == np.float32 else 1e-12,
        atol=2e-6 if dtype == np.float32 else 1e-12)
    np.testing.assert_array_equal(result[1027:], 0)


@pytest.mark.parametrize("use_scan", [False, True])
def test_sparse_rtrl_matches_dense_trace_update(use_scan):
    from snapjax.algos import rtrl
    from snapjax.cells.utils import densify_jacobian_mask
    from snapjax.losses import masked_quadratic
    from snapjax.sp_jacrev import make_jacobian_projection
    from snapjax.tests.utils import get_sparse_continous_rnn

    model = get_sparse_continous_rnn(8, 5, sparsity_fraction=0.75)
    mask = model.get_snap_n_mask(2)
    dense_mask = densify_jacobian_mask(mask)
    rng = np.random.default_rng(23)
    inputs, targets = (jnp.asarray(rng.normal(size=(7, 5))) for _ in range(2))
    loss_mask = jnp.array([0., 1., 1., 0., 1., 1., 1.])

    def run(jacobian_mask):
        return rtrl(model, inputs, targets, loss_mask,
                    jacobian_mask=jacobian_mask,
                    jacobian_projection=make_jacobian_projection(jacobian_mask),
                    loss_func=masked_quadratic, use_scan=use_scan)

    actual, expected = run(mask), run(dense_mask)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)


def test_scan_uses_native_target():
    args, expected = example(np.float64)

    def run(rhs):
        return jax.lax.scan(
            lambda carry, _: (carry, spp_csr_matmul(*args[:3], carry, args[4])),
            rhs, None, length=3,
        )[1]

    compiled = jax.jit(run)
    np.testing.assert_allclose(compiled(args[3]), np.tile(expected, (3, 1)))
    hlo = compiled.lower(args[3]).as_text()
    suffix = "cuda_float64" if jax.default_backend() == "gpu" else "float64"
    assert f"snapjax_spp_csr_matmul_{suffix}" in hlo
    assert "stablehlo.dot_general" not in hlo


@pytest.mark.parametrize("position", [1, 2, 4])
def test_rejects_non_int32_indices(position):
    args, _ = example(np.float64)
    args = list(args)
    args[position] = args[position].astype(jnp.int64)
    with pytest.raises(TypeError, match="int32"):
        jax.jit(spp_csr_matmul)(*args)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.int32, jnp.complex64])
def test_rejects_unsupported_values(dtype):
    args, _ = example(np.float32)
    with pytest.raises(TypeError, match="float32 or float64"):
        spp_csr_matmul(args[0].astype(dtype), *args[1:3], args[3].astype(dtype), args[4])


def test_rejects_mismatched_dtypes_and_shapes():
    args, _ = example(np.float64)
    with pytest.raises(TypeError, match="matching"):
        spp_csr_matmul(args[0], *args[1:3], args[3].astype(jnp.float32), args[4])
    for position, replacement in [
        (0, args[0][None]), (1, args[1][:-1]),
        (2, jnp.empty(0, dtype=jnp.int32)), (3, args[3].ravel()),
        (4, jnp.zeros((2, 3), dtype=jnp.int32)),
    ]:
        bad = list(args)
        bad[position] = replacement
        with pytest.raises(ValueError):
            jax.jit(spp_csr_matmul)(*bad)
