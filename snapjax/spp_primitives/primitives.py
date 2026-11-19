"""Sparse-preserving multiplication with native CPU and optional CUDA kernels.

The Numba callbacks use XLA's unified, status-returning custom-call ABI. Keep them
alive for as long as compiled JAX executables may hold their function pointers.
The separately built snapjax_cuda extension supplies typed GPU FFI handlers.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching, mlir, xla
from numba import carray, cfunc, types


_CALL_SIGNATURE = types.void(
    types.voidptr, types.CPointer(types.voidptr), types.voidptr, types.uintp, types.voidptr
)


def _make_cpu_kernel(dtype):
    @cfunc(_CALL_SIGNATURE)
    def kernel(output_ptr, input_ptrs, opaque, opaque_len, status):
        data_size = carray(input_ptrs[5], 1, dtype=np.int32)[0]
        indptr_size = carray(input_ptrs[6], 1, dtype=np.int32)[0]
        b_rows = carray(input_ptrs[7], 1, dtype=np.int32)[0]
        b_cols = carray(input_ptrs[8], 1, dtype=np.int32)[0]
        sp_size = carray(input_ptrs[9], 1, dtype=np.int32)[0]

        data = carray(input_ptrs[0], data_size, dtype=dtype)
        cols = carray(input_ptrs[1], data_size, dtype=np.int32)
        indptr = carray(input_ptrs[2], indptr_size, dtype=np.int32)
        rhs = carray(input_ptrs[3], (b_rows, b_cols), dtype=dtype)
        sp = carray(input_ptrs[4], (sp_size, 2), dtype=np.int32)
        out = carray(output_ptr, sp_size, dtype=dtype)

        for n in range(sp_size):
            i, j = sp[n]
            total = dtype(0)
            # BCOO may pad its output pattern with out-of-bounds indices.
            if 0 <= i < indptr_size - 1 and 0 <= j < b_cols:
                for pos in range(indptr[i], indptr[i + 1]):
                    k = cols[pos]
                    if 0 <= k < b_rows:
                        total += data[pos] * rhs[k, j]
            out[n] = total

    return kernel


_CPU_KERNELS = {
    np.dtype(dtype): _make_cpu_kernel(dtype) for dtype in (np.float32, np.float64)
}
_CPU_TARGETS = {
    dtype: f"snapjax_spp_csr_matmul_{dtype.name}" for dtype in _CPU_KERNELS
}
for _dtype, _kernel in _CPU_KERNELS.items():
    ffi.register_ffi_target(
        _CPU_TARGETS[_dtype], ffi.pycapsule(_kernel.ctypes), platform="cpu", api_version=0
    )

try:
    import snapjax_cuda as _cuda
except ModuleNotFoundError as error:
    if error.name != "snapjax_cuda":
        raise
    _cuda = None

_CUDA_TARGETS = {
    dtype: f"snapjax_spp_csr_matmul_cuda_{dtype.name}" for dtype in _CPU_KERNELS
}
if _cuda is not None:
    for _name, _handler in _cuda.registrations().items():
        ffi.register_ffi_target(_name, _handler, platform="CUDA", api_version=1)


def spp_csr_matmul(data, cols, indptr, RHS, sp):
    """Return ``(A @ RHS)[sp[:, 0], sp[:, 1]]`` without forming ``A @ RHS``.

    ``A`` is a valid CSR matrix described by one-dimensional ``data``, ``cols``,
    and ``indptr`` arrays. ``data`` and the two-dimensional ``RHS`` must have
    the same float32 or float64 dtype. All indices must be int32; ``sp`` has
    shape ``(n_outputs, 2)``. Out-of-bounds pattern entries produce zero, as for
    padded BCOO arrays. CSR row pointers must be monotone, start at zero, and
    end at ``len(data)``. The native kernel trusts these structural invariants.

    Supports CPU and optional CUDA execution, JIT, and vmap. No autodiff rule is provided for
    this trace-update operation. Enable JAX's x64 mode before using float64.
    """
    return spp_csr_matmul_p.bind(
        *(jnp.asarray(arg) for arg in (data, cols, indptr, RHS, sp))
    )


spp_csr_matmul_p = Primitive("spp_csr_matmul")
spp_csr_matmul_p.def_impl(partial(xla.apply_primitive, spp_csr_matmul_p))


def _abstract_eval(data, cols, indptr, rhs, sp):
    if data.ndim != 1 or cols.shape != data.shape or indptr.ndim != 1:
        raise ValueError("CSR data and cols must be matching vectors; indptr must be a vector")
    if indptr.shape[0] < 1:
        raise ValueError("CSR indptr must contain at least the initial zero")
    if rhs.ndim != 2 or sp.ndim != 2 or sp.shape[1] != 2:
        raise ValueError("RHS must be a matrix and sp must have shape (n_outputs, 2)")
    if data.dtype not in _CPU_KERNELS or rhs.dtype != data.dtype:
        raise TypeError("data and RHS must have matching float32 or float64 dtypes")
    if any(arg.dtype != np.dtype(np.int32) for arg in (cols, indptr, sp)):
        raise TypeError("cols, indptr, and sp must have dtype int32")
    if any(dim > np.iinfo(np.int32).max for arg in (data, indptr, rhs, sp) for dim in arg.shape):
        raise ValueError("Kernel dimensions must fit in int32")
    return ShapedArray((sp.shape[0],), data.dtype)


spp_csr_matmul_p.def_abstract_eval(_abstract_eval)


def _cpu_lowering(ctx, data, cols, indptr, rhs, sp):
    avals = ctx.avals_in
    sizes = (avals[0].shape[0], avals[2].shape[0], *avals[3].shape, avals[4].shape[0])
    size_avals = [ShapedArray((), np.dtype(np.int32)) for _ in sizes]
    return ffi.ffi_lowering(
        _CPU_TARGETS[avals[0].dtype],
        # Unified ABI: output, inputs, opaque data, opaque size, status.
        # Leaving status untouched signals success; no opaque data is used.
        # This lowering version is distinct from registration's api_version=0.
        api_version=3,
    )(
        ctx.replace(avals_in=[*avals, *size_avals]),
        data, cols, indptr, rhs, sp,
        *[mlir.ir_constant(np.int32(size)) for size in sizes],
    )


mlir.register_lowering(spp_csr_matmul_p, _cpu_lowering, platform="cpu")


def _cuda_lowering(ctx, data, cols, indptr, rhs, sp):
    if _cuda is None:
        raise RuntimeError(
            "The SnapJax CUDA extension is not installed. On Linux with a CUDA toolkit, "
            "run `uv pip install ./snapjax/spp_primitives/gpu` from the source checkout."
        )
    return ffi.ffi_lowering(_CUDA_TARGETS[ctx.avals_in[0].dtype], api_version=4)(
        ctx, data, cols, indptr, rhs, sp
    )


mlir.register_lowering(spp_csr_matmul_p, _cuda_lowering, platform="cuda")


def _batch(args, axes):
    # Map only batched operands: shared CSR structures/RHS stay shared, and
    # lax.map keeps the compiled graph independent of the batch size.
    mapped = tuple(
        jnp.moveaxis(arg, axis, 0)
        for arg, axis in zip(args, axes)
        if axis is not None
    )

    def one(values):
        values = iter(values)
        operands = [
            next(values) if axis is not None else arg
            for arg, axis in zip(args, axes)
        ]
        return spp_csr_matmul_p.bind(*operands)

    return jax.lax.map(one, mapped), 0


batching.primitive_batchers[spp_csr_matmul_p] = _batch
