import ctypes
from ctypes import pythonapi
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import batching, mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from numba import carray, cfunc, types
from numpy.typing import NDArray

from . import sp_ops

"""
The new primitive works as follows. It takes an CSR matrix,
with data, cols, indptr, a B dense matrix and a sparse pattern matrix,
indices of non zero entries. and performs Sp(A @ B).

It returns data, sp
"""
# WTF with this bug. I need to execute
# a dumb call to jax in order for my registration
# with xla_client does not fail. I need to further test
# this and maybe report it to the github.
jnp.empty(1)

pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,  # void* pointer
    ctypes.c_char_p,  # const char *name
    ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
pythonapi.PyCapsule_New.restype = ctypes.py_object

spp_csr_matmul_sig = types.void(
    types.voidptr,  # data
    types.voidptr,  # cols
    types.voidptr,  # indptr
    types.voidptr,  # B
    types.voidptr,  # sp
    types.voidptr,  # out
    types.intc,  # data_size
    types.intc,  # indptr_size
    types.intc,  # B_dim_1
    types.intc,  # B_dim_2
    types.intc,  # sp_dim_1
)


def pycapsule_new(ptr, name, destructor=None) -> ctypes.py_object:
    """
    Wraps a C function pointer into an XLA-compatible PyCapsule.

    Args:
        ptr: A CFFI pointer to a function
        name: A binary string
        destructor: Optional PyCapsule object run at destruction

    Returns
        a PyCapsule (ctypes.py_object)
    """
    return ctypes.pythonapi.PyCapsule_New(ptr, name, None)


@cfunc(spp_csr_matmul_sig)
def spp_csr_matmul_cfunc(
    data_ptr: NDArray[np.float32],
    cols_ptr: NDArray[np.int32],
    indptr_ptr: NDArray[np.int32],
    B_ptr: NDArray[np.float32],
    sp_ptr: NDArray[np.int32],
    out_ptr: NDArray[np.float32],
    data_size: np.int32,
    indptr_size: np.int32,
    B_dim_1: np.int32,
    B_dim_2: np.int32,
    sp_dim_1: np.int32,
):
    """
    Performs a sparse preserving matrix multiplciation
    between A(data, cols, indptr) @ B, where only
    the entries of the sparse pattern are computed.

    Args:
        data: The data array of the csr format for matrix A.
        cols: The cols array of the csr format for matrix A.
        indptr: The row pointers of the csr format for matrix A.
        B: The dense matrix B
        sp: A 2D array where each row contains the indeces of the
            non-zero entries of the sparse pattern.

    Returns:
        The data array to be used in the construction of a COO
        matrix.
    """
    N = sp_dim_1
    data = carray(data_ptr, (data_size), dtype=np.float32)
    cols = carray(cols_ptr, (data_size), dtype=np.int32)
    indptr = carray(indptr_ptr, (indptr_size), dtype=np.int32)
    B = carray(B_ptr, (B_dim_1, B_dim_2), dtype=np.float32)
    sp = carray(sp_ptr, (sp_dim_1, 2), dtype=np.int32)
    out = carray(out_ptr, (sp_dim_1), dtype=np.float32)

    for n in range(N):
        i, j = sp[n]
        row_start = indptr[i]
        row_end = indptr[i + 1]
        if row_end - row_start == 0:
            out[n] = 0
        else:
            row = data[row_start:row_end]
            local_cols_indices = cols[row_start:row_end]
            out[n] = np.dot(row, B[:, j][local_cols_indices])


xla_call_sig = types.void(
    types.voidptr,  # output_ptrs
    types.CPointer(types.voidptr),  # input_ptrs
)


def xla_spp_csr_matmul(output_ptr, input_ptrs):
    data_ptr = input_ptrs[0]
    cols_ptr = input_ptrs[1]
    indptr_ptr = input_ptrs[2]
    B_ptr = input_ptrs[3]
    sp_ptr = input_ptrs[4]

    data_size = carray(input_ptrs[5], 1, dtype=np.int32)[0]
    indptr_size = carray(input_ptrs[6], 1, dtype=np.int32)[0]
    B_dim_1 = carray(input_ptrs[7], 1, dtype=np.int32)[0]
    B_dim_2 = carray(input_ptrs[8], 1, dtype=np.int32)[0]
    sp_dim_1 = carray(input_ptrs[9], 1, dtype=np.int32)[0]

    out_ptr = output_ptr

    spp_csr_matmul_cfunc(
        data_ptr,
        cols_ptr,
        indptr_ptr,
        B_ptr,
        sp_ptr,
        out_ptr,
        data_size,
        indptr_size,
        B_dim_1,
        B_dim_2,
        sp_dim_1,
    )


def spp_csr_matmul(data, cols, indptr, RHS, sp):
    return spp_csr_matmul_p.bind(data, cols, indptr, RHS, sp)


spp_csr_matmul_p = core.Primitive("spp_csr_matmul")
spp_csr_matmul_p.def_impl(partial(xla.apply_primitive, spp_csr_matmul_p))


def spp_csr_matmul_abstract_eval(data, cols, indptr, RHS, sp):
    return core.ShapedArray(shape=(sp.shape[0],), dtype=data.dtype)


spp_csr_matmul_p.def_abstract_eval(spp_csr_matmul_abstract_eval)

xla_client.register_custom_call_target(
    b"spp_csr_matmul",
    pycapsule_new(
        cfunc(xla_call_sig)(xla_spp_csr_matmul).address, b"xla._CUSTOM_CALL_TARGET"
    ),
    "cpu",
)


def _get_layout(shape):
    return range(len(shape) - 1, -1, -1)


def _spp_csr_matmul_lowering(ctx, data, cols, indptr, RHS, sp):
    data_type = ir.RankedTensorType(data.type)
    cols_type = ir.RankedTensorType(cols.type)
    indptr_type = ir.RankedTensorType(indptr.type)
    RHS_type = ir.RankedTensorType(RHS.type)
    sp_type = ir.RankedTensorType(sp.type)

    data_size = mlir.ir_constant(data_type.shape[0])
    indptr_size = mlir.ir_constant(indptr_type.shape[0])
    b_dim_1 = mlir.ir_constant(RHS_type.shape[0])
    b_dim_2 = mlir.ir_constant(RHS_type.shape[1])
    sp_dim_1 = mlir.ir_constant(sp_type.shape[0])

    out_shape = (sp_type.shape[0],)

    out = mlir.custom_call(
        "spp_csr_matmul",
        result_types=[ir.RankedTensorType.get(out_shape, data_type.element_type)],
        operands=[
            data,
            cols,
            indptr,
            RHS,
            sp,
            data_size,
            indptr_size,
            b_dim_1,
            b_dim_2,
            sp_dim_1,
        ],
        operand_layouts=[
            _get_layout(data_type.shape),
            _get_layout(cols_type.shape),
            _get_layout(indptr_type.shape),
            _get_layout(RHS_type.shape),
            _get_layout(sp_type.shape),
            (),
            (),
            (),
            (),
            (),
        ],
        result_layouts=[_get_layout(out_shape)],
    ).results

    return out


mlir.register_lowering(spp_csr_matmul_p, _spp_csr_matmul_lowering, platform="cpu")

# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in sp_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def _spp_csr_matmul_cuda_lowering(ctx, data, cols, indptr, RHS, sp):
    data_type = ir.RankedTensorType(data.type)
    cols_type = ir.RankedTensorType(cols.type)
    indptr_type = ir.RankedTensorType(indptr.type)
    RHS_type = ir.RankedTensorType(RHS.type)
    sp_type = ir.RankedTensorType(sp.type)

    descriptor_opaque = sp_ops.build_csr_and_dense_descriptor(
        data_type.shape[0],
        indptr_type.shape[0],
        sp_type.shape[0],
        RHS_type.shape[0],
        RHS_type.shape[1],
    )

    out_shape = (sp_type.shape[0],)

    out = mlir.custom_call(
        b"spp_csr_matmul_cuda",
        result_types=[ir.RankedTensorType.get(out_shape, data_type.element_type)],
        operands=[
            data,
            cols,
            indptr,
            RHS,
            sp,
        ],
        operand_layouts=[
            _get_layout(data_type.shape),
            _get_layout(cols_type.shape),
            _get_layout(indptr_type.shape),
            _get_layout(RHS_type.shape),
            _get_layout(sp_type.shape),
        ],
        result_layouts=[_get_layout(out_shape)],
        backend_config=descriptor_opaque,
    ).results

    return out


mlir.register_lowering(
    spp_csr_matmul_p,
    _spp_csr_matmul_cuda_lowering,
    platform="gpu",
)

# Batching

# Jax will do loop unrolling here since spp_csr_matmul_batch must be traceable
# unless I make my primitive closed under batching for which I should code a
# bit more, it should not be that hard though. Just need to spend time in Numba
# for the CPU lowering and in CUDA for the GPU. In the mean time just do loop
# unrolling. The loop unrolling done here will be equal to the size of the
# batch.


def spp_csr_matmul_batch(args, batch_axes):
    data, cols, indptr, RHS, sp = args

    if data.ndim == 1:
        # Then I'm batching only over RHS
        data_batched = [
            spp_csr_matmul(data, cols, indptr, RHS[i], sp) for i in range(RHS.shape[0])
        ]
        data_batched = jnp.stack(data_batched)
        return data_batched, 0
    if data.ndim == 2:
        # Then I'm batching over data and RHS.
        data_batched = [
            spp_csr_matmul(data[i], cols, indptr, RHS[i], sp)
            for i in range(data.shape[0])
        ]
        data_batched = jnp.stack(data_batched)
        return data_batched, 0


batching.primitive_batchers[spp_csr_matmul_p] = spp_csr_matmul_batch
