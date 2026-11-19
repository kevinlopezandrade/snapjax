#include <limits>
#include <nanobind/nanobind.h>
#include "xla/ffi/api/ffi.h"
#include "kernel.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

template <ffi::DataType dtype>
ffi::Error SppImpl(cudaStream_t stream, ffi::Buffer<dtype, 1> data,
    ffi::Buffer<ffi::S32, 1> cols, ffi::Buffer<ffi::S32, 1> indptr,
    ffi::Buffer<dtype, 2> rhs, ffi::Buffer<ffi::S32, 2> pattern,
    ffi::ResultBuffer<dtype, 1> output) {
    const auto nnz = data.dimensions()[0];
    const auto pointers = indptr.dimensions()[0];
    const auto rhs_rows = rhs.dimensions()[0];
    const auto rhs_cols = rhs.dimensions()[1];
    const auto count = pattern.dimensions()[0];
    if (cols.dimensions()[0] != nnz || pointers < 1 ||
        pattern.dimensions()[1] != 2 || output->dimensions()[0] != count) {
        return ffi::Error::InvalidArgument("Invalid sparse product buffer shapes");
    }
    for (auto size : {nnz, pointers, rhs_rows, rhs_cols, count}) {
        if (size > std::numeric_limits<int32_t>::max()) {
            return ffi::Error::InvalidArgument("Kernel dimensions must fit in int32");
        }
    }
    const auto status = SppMatmul(stream, data.typed_data(), cols.typed_data(),
        indptr.typed_data(), rhs.typed_data(), pattern.typed_data(),
        int32_t(pointers - 1), int32_t(rhs_rows), int32_t(rhs_cols),
        int32_t(count), output->typed_data());
    if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SppFloat32, SppImpl<ffi::F32>,
    (ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32, 1>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Arg<ffi::Buffer<ffi::F32, 2>>()
        .Arg<ffi::Buffer<ffi::S32, 2>>()
        .Ret<ffi::Buffer<ffi::F32, 1>>()));

XLA_FFI_DEFINE_HANDLER_SYMBOL(SppFloat64, SppImpl<ffi::F64>,
    (ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64, 1>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Arg<ffi::Buffer<ffi::S32, 1>>()
        .Arg<ffi::Buffer<ffi::F64, 2>>()
        .Arg<ffi::Buffer<ffi::S32, 2>>()
        .Ret<ffi::Buffer<ffi::F64, 1>>()));

NB_MODULE(snapjax_cuda, module) {
    module.def("registrations", [] {
        nb::dict targets;
        targets["snapjax_spp_csr_matmul_cuda_float32"] = nb::capsule(
            reinterpret_cast<void*>(SppFloat32), "xla._CUSTOM_CALL_TARGET");
        targets["snapjax_spp_csr_matmul_cuda_float64"] = nb::capsule(
            reinterpret_cast<void*>(SppFloat64), "xla._CUSTOM_CALL_TARGET");
        return targets;
    });
}
