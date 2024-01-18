#include <nanobind/nanobind.h>
#include <bit>
#include "kernel.h"
#include <cuda_runtime_api.h>

// Required by JAX to encapsulate in _CUSTOM_CALL_TARGET
template <typename T>
nanobind::capsule EncapsulateFunction(T *fn) {
    return nanobind::capsule(std::bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
    return std::string(std::bit_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T>
nanobind::bytes PackDescriptor(const T& descriptor) {
    std::string s = PackDescriptorAsString(descriptor);
    return nanobind::bytes(s.data(), s.size());
}

nanobind::dict Registrations() {
    nanobind::dict dict;
    dict["spp_csr_matmul_cuda"] = EncapsulateFunction(spp_mat_mul);
    return dict;
}

nanobind::bytes BuildCSRAndDenseDescriptor(
        int data_size, int indptr_size, int sp_size,int B_dim_1, int B_dim_2)
{
    return PackDescriptor(CSRAndDenseDescriptor{data_size, indptr_size, sp_size, B_dim_1, B_dim_2});
}

NB_MODULE(sp_ops, m) {
    m.def("registrations", &Registrations);
    m.def("build_csr_and_dense_descriptor", &BuildCSRAndDenseDescriptor);
}
