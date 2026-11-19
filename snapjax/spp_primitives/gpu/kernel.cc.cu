#include "kernel.h"
#include <cuda_runtime.h>

namespace {

// The original sparse-preserving algorithm: one thread per requested output,
// accumulating over its CSR row without constructing the dense product.
template <typename T>
__global__ void kernel_spp_mat_mul(const T* data, const int32_t* cols,
    const int32_t* indptr, const T* rhs, const int32_t* pattern,
    int32_t rows, int32_t rhs_rows, int32_t rhs_cols, int32_t count, T* output) {
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const int32_t i = pattern[2 * tid];
    const int32_t j = pattern[2 * tid + 1];
    T sum = T(0);
    // BCOO pads patterns with out-of-bounds indices. Match the CPU semantics.
    if (i >= 0 && i < rows && j >= 0 && j < rhs_cols) {
        for (int32_t pos = indptr[i]; pos < indptr[i + 1]; ++pos) {
            const int32_t k = cols[pos];
            if (k >= 0 && k < rhs_rows) {
                sum += data[pos] * rhs[int64_t(k) * rhs_cols + j];
            }
        }
    }
    output[tid] = sum;
}

template <typename T>
cudaError_t Launch(cudaStream_t stream, const T* data, const int32_t* cols,
    const int32_t* indptr, const T* rhs, const int32_t* pattern,
    int32_t rows, int32_t rhs_rows, int32_t rhs_cols, int32_t count, T* output) {
    // Launching a grid with zero blocks is invalid in CUDA.
    if (count == 0) return cudaSuccess;
    constexpr int block_size = 256;
    const int grid_size = int((int64_t(count) + block_size - 1) / block_size);
    kernel_spp_mat_mul<<<grid_size, block_size, 0, stream>>>(
        data, cols, indptr, rhs, pattern, rows, rhs_rows, rhs_cols, count, output);
    return cudaGetLastError();
}

}  // namespace

cudaError_t SppMatmul(cudaStream_t stream, const float* data,
    const int32_t* cols, const int32_t* indptr, const float* rhs,
    const int32_t* pattern, int32_t rows, int32_t rhs_rows,
    int32_t rhs_cols, int32_t count, float* output) {
    return Launch(stream, data, cols, indptr, rhs, pattern,
                  rows, rhs_rows, rhs_cols, count, output);
}

cudaError_t SppMatmul(cudaStream_t stream, const double* data,
    const int32_t* cols, const int32_t* indptr, const double* rhs,
    const int32_t* pattern, int32_t rows, int32_t rhs_rows,
    int32_t rhs_cols, int32_t count, double* output) {
    return Launch(stream, data, cols, indptr, rhs, pattern,
                  rows, rhs_rows, rhs_cols, count, output);
}
