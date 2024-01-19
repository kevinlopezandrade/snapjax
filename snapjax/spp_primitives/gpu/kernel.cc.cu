#include "kernel.h"
#include <nanobind/nanobind.h>
#include <bit>
#include <cuda_runtime_api.h>
#include <iostream>


template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error("Invalid opaque object size");
    }
    return std::bit_cast<const T*>(opaque);
}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

__global__ void kernel_spp_mat_mul(
        float *data, int *cols, int *indptr, float *B, int B_cols, int *sp, int sp_size, float *out) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if (tid < sp_size) {
        int i = sp[tid * 2];
        int j = sp[tid * 2 + 1];

        // For a valid sparse pattern
        // i is guranteed to not pass
        // the limit of the indptr.
        int row_start = indptr[i];
        int row_end = indptr[i + 1];

        int size = row_end - row_start;
        if (size == 0) {
            out[tid] = 0.0;
        } else {
            float res = 0.0;
            int pos;
            int z;
            for (z = 0; z < size; z++) {
                pos = row_start + z;
                res += data[pos] * B[cols[pos] * B_cols + j];
            }
            out[tid] = res;
        }

    }
}

__global__ void kernel_spp_mat_mul_double(
        double *data, int *cols, int *indptr, double *B, int B_cols, int *sp, int sp_size, double *out) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if (tid < sp_size) {
        int i = sp[tid * 2];
        int j = sp[tid * 2 + 1];

        // For a valid sparse pattern
        // i is guranteed to not pass
        // the limit of the indptr.
        int row_start = indptr[i];
        int row_end = indptr[i + 1];

        int size = row_end - row_start;
        if (size == 0) {
            out[tid] = 0.0;
        } else {
            double res = 0.0;
            int pos;
            int z;
            for (z = 0; z < size; z++) {
                pos = row_start + z;
                res += data[pos] * B[cols[pos] * B_cols + j];
            }
            out[tid] = res;
        }

    }
}

// XLA Signature call
void spp_mat_mul(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    // This function its gonna use the kernel I define.
    float* data = static_cast<float *>(buffers[0]);
    int* cols = static_cast<int *>(buffers[1]);
    int* indptr = static_cast<int *>(buffers[2]);
    float* B = static_cast<float *>(buffers[3]);
    int* sp = static_cast<int *>(buffers[4]);
    float* out = static_cast<float *>(buffers[5]);

    const CSRAndDenseDescriptor &desc = *UnpackDescriptor<CSRAndDenseDescriptor>(opaque, opaque_len);
    const int sp_size = desc.sp_size;
    const int b_cols = desc.B_dim_2;

    const int block_size = 128;
    const int grid_size = int((sp_size + block_size - 1) / block_size); // >= 1

    kernel_spp_mat_mul<<<grid_size, block_size, 0, stream>>>(data, cols, indptr, B, b_cols, sp, sp_size, out);
    ThrowIfError(cudaGetLastError());
}

// XLA Signature call
void spp_mat_mul_double(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    // This function its gonna use the kernel I define.
    double* data = static_cast<double *>(buffers[0]);
    int* cols = static_cast<int *>(buffers[1]);
    int* indptr = static_cast<int *>(buffers[2]);
    double* B = static_cast<double *>(buffers[3]);
    int* sp = static_cast<int *>(buffers[4]);
    double* out = static_cast<double *>(buffers[5]);

    const CSRAndDenseDescriptor &desc = *UnpackDescriptor<CSRAndDenseDescriptor>(opaque, opaque_len);
    const int sp_size = desc.sp_size;
    const int b_cols = desc.B_dim_2;

    const int block_size = 128;
    const int grid_size = int((sp_size + block_size - 1) / block_size); // >= 1

    kernel_spp_mat_mul_double<<<grid_size, block_size, 0, stream>>>(data, cols, indptr, B, b_cols, sp, sp_size, out);
    ThrowIfError(cudaGetLastError());
}
