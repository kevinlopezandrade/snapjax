#include "kernel.h"
#include <nanobind/nanobind.h>
#include <bit>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>


#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error("Invalid opaque object size");
    }
    return reinterpret_cast<const T*>(opaque);
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
    float* data = reinterpret_cast<float *>(buffers[0]);
    int* cols = reinterpret_cast<int *>(buffers[1]);
    int* indptr = reinterpret_cast<int *>(buffers[2]);
    float* B = reinterpret_cast<float *>(buffers[3]);
    int* sp = reinterpret_cast<int *>(buffers[4]);
    float* out = reinterpret_cast<float *>(buffers[5]);

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
    double* data = reinterpret_cast<double *>(buffers[0]);
    int* cols = reinterpret_cast<int *>(buffers[1]);
    int* indptr = reinterpret_cast<int *>(buffers[2]);
    double* B = reinterpret_cast<double *>(buffers[3]);
    int* sp = reinterpret_cast<int *>(buffers[4]);
    double* out = reinterpret_cast<double *>(buffers[5]);

    const CSRAndDenseDescriptor &desc = *UnpackDescriptor<CSRAndDenseDescriptor>(opaque, opaque_len);
    const int sp_size = desc.sp_size;
    const int b_cols = desc.B_dim_2;

    const int block_size = 128;
    const int grid_size = int((sp_size + block_size - 1) / block_size); // >= 1

    kernel_spp_mat_mul_double<<<grid_size, block_size, 0, stream>>>(data, cols, indptr, B, b_cols, sp, sp_size, out);
    ThrowIfError(cudaGetLastError());
}



void mask_mat_mul(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
    // This function its gonna use the kernel I define.
    float* data = reinterpret_cast<float *>(buffers[0]);
    int* cols = reinterpret_cast<int *>(buffers[1]);
    int* indptr = reinterpret_cast<int *>(buffers[2]);
    float* A = reinterpret_cast<float *>(buffers[3]);
    float* B = reinterpret_cast<float *>(buffers[4]);
    float* out = reinterpret_cast<float *>(buffers[5]);

    const SDDMMDescriptor &desc = *UnpackDescriptor<SDDMMDescriptor>(opaque, opaque_len);

    const int nnz = desc.nnz;
    const int A_num_rows = desc.A_dim_1;
    const int A_num_cols = desc.A_dim_2;
    const int B_num_rows = desc.B_dim_1;
    const int B_num_cols = desc.B_dim_2;

    float alpha = 1.0f;
    float beta = 0.0f;


    cusparseStatus_t status;
    cusparseHandle_t handle = NULL;
    status = cusparseCreate(&handle);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Did not work");
    }

    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    cusparseDnMatDescr_t matOut;

    cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, A_num_cols, A, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, B_num_cols, B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, nnz, indptr, cols, data,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matOut, A_num_rows, B_num_cols, B_num_cols, out,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseSDDMM_bufferSize(handle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                             CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);

    void* dBufferRes = NULL;
    size_t bufferSizeRes = 0;
    cusparseSparseToDense_bufferSize(handle, matC, matOut,
                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                     &bufferSizeRes
                                     );

    cudaMalloc(&dBuffer, bufferSize);
    cudaMalloc(&dBufferRes, bufferSizeRes);


    cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

    cusparseSparseToDense(handle, matC, matOut,
                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                          dBufferRes);

    cudaDeviceSynchronize();
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(dBufferRes);

    ThrowIfError(cudaGetLastError());
}
