#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// Compute only the requested entries of a CSR--dense product on XLA's stream.
cudaError_t SppMatmul(cudaStream_t stream, const float* data,
    const int32_t* cols, const int32_t* indptr, const float* rhs,
    const int32_t* pattern, int32_t rows, int32_t rhs_rows,
    int32_t rhs_cols, int32_t count, float* output);
cudaError_t SppMatmul(cudaStream_t stream, const double* data,
    const int32_t* cols, const int32_t* indptr, const double* rhs,
    const int32_t* pattern, int32_t rows, int32_t rhs_rows,
    int32_t rhs_cols, int32_t count, double* output);
