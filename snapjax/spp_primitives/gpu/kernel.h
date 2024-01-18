#include <cuda_runtime_api.h>

// I need a struct for the data.
struct CSRAndDenseDescriptor{
    int data_size;
    int indptr_size;
    int sp_size;
    int B_dim_1;
    int B_dim_2;
};

void spp_mat_mul(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
