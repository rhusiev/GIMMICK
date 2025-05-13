#include "./chunk_dOOm_gen.hpp"
#include "./cuda_noise.cuh"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call)                                                 \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << " at line "           \
                      << __LINE__ << ": " << cudaGetErrorString(err) << " ("   \
                      << err << ")" << std::endl;                              \
            exit(1);                                                           \
        }                                                                      \
    }

std::unique_ptr<ChunkSmol, CudaDeleter> alloc_helper() {
    ChunkSmol *chunk_smol;
    CHECK_CUDA_ERROR(cudaMalloc(&chunk_smol, sizeof(ChunkSmol) * 24));
    return std::unique_ptr<ChunkSmol, CudaDeleter>(chunk_smol);
}

Chunk::Chunk(int32_t x, int32_t z) : x(x), z(z), chunk_smols(alloc_helper()) {
    // Initialize with 24 subchunks - will be filled by the generator
}
