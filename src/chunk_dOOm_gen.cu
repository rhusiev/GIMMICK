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

Chunk::Chunk(int32_t x, int32_t z) : x(x), z(z) {
    CHECK_CUDA_ERROR(cudaMalloc(&chunk_smols, sizeof(ChunkSmol) * 24));
    // Initialize with 24 subchunks - will be filled by the generator
}

Chunk::~Chunk() {
    std::cout << "Freeing chunk memory" << std::endl;
    std::cout << "chunk_smols: " << (void *)chunk_smols << std::endl;
    // Free the allocated memory for chunk_smols
    CHECK_CUDA_ERROR(cudaFree(chunk_smols));
}

Chunk ChunkGenerator::generate(int32_t x, int32_t z) {
    Chunk chunk{x, z};

    // Create a device vector to store heights
    thrust::device_vector<float> heights(16 * 16);

    // Use a lambda to map from index to (x,z) and call getBaseTerrainHeight
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(16 * 16), heights.begin(),
        [this, chunk_x = chunk.x, chunk_z = chunk.z] __device__(uint32_t idx) {
            int32_t local_x = idx % 16;
            int32_t local_z = idx / 16;
            return getBaseTerrainHeight(chunk_x + local_x, chunk_z + local_z);
        });

    // Apply only basic stone generation
    generateBaseStructure(chunk, heights);

    return chunk;
}

// Device function to generate a single ChunkSmol
__device__ void generateSmolChunk(ChunkSmol *chunk_smol, int32_t i_ch,
                                  const float *heights) {
    // Process all 4096 blocks in this subchunk
    for (uint32_t i_z = 0; i_z < 16; i_z++) {
        for (uint32_t i_x = 0; i_x < 16; i_x++) {
            // Get height value from device vector
            float height = heights[i_z * 16 + i_x];
            for (uint32_t i_y = 0; i_y < 16; i_y++) {

                // Calculate absolute Y coordinate in the world
                int32_t absolute_y = i_y + i_ch * 16;

                // Set the appropriate block based on height
                if (height > absolute_y) {
                    chunk_smol->setBlock(i_y, i_z, 15 - i_x,
                                         make_block("minecraft:stone"));
                } else {
                    chunk_smol->setBlock(i_y, i_z, 15 - i_x,
                                         make_block("minecraft:air"));
                }
            }
        }
    }
}

void ChunkGenerator::generateBaseStructure(
    Chunk &chunk, const thrust::device_vector<float> &d_heights) {
    // Get raw pointers for use in the device code
    ChunkSmol *raw_chunks = chunk.chunk_smols;
    const float *raw_heights = thrust::raw_pointer_cast(d_heights.data());

    // Process all 24 smol chunks in parallel using thrust::for_each with lambda
    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(24),
        [raw_chunks, raw_heights] __device__(const uint32_t &i_ch) {
            // Placement new to construct the ChunkSmol object
            new (raw_chunks + i_ch) ChunkSmol();

            // Generate this smol chunk
            generateSmolChunk(raw_chunks + i_ch, i_ch, raw_heights);
        });

    std::cout << "Base structure generated" << std::endl;
}

__device__ float ChunkGenerator::getBaseTerrainHeight(float x, float z) {
    auto noise = cudaNoise::repeaterSimplex(make_float3(x, 0.f, z), 0.1f, 0, 3,
                                            2.0f, 0.5f);

    // Simple height calculation: noise range -1 to 1, convert to 32-160 range
    auto height = (noise + 1) * 64 + 32; // 32 - 160

    return std::clamp<double>(height, 0.0, 320.0);
}

// Legacy function that uses the new ChunkGenerator
Chunk generate_chunk(int32_t x, int32_t z) {
    static ChunkGenerator generator;
    return generator.generate(x, z);
}
