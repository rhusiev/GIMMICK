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

// Device function to generate a single ChunkSmol
__device__ void generateSmolChunk(ChunkSmol *chunk_smol, int32_t i_ch,
                                  const float *heights) {
    for (uint32_t i_y = 0; i_y < 16; i_y++) {
        // Process all 4096 blocks in this subchunk
        for (uint32_t i_z = 0; i_z < 16; i_z++) {
            for (uint32_t i_x = 0; i_x < 16; i_x++) {
                // Get height value from device vector
                float height = heights[i_z * 16 + i_x];

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

Chunk *ChunkGenerator::generate_all(int32_t region_x, int32_t region_z) {
    Chunk *chunks = static_cast<Chunk *>(malloc(sizeof(Chunk) * 32 * 32));

    thrust::device_vector<float> heights(16 * 16 * 32 * 32);

    // Use a lambda to map from index to (x, z) and call
    // getBaseTerrainHeight
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(16 * 16 * 32 * 32), heights.begin(),
        [region_x, region_z] __device__(uint32_t idx) {
            // Array of [[chunk]] where chunk is locally indexed
            int32_t cell_id = idx / (16 * 16);

            int32_t cell_id_x = cell_id / 32;
            int32_t cell_id_z = cell_id % 32;

            int32_t local_x = idx % 16;
            int32_t local_z = (idx / 16) % 16;

            int32_t chunk_x = (region_x * 32 + cell_id_x) * 16;
            int32_t chunk_z = (region_z * 32 + cell_id_z) * 16;

            return getBaseTerrainHeight(chunk_x + local_x, chunk_z + local_z);
        });

    thrust::device_vector<ChunkSmol *> chunk_smols(32 * 32);

    for (auto cell_id = 0; cell_id < 32 * 32; cell_id++) {
        int32_t cell_id_x = cell_id / 32;
        int32_t cell_id_z = cell_id % 32;
        float *cell_heights = thrust::raw_pointer_cast(heights.data()) +
                              (cell_id_x * 16 * 16 * 32 + cell_id_z * 16 * 16);

        int32_t x = (region_x * 32 + cell_id_x) * 16;
        int32_t z = (region_z * 32 + cell_id_z) * 16;

        new (chunks + cell_id)
            Chunk(x, z); // Placement new to construct the Chunk object
        auto chunk = chunks + cell_id; // Get the pointer to the current chunk

        chunk_smols[cell_id] = chunk->chunk_smols.get();
    }

    ChunkSmol **all_chunks = thrust::raw_pointer_cast(chunk_smols.data());
    float *all_heights = thrust::raw_pointer_cast(heights.data());

    thrust::for_each(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(24 * 32 * 32),
        [all_chunks, all_heights] __device__(const uint32_t &smol_idx) {
            const uint32_t i_ch = smol_idx % 24;
            const uint32_t cell_id = smol_idx / 24;

            ChunkSmol *raw_chunks = all_chunks[cell_id];

            // Placement new to construct the ChunkSmol object
            new (raw_chunks + i_ch) ChunkSmol();

            // Generate this smol chunk
            generateSmolChunk(raw_chunks + i_ch, i_ch,
                              all_heights + cell_id * 16 * 16);
        });

    return chunks;
}

__device__ float ChunkGenerator::getBaseTerrainHeight(float x, float z) {
    auto noise = cudaNoise::repeaterSimplex(
        make_float3(x * 0.01f, 0.f, z * 0.01f), 1.0f, 0, 3, 2.0f, 0.5f);

    // Simple height calculation: noise range -1 to 1, convert to 32-160
    // range
    auto height = (noise + 1) * 64 + 32; // 32 - 160

    return std::clamp<double>(height, 0.0, 320.0);
}
