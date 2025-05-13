#include "./chunk_dOOm_gen.hpp"
#include "./cuda_noise.cuh"
#include "./generator.hpp"
#include "block_template.hpp"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

__device__ float noise(int32_t seed, int32_t x, int32_t z, float frequency,
                       int32_t octaves) {
    return cudaNoise::repeaterSimplex(
        make_float3(x * frequency, 0.f, z * frequency), 1.0f, seed, octaves,
        2.0f, 0.5f);
}

__device__ float noise(int32_t seed, int32_t x, int32_t y, int32_t z,
                       float frequency, int32_t octaves) {
    return cudaNoise::repeaterSimplex(
        make_float3(x * frequency, y * frequency, z * frequency), 1.0f, seed,
        octaves, 2.0f, 0.5f);
}

__device__ float cave_noise(int32_t seed, int32_t x, int32_t y, int32_t z) {
    float frequency = 0.05;
    return cudaNoise::repeaterSimplex(
        make_float3(x * frequency, y * frequency, z * frequency), 1.0f, seed, 2,
        2.0f, 0.5f);
}

// All of the 2d noises should be computed here
__device__ FlatInfo ChunkGenerator::get_flat_info(int32_t seed, int32_t x,
                                                  int32_t z) {
    auto shatter = std::clamp<float>(
        noise(seed + 1, x, z, 0.001f, 3) * 0.5f + 0.5f, 0.f, 1.f);
    auto height = std::clamp<float>(
        (noise(seed, x, z, 0.01f, 3) + 1) * (32 + 32 * shatter) + 32, 0, 384);

    return FlatInfo{height, shatter};
}

// And here we use them
__device__ void ChunkGenerator::generateSmolChunk(ChunkSmol *chunk_smol,
                                                  int32_t seed, int32_t chunk_x,
                                                  int32_t chunk_y,
                                                  int32_t chunk_z,
                                                  const FlatInfo *flat_info) {
    for (uint32_t local_y = 0; local_y < 16; local_y++) {
        // Process all 4096 blocks in this subchunk
        for (uint32_t local_z = 0; local_z < 16; local_z++) {
            for (uint32_t local_x = 0; local_x < 16; local_x++) {
                int32_t absolute_y = local_y + chunk_y;
                FlatInfo flat = flat_info[local_z * 16 + local_x];

                auto threshold = std::clamp<float>(absolute_y - flat.height,
                                                   -flat.shatter * 10.f,
                                                   flat.shatter * 10.f) *
                                 0.1f / flat.shatter;
                auto density = noise(seed + 4, chunk_x + local_x, absolute_y,
                                     chunk_z + local_z, 0.1f, 2);
                bool cave = cave_noise(seed + 5, chunk_x + local_x, absolute_y,
                                       chunk_z + local_z) < -0.2f;

                if (density > threshold && !cave) {
                    chunk_smol->setBlock(local_y, local_z, 15 - local_x,
                                         make_block("minecraft:stone"));
                } else {
                    chunk_smol->setBlock(local_y, local_z, 15 - local_x,
                                         make_block("minecraft:air"));
                }
            }
        }
    }
}

__device__ void ChunkGenerator::replaceSurface(ChunkWrapper &chunk,
                                               int32_t seed) {
    for (int32_t local_z = 0; local_z < 16; local_z++) {
        for (int32_t local_x = 0; local_x < 16; local_x++) {
            FlatInfo info = chunk.get_flat_info(local_x, local_z);
            float starting_height = info.height + 15;

            for (int32_t local_y = starting_height; local_y > 32; local_y--) {
                if (chunk.isSameBlock(local_y, local_z, 15 - local_x,
                                      make_block("minecraft:stone"))) {
                    chunk.setBlock(local_y, local_z, 15 - local_x,
                                   make_block("minecraft:grass_block"));
                    break;
                }
            }
        }
    }
};

std::vector<Chunk> ChunkGenerator::generate_all(int32_t region_x,
                                                int32_t region_z) {
    std::vector<Chunk> chunks;
    chunks.reserve(32 * 32);

    // Storing all 2d noise etc.
    thrust::device_vector<FlatInfo> flats(16 * 16 * 32 * 32);

    // Use a lambda to map from index to (x, z) and call
    // getBaseTerrainHeight
    thrust::transform(
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(16 * 16 * 32 * 32), flats.begin(),
        [seed = seed, region_x, region_z] __device__(uint32_t idx) {
            // Array of [[chunk]] where chunk is locally indexed
            int32_t cell_id = idx / (16 * 16);

            int32_t cell_id_x = cell_id / 32;
            int32_t cell_id_z = cell_id % 32;

            int32_t local_x = idx % 16;
            int32_t local_z = (idx / 16) % 16;

            int32_t chunk_x = (region_x * 32 + cell_id_x) * 16;
            int32_t chunk_z = (region_z * 32 + cell_id_z) * 16;

            return get_flat_info(seed, chunk_x + local_x, chunk_z + local_z);
        });

    thrust::device_vector<ChunkSmol *> chunk_smols(32 * 32);

    for (auto cell_id = 0; cell_id < 32 * 32; cell_id++) {
        int32_t cell_id_x = cell_id / 32;
        int32_t cell_id_z = cell_id % 32;

        int32_t x = (region_x * 32 + cell_id_x) * 16;
        int32_t z = (region_z * 32 + cell_id_z) * 16;

        auto chunk = &chunks.emplace_back(x, z);
        chunk_smols[cell_id] = chunk->chunk_smols.get();
    }

    ChunkSmol **all_chunks = thrust::raw_pointer_cast(chunk_smols.data());
    FlatInfo *all_flats = thrust::raw_pointer_cast(flats.data());

    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(24 * 32 * 32),
                     [seed = seed, region_x, region_z, all_chunks,
                      all_flats] __device__(const uint32_t &smol_idx) {
                         const uint32_t i_ch = smol_idx % 24;
                         const uint32_t cell_id = smol_idx / 24;

                         ChunkSmol *raw_chunks = all_chunks[cell_id];

                         // Placement new to construct the ChunkSmol object
                         new (raw_chunks + i_ch) ChunkSmol();

                         int32_t cell_id_x = cell_id / 32;
                         int32_t cell_id_z = cell_id % 32;

                         int32_t chunk_x = (region_x * 32 + cell_id_x) * 16;
                         int32_t chunk_z = (region_z * 32 + cell_id_z) * 16;

                         // Generate this smol chunk
                         generateSmolChunk(raw_chunks + i_ch, seed, chunk_x,
                                           i_ch * 16, chunk_z,
                                           all_flats + cell_id * 16 * 16);
                     });

    cudaDeviceSynchronize();

    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(32 * 32),
                     [seed = seed, region_x, region_z, all_chunks,
                      all_flats] __device__(const uint32_t &cell_id) {
                         ChunkSmol *raw_chunks = all_chunks[cell_id];
                         ChunkWrapper wrapper(raw_chunks,
                                              all_flats + cell_id * 16 * 16);
                         replaceSurface(wrapper, seed);
                     });

    return chunks;
}
