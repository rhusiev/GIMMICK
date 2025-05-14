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
    return cudaNoise::repeaterSimplex(make_float3(x, 0.f, z), frequency, seed,
                                      octaves, 2.0f, 0.5f);
}

__device__ float noise(int32_t seed, int32_t x, int32_t y, int32_t z,
                       float frequency, int32_t octaves) {
    return cudaNoise::repeaterSimplex(make_float3(x, y, z), frequency, seed,
                                      octaves, 2.0f, 0.5f);
}

__device__ bool cave_noise(int32_t seed, int32_t x, int32_t y, int32_t z) {
    float noodle1 = noise(seed, x, y, z, 0.01f, 2);
    float noodle2 = noise(seed + 1, x, y, z, 0.02f, 2);

    float cavern = noise(seed + 2, x, y, z, 0.01, 3);

    float noodle1_probability =
        std::max(1.f - std::abs(noodle1 - 0.2f) * 10.0f, 0.f);
    float noodle2_probability =
        std::max(1.f - std::abs(noodle2 - 0.2f) * 10.0f, 0.f);

    float cavern_probability = std::max(0.f, cavern - 0.6f);

    return (noodle1_probability * noodle2_probability + cavern_probability) >
           0.1f;
}

// All of the 2d noises should be computed here
__device__ FlatInfo ChunkGenerator::get_flat_info(int32_t seed, int32_t x,
                                                  int32_t z) {
    auto continentalness =
        std::clamp<float>((noise(seed, x, z, 0.005f, 2) + 1) * 0.5f, 0, 1);

    auto shatter =
        std::clamp<float>(noise(seed + 1, x, z, 0.005f, 2) * 0.5f + 0.5f, 0.f,
                          1.f) *
        continentalness * std::clamp<float>(continentalness * 2 - 0.5, 0, 1);
    auto heightNoise = (noise(seed, x, z, 0.01f, 3) + 2.f + shatter) / 4.f;
    auto height = heightNoise * continentalness * 64.f + 48.f;

    auto temperature = std::clamp<float>(
        noise(seed + 2, x, z, 0.005, 4) * 0.5f + 0.5f, 0.f, 1.f);
    auto vegetation = std::clamp<float>(noise(seed + 3, x, z, 0.25f, 1) * 0.5f +
                                            0.5f + shatter * 0.1f,
                                        0.f, 1.f);

    return FlatInfo{height, shatter, temperature, vegetation, continentalness};
}

// And here we use them
__device__ void
ChunkGenerator::generateSmolChunk(ChunkSmol *chunk_smol, int32_t seed,
                                  int32_t chunk_x, int32_t chunk_y,
                                  int32_t chunk_z, const FlatInfo *flat_info,
                                  const VolumetricInfo *volumetric_info) {
    for (uint32_t local_y = 0; local_y < 16; local_y++) {
        // Process all 4096 blocks in this subchunk
        for (uint32_t local_z = 0; local_z < 16; local_z++) {
            for (uint32_t local_x = 0; local_x < 16; local_x++) {
                int32_t absolute_y = local_y + chunk_y - 64;
                FlatInfo flat = flat_info[local_z * 16 + local_x];
                VolumetricInfo volumetric =
                    volumetric_info[local_y * 16 * 16 + local_z * 16 + local_x];

                auto threshold = std::clamp<float>(absolute_y - flat.height,
                                                   -flat.shatter * 10.f,
                                                   flat.shatter * 10.f) *
                                 0.1f / flat.shatter;

                if (volumetric.density > threshold && !volumetric.cave) {
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
    int32_t surface_heights[16][16];

    for (int32_t local_z = 0; local_z < 16; local_z++) {
        for (int32_t local_x = 0; local_x < 16; local_x++) {
            FlatInfo info = chunk.get_flat_info(local_x, local_z);

            float starting_height = std::max<float>(info.height + 15 + 64,
                                                    64 + 64); // because local

            bool cold = info.temperature < 0.4f;
            bool hit_surface = false;
            surface_heights[local_x][local_z] = 0;

            for (int32_t local_y = starting_height; local_y > 32; local_y--) {
                int32_t absolute_y = local_y - 64;

                if (!surface_heights[local_x][local_z] && absolute_y < 65 &&
                    info.continentalness < 0.75) {
                    if (chunk.isSameBlock(local_y, local_z, 15 - local_x,
                                          make_block("minecraft:air"))) {
                        chunk.setBlock(local_y, local_z, 15 - local_x,
                                       make_block("minecraft:water"));
                    }
                }

                bool air_above =
                    chunk.isSameBlock(local_y + 1, local_z, 15 - local_x,
                                      make_block("minecraft:air"));
                bool water_above =
                    chunk.isSameBlock(local_y + 1, local_z, 15 - local_x,
                                      make_block("minecraft:water"));

                if (cold && air_above &&
                    info.vegetation > (info.temperature - 0.1f) &&
                    chunk.isSameBlock(local_y, local_z, 15 - local_x,
                                      make_block("minecraft:water"))) {
                    chunk.setBlock(local_y, local_z, 15 - local_x,
                                   make_block("minecraft:ice"));
                    continue;
                }

                if (chunk.isSameBlock(local_y, local_z, 15 - local_x,
                                      make_block("minecraft:stone"))) {
                    if (!surface_heights[local_x][local_z]) {
                        surface_heights[local_x][local_z] = local_y;
                    }

                    if (air_above && cold) {
                        chunk.setBlock(local_y + 1, local_z, 15 - local_x,
                                       make_block("minecraft:snow"));
                    }

                    if (info.continentalness < 0.5 || water_above) {
                        if (absolute_y < info.height - 5) {
                            break;
                        }

                        chunk.setBlock(local_y, local_z, 15 - local_x,
                                       make_block("minecraft:sand"));

                        if (info.continentalness < 0.4 &&
                            info.temperature > 0.5f && water_above &&
                            info.vegetation > 0.6 &&
                            surface_heights[local_x][local_z] == local_y) {
                            //  Should grow kelp
                            float max_height =
                                std::min<float>((64.f - absolute_y) / 20.f, 1) *
                                15.f;
                            float height_coef = std::max<float>(
                                (info.vegetation - 0.6) * 10.f, 1);
                            float height = height_coef * max_height;
                            int32_t kelp_top = height + local_y;

                            for (int32_t kelp_y = local_y + 1;
                                 kelp_y < kelp_top; kelp_y++) {
                                if (!chunk.isSameBlock(
                                        kelp_y, local_z, 15 - local_x,
                                        make_block("minecraft:water")))
                                    break;

                                if (kelp_y == kelp_top - 1) {
                                    chunk.setBlock(
                                        kelp_y, local_z, 15 - local_x,
                                        make_block("minecraft:kelp"));
                                } else {
                                    chunk.setBlock(
                                        kelp_y, local_z, 15 - local_x,
                                        make_block("minecraft:kelp_plant"));
                                }
                            }
                        }

                    } else if (air_above) {
                        if (cold) {
                            chunk.setBlock(local_y, local_z, 15 - local_x,
                                           make_block<MAKE_KV("snowy", "true")>(
                                               "minecraft:grass_block"));
                            continue;
                        }

                        chunk.setBlock(local_y, local_z, 15 - local_x,
                                       make_block("minecraft:grass_block"));

                        if (info.vegetation > 0.65) {
                            chunk.setBlock(local_y + 1, local_z, 15 - local_x,
                                           make_block("minecraft:tall_grass"));
                            chunk.setBlock(local_y + 2, local_z, 15 - local_x,
                                           make_block<MAKE_KV("half", "upper")>(
                                               "minecraft:tall_grass"));
                        } else if (info.vegetation > 0.6) {
                            chunk.setBlock(local_y + 1, local_z, 15 - local_x,
                                           make_block("minecraft:short_grass"));
                        }
                    } else if (absolute_y > info.height - 5) {
                        chunk.setBlock(local_y, local_z, 15 - local_x,
                                       make_block("minecraft:dirt"));
                    } else {
                        break;
                    }
                }
            }
        }
    }
    for (int32_t local_z = 3; local_z < 13; local_z++) {
        for (int32_t local_x = 3; local_x < 13; local_x++) {
            FlatInfo info = chunk.get_flat_info(local_x, local_z);

            float starting_height = info.height + 15 + 64; // because local

            bool cold = info.temperature < 0.4f;
            int32_t surface_height = surface_heights[local_x][local_z];

            // Generate a tree
            if (surface_height > 130 && info.continentalness > 0.55 &&
                info.continentalness < 0.9 && info.vegetation > 0.675) {

                float tree_size =
                    5 + 2. * std::min<float>((info.vegetation - 0.675) / 0.025f,
                                             1.f);

                // Leaves:
                for (int y = 0; y < 4; y++) {
                    for (int z = -2; z < 3; z++) {
                        for (int x = -2; x < 3; x++) {
                            if (z * z + x * x + y * y + y > tree_size)
                                continue;

                            chunk.setBlock(surface_height + y + 3, local_z + z,
                                           15 - local_x + x,
                                           make_block("minecraft:oak_leaves"));
                        }
                    }
                }

                // Trunk:
                for (int y = 1; y < 4; y++) {
                    chunk.setBlock(surface_height + y, local_z, 15 - local_x,
                                   make_block("minecraft:oak_log"));
                }
            }
        }
    }
};

std::vector<Chunk> ChunkGenerator::generate_all(int32_t region_x,
                                                int32_t region_z) {
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

    thrust::device_vector<VolumetricInfo> volumetrics(16 * 16 * 16 * 24 * 32 *
                                                      32);

    thrust::transform(
        thrust::counting_iterator<int32_t>(0),
        thrust::counting_iterator<int32_t>(16 * 16 * 32 * 32 * 24 * 16),
        volumetrics.begin(),
        [seed = seed, region_x, region_z] __device__(int32_t idx) {
            // 16 x 16 x 16 blocks in subchunk
            // 24 subchunks in chunk
            // 32 x 32 chunks in region

            int32_t local_x = idx % 16;
            int32_t local_z = (idx / 16) % 16;
            int32_t local_y = (idx / 16 / 16) % 16;
            int32_t subchunk_id = (idx / (16 * 16 * 16)) % 24;
            int32_t cell_id = idx / (16 * 16 * 16 * 24);

            int32_t cell_id_x = cell_id / 32;
            int32_t cell_id_z = cell_id % 32;

            int32_t chunk_x = (region_x * 32 + cell_id_x) * 16;
            int32_t chunk_z = (region_z * 32 + cell_id_z) * 16;

            int32_t x = chunk_x + local_x;
            int32_t y = subchunk_id * 16 - 64 + local_y;
            int32_t z = chunk_z + local_z;

            auto density = noise(seed + 4, x, y, z, 0.1f, 2);
            bool cave = cave_noise(seed + 5, x, y, z);

            return VolumetricInfo{density, cave};
        });

    thrust::device_vector<ChunkSmol *> chunk_smols(32 * 32);
    std::vector<Chunk> chunks;
    chunks.reserve(32 * 32);

    for (auto cell_id = 0; cell_id < 32 * 32; cell_id++) {
        int32_t cell_id_x = cell_id / 32;
        int32_t cell_id_z = cell_id % 32;

        int32_t x = (region_x * 32 + cell_id_x) * 16;
        int32_t z = (region_z * 32 + cell_id_z) * 16;

        auto chunk = &chunks.emplace_back(x, z);
        chunk_smols[cell_id] = chunk->chunk_smols.get();
    }

    cudaDeviceSynchronize();

    ChunkSmol **all_chunks = thrust::raw_pointer_cast(chunk_smols.data());
    FlatInfo *all_flats = thrust::raw_pointer_cast(flats.data());
    VolumetricInfo *all_volumetrics =
        thrust::raw_pointer_cast(volumetrics.data());

    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(24 * 32 * 32),
                     [seed = seed, region_x, region_z, all_chunks, all_flats,
                      all_volumetrics] __device__(const uint32_t &smol_idx) {
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
                         generateSmolChunk(
                             raw_chunks + i_ch, seed, chunk_x, i_ch * 16,
                             chunk_z, all_flats + cell_id * 16 * 16,
                             all_volumetrics + smol_idx * 16 * 16 * 16);
                     });

    cudaDeviceSynchronize();

    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(32 * 32),
                     [seed = seed, region_x, region_z, all_chunks,
                      all_flats] __device__(const uint32_t &cell_id) {
                         ChunkSmol *raw_chunks = all_chunks[cell_id];

                         int32_t cell_id_x = cell_id / 32;
                         int32_t cell_id_z = cell_id % 32;

                         int32_t chunk_x = (region_x * 32 + cell_id_x) * 16;
                         int32_t chunk_z = (region_z * 32 + cell_id_z) * 16;

                         ChunkWrapper wrapper(raw_chunks,
                                              all_flats + cell_id * 16 * 16,
                                              chunk_x, chunk_z);
                         replaceSurface(wrapper, seed);
                     });

    return chunks;
}
