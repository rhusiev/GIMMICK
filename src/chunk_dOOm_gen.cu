#include "./chunk_dOOm_gen.hpp"
#include "./cuda_noise.cuh"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

ChunkSmol::ChunkSmol() {
    for (int32_t y = 0; y < 16; y++) {
        for (int32_t z = 0; z < 16; z++) {
            for (int32_t x = 0; x < 16; x++) {
                block_ids[y][z][x] = 0;
            }
        }
    }
}

uint8_t ChunkSmol::getBlockId(int32_t y, int32_t z, int32_t x) const {
    return block_ids[y][z][x];
}

void ChunkSmol::serializeBlockStates(NBTSerializer *serializer) const {
    // Serialize the palette directly from registry
    registry.serializePalette(serializer);

    // Get bit count required for encoding and encode the block data
    encodeBlockData(serializer);
}

void ChunkSmol::encodeBlockData(NBTSerializer *serializer) const {
    // First determine the number of bits needed per block
    uint32_t n_bits = 4; // Minimum 4 bits as per Minecraft's requirements

    // Skip data section entirely if palette is empty or only has air
    if (n_bits == 0) {
        return;
    }

    uint32_t blocks_per_long = 64 / n_bits;
    uint32_t longs_needed = 4096 / blocks_per_long;

    serializer->writeTagHeader("data", NBT_TagType::TAG_Long_Array);
    serializer->writeInt(longs_needed);

    for (uint32_t i = 0; i < longs_needed; i++) {
        uint32_t first_block = i * blocks_per_long;
        uint64_t packed_data = 0;

        for (uint32_t j = 0; j < blocks_per_long && (first_block + j) < 4096;
             j++) {
            uint32_t block_index = first_block + j;
            uint32_t y = block_index / 256;
            uint32_t z = (block_index % 256) / 16;
            uint32_t x = block_index % 16;

            uint64_t block_id = getBlockId(y, z, x);

            uint32_t shift = 64 - ((j + 1) * n_bits);
            packed_data |= (block_id << shift);
        }

        serializer->writeLong(packed_data);
    }
}

Chunk::Chunk(int32_t x, int32_t z) : x(x), z(z) {}

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

void ChunkGenerator::generateBaseStructure(Chunk &chunk,
                                           const thrust::device_vector<float> &d_heights) {
    // Copy heights to host for further processing
    thrust::host_vector<float> h_heights = d_heights;

    // Iterate through each sub-chunk (chunk_smol)
    for (int32_t i_ch = 0; i_ch < 24; i_ch++) {
        ChunkSmol &chunk_smol = chunk.chunk_smols[i_ch];

        // For each y level in the sub-chunk (best memory locality)
        for (int32_t i_y = 0; i_y < 16; i_y++) {
            int32_t absolute_y = i_y + i_ch * 16;

            // Then z coordinate
            for (int32_t i_z = 0; i_z < 16; i_z++) {
                // Then x coordinate
                for (int32_t i_x = 0; i_x < 16; i_x++) {
                    float height = h_heights[i_z * 16 + i_x];

                    if (height > absolute_y) {
                        static constexpr char key_snowy[] = "snowy";
                        static constexpr char val_true[] = "true";
                        chunk_smol.setBlock(i_y, i_z, 15 - i_x,
                                            make_block<make_kv(key_snowy, val_true)>("minecraft:grass_block"));
                    } else {
                        chunk_smol.setBlock(i_y, i_z, 15 - i_x,
                                            make_block("minecraft:air"));
                    }
                }
            }
        }
    }
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
