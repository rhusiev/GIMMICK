#include "./chunk_dOOm_gen.hpp"
#include "./SimplexNoise.h"
#include <algorithm>
#include <cstdlib>

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

void ChunkSmol::serializeBlockStates(NBTSerializer* serializer) const {
    // Serialize the palette directly from registry
    registry.serializePalette(serializer);
    
    // Get bit count required for encoding and encode the block data
    encodeBlockData(serializer);
}

void ChunkSmol::encodeBlockData(NBTSerializer* serializer) const {
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
        
        for (uint32_t j = 0; j < blocks_per_long && (first_block + j) < 4096; j++) {
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

    // Pre-calculate heights for the entire chunk
    float heights[16][16];
    for (int32_t i_x = 0; i_x < 16; i_x++) {
        for (int32_t i_z = 0; i_z < 16; i_z++) {
            float height = getBaseTerrainHeight(chunk.x + i_x, chunk.z + i_z);
            heights[i_x][i_z] = height;
#ifdef DEBUG_HEIGHTS
            chunk.debug_heights[i_x][i_z] = height;
#endif
        }
    }

    // Apply only basic stone generation
    generateBaseStructure(chunk, heights);

    return chunk;
}

void ChunkGenerator::generateBaseStructure(Chunk &chunk,
                                           const float heights[16][16]) {
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
                    float height = heights[i_x][i_z];

                    if (height > absolute_y) {
                        chunk_smol.setBlock(i_y, i_z, 15 - i_x, make_block("minecraft:stone"));
                    } else {
                        chunk_smol.setBlock(i_y, i_z, 15 - i_x, make_block("minecraft:air"));
                    }
                }
            }
        }
    }
}

float ChunkGenerator::getBaseTerrainHeight(float x, float z) {
    auto noise = SimplexNoise(0.01f).fractal(3, x, z);
    
    // Simple height calculation: noise range -1 to 1, convert to 32-160 range
    auto height = (noise + 1) * 64 + 32; // 32 - 160
    
    return std::clamp<double>(height, 0.0, 320.0);
}

// Legacy function that uses the new ChunkGenerator
Chunk generate_chunk(int32_t x, int32_t z) {
    static ChunkGenerator generator;
    return generator.generate(x, z);
}
