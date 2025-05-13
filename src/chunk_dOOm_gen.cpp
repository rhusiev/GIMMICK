#include "./chunk_dOOm_gen.hpp"
#include "./SimplexNoise.h"
#include <algorithm>
#include <cstdlib>

Block::Block() : block_type(BlockType::Air) {}

Block::Block(BlockType type) : block_type(type) {}

BlockRegistry::BlockRegistry() {
    blocks[0] = Block(BlockType::Air);
    block_count = 1;
}

uint8_t BlockRegistry::addBlock(const Block& block) {
    for (size_t i = 0; i < block_count; i++) {
        if (blocks[i].block_type == block.block_type) {
            return static_cast<uint8_t>(i);
        }
    }
    
    if (block_count < MAX_BLOCKS) {
        blocks[block_count] = block;
        return static_cast<uint8_t>(block_count++);
    }
    
    return 0;
}

const Block& BlockRegistry::getBlock(uint8_t id) const {
    if (id < block_count) {
        return blocks[id];
    }
    return blocks[0];
}

size_t BlockRegistry::getBlockCount() const {
    return block_count;
}

ChunkSmol::ChunkSmol() {
    for (int32_t y = 0; y < 16; y++) {
        for (int32_t z = 0; z < 16; z++) {
            for (int32_t x = 0; x < 16; x++) {
                block_ids[y][z][x] = 0;
            }
        }
    }
}

void ChunkSmol::setBlock(int32_t y, int32_t z, int32_t x, const Block& block) {
    if (y >= 0 && y < 16 && z >= 0 && z < 16 && x >= 0 && x < 16) {
        uint8_t id = registry.addBlock(block);
        block_ids[y][z][x] = id;
    }
}

const BlockRegistry& ChunkSmol::getRegistry() const {
    return registry;
}

uint8_t ChunkSmol::getBlockId(int32_t y, int32_t z, int32_t x) const {
    return block_ids[y][z][x];
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
                        chunk_smol.setBlock(i_y, i_z, 15 - i_x, Block(BlockType::Stone));
                    } else {
                        chunk_smol.setBlock(i_y, i_z, 15 - i_x, Block(BlockType::Air));
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
