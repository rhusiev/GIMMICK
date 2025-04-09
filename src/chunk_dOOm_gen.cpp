#include "./chunk_dOOm_gen.hpp"
#include "./SimplexNoise.h"

Block::Block() : block_type(BlockType::Air) {}

Block::Block(BlockType type) : block_type(type) {}

float Generator::generate_octave(float x, float z) {
    float total = 0;
    for (auto &[scale, weight_percent, offset] : perlin_scales_offsets) {
        auto noise_0_1 =
            (SimplexNoise::noise(x * scale + offset.x, z * scale + offset.z) +
             1) /
            2;
        total += noise_0_1 * weight_percent;
    }
    return total / 100; // because weight in percent
}

Chunk::Chunk(int32_t x, int32_t z) : x(x), z(z) {}

Chunk generate_chunk(int32_t x, int32_t z) {
    Chunk chunk{x, z};
    PerlinParam param1{0.0005, 50, {0.1, 0.1}};
    PerlinParam param2{0.0004, 40, {0.5, 0.5}};
    PerlinParam param3{0.001, 10, {0.7, 0.7}};
    Generator gen{{param1, param2, param3}};
    for (size_t i_x = 0; i_x < 16; i_x++) {
        for (size_t i_z = 0; i_z < 16; i_z++) {
            float height =
                gen.generate_octave(chunk.x + i_x, chunk.z + i_z) * 384 - 128;
#ifdef DEBUG_HEIGHTS
            chunk.debug_heights[i_x][i_z] = height;

#endif
            for (size_t i_ch = 0; i_ch < 24; i_ch++) {
                ChunkSmol &chunk_smol = chunk.chunk_smols[i_ch];
                for (size_t i_y = 0; i_y < 16; i_y++) {
                    if (height > i_y + i_ch * 16 + 1) {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Stone;
                    } else if (height > i_y + i_ch * 16) {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Grass;
                    } else {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Air;
                    }
                }
            }
        }
    }
    return chunk;
}
