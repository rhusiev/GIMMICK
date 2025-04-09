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
        if (weight_percent == -1) {
            total *= noise_0_1;
        } else {
            total += noise_0_1 * weight_percent;
        }
    }
    return total / 100; // because weight in percent
}

Chunk::Chunk(int32_t x, int32_t z) : x(x), z(z) {}

Chunk generate_chunk(int32_t x, int32_t z) {
    Chunk chunk{x, z};
    PerlinParam param1{0.001, -1, {0.1, 0.1}};
    PerlinParam param2{0.005, 400, {0.5, 0.5}};
    PerlinParam param3{0.04, 50, {0.7, 0.7}};
    PerlinParam param4{0.001, -1, {0.1, 0.1}};
    PerlinParam param5{0.001, -50, {0.1, 0.1}};
    PerlinParam param6{0.0005, -15, {0.6, 0.3}};
    Generator gen{{param3, param1, param2, param4, param5, param6}};

    PerlinParam grass_param1{0.1, 50, {0.1, 0.1}};
    PerlinParam grass_param2{1.0, 50, {0.1, 0.1}};
    Generator grass_gen{{grass_param1, grass_param2}};

    for (int32_t i_x = 0; i_x < 16; i_x++) {
        for (int32_t i_z = 0; i_z < 16; i_z++) {
            float height =
                gen.generate_octave(chunk.x + i_x, chunk.z + i_z) * 32 + 64;
#ifdef DEBUG_HEIGHTS
            chunk.debug_heights[i_x][i_z] = height;

#endif
            for (int32_t i_ch = 0; i_ch < 24; i_ch++) {
                ChunkSmol &chunk_smol = chunk.chunk_smols[i_ch];
                bool generate_grass = grass_gen.generate_octave(
                                          chunk.x + i_x, chunk.z + i_z) > 0.8;

                for (int32_t i_y = 0; i_y < 16; i_y++) {
                    if (height > i_y + i_ch * 16 + 1) {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Stone;
                    } else if (height > i_y + i_ch * 16) {
                        if (height > 65) {
                            chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                                BlockType::Grass;
                        } else {
                            chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                                BlockType::Sand;
                        }
                    } else if (height > i_y + i_ch * 16 - 1 && height > 65 &&
                               generate_grass) {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Shortgrass;
                    } else if (i_y + i_ch * 16 < 65) {
                        if (generate_grass) {
                            int kelp_boundary = 64;
                            if (height + 5 < kelp_boundary) {
                                kelp_boundary = height + 5;
                            }

                            // Check if we're above the boundary
                            if (i_y + i_ch * 16 > kelp_boundary) {
                                chunk_smol.blocks[i_y][i_z][15 - i_x]
                                    .block_type = BlockType::Water;
                                continue;
                            }

                            // Check if we're at the boundary
                            if (i_y + i_ch * 16 > kelp_boundary - 1) {
                                chunk_smol.blocks[i_y][i_z][15 - i_x]
                                    .block_type = BlockType::Kelp;
                            } else {
                                chunk_smol.blocks[i_y][i_z][15 - i_x]
                                    .block_type = BlockType::KelpPlant;
                            }
                        } else {
                            chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                                BlockType::Water;
                        }
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
