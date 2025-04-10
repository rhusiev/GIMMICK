#include "./chunk_dOOm_gen.hpp"
#include "./SimplexNoise.h"
#include <algorithm>
#include <cstdlib>

Block::Block() : block_type(BlockType::Air) {}

Block::Block(BlockType type) : block_type(type) {}

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

    // Pre-calculate vegetation flags to avoid redundant noise calculations
    bool grass_flags[16][16];
    bool kelp_flags[16][16];
    for (int32_t i_x = 0; i_x < 16; i_x++) {
        for (int32_t i_z = 0; i_z < 16; i_z++) {
            grass_flags[i_x][i_z] =
                shouldGenerateGrass(chunk.x + i_x, chunk.z + i_z);
            kelp_flags[i_x][i_z] =
                shouldGenerateKelp(chunk.x + i_x, chunk.z + i_z);
        }
    }

    // Apply generation stages in sequence
    generateBaseStructure(chunk, heights);
    generateCaves(chunk, heights);
    generateSurface(chunk, heights);
    generateWater(chunk, heights);
    generateTrees(chunk, heights);
    generateVegetation(chunk, heights, grass_flags, kelp_flags);

    return chunk;
}

void ChunkGenerator::generateBaseStructure(Chunk &chunk,
                                           const float heights[16][16]) {
    auto tiny = SimplexNoise(1.0f);

    float bedrock_heights[16][16];
    float deepslate_heights[16][16];

    for (int32_t x = 0; x < 16; x++) {
        for (int32_t z = 0; z < 16; z++) {
            auto noise = tiny.fractal(1, chunk.x + x, chunk.z + z);
            bedrock_heights[x][z] = 2.0f + noise;
            deepslate_heights[x][z] = 32.0f + noise * 2.0f;
        }
    }

    auto ore_noise = SimplexNoise(0.1f);

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

                    float depth = (height - absolute_y);
                    float depth_probability_scaling =
                        std::min(depth * 0.125f, 1.f);

                    float x = chunk.x + i_x;
                    float z = chunk.z + i_z;

                    auto is_coal =
                        ore_noise.fractal(2, x, absolute_y, z) * 0.5 + 0.5 <
                        0.11 * depth_probability_scaling;
                    auto is_iron =
                        ore_noise.fractal(2, x, absolute_y + 320, z) * 0.5 +
                            0.5 <
                        0.11 * depth_probability_scaling;
                    auto is_copper =
                        ore_noise.fractal(2, x, absolute_y + 320 * 2, z) * 0.5 +
                            0.5 <
                        0.05 * depth_probability_scaling;
                    auto is_gold =
                        ore_noise.fractal(2, x, absolute_y + 320 * 3, z) * 0.5 +
                            0.5 <
                        0.1 * depth_probability_scaling;
                    auto is_lapis =
                        ore_noise.fractal(2, x, absolute_y + 320 * 4, z) * 0.5 +
                            0.5 <
                        0.025 * depth_probability_scaling;
                    auto is_redstone =
                        ore_noise.fractal(2, x, absolute_y + 320 * 5, z) * 0.5 +
                            0.5 <
                        0.05 * depth_probability_scaling;
                    auto is_emerald =
                        ore_noise.fractal(2, x, absolute_y + 320 * 6, z) * 0.5 +
                            0.5 <
                        0.025 * depth_probability_scaling;
                    auto is_diamond =
                        ore_noise.fractal(2, x, absolute_y + 320 * 7, z) * 0.5 +
                            0.5 <
                        0.025 * depth_probability_scaling;

                    if (absolute_y < bedrock_heights[i_x][i_z]) {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Bedrock;
                    } else if (absolute_y < deepslate_heights[i_x][i_z]) {
                        auto block = BlockType::Deepslate;

                        if (is_coal) {
                            block = BlockType::Deepslate_CoalOre;
                        } else if (is_iron) {
                            block = BlockType::Deepslate_IronOre;
                        } else if (is_copper) {
                            block = BlockType::Deepslate_CopperOre;
                        } else if (is_gold) {
                            block = BlockType::Deepslate_GoldOre;
                        } else if (is_lapis) {
                            block = BlockType::Deepslate_LapisOre;
                        } else if (is_redstone) {
                            block = BlockType::Deepslate_RedstoneOre;
                        } else if (is_emerald) {
                            block = BlockType::Deepslate_EmeraldOre;
                        } else if (is_diamond) {
                            block = BlockType::Deepslate_DiamondOre;
                        }

                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            block;
                    } else if (height > absolute_y + 1) {
                        auto block = BlockType::Stone;

                        if (is_coal) {
                            block = BlockType::Stone_CoalOre;
                        } else if (is_iron) {
                            block = BlockType::Stone_IronOre;
                        } else if (is_copper) {
                            block = BlockType::Stone_CopperOre;
                        } else if (is_gold) {
                            block = BlockType::Stone_GoldOre;
                        } else if (is_lapis) {
                            block = BlockType::Stone_LapisOre;
                        } else if (is_redstone) {
                            block = BlockType::Stone_RedstoneOre;
                        } else if (is_emerald) {
                            block = BlockType::Stone_EmeraldOre;
                        } else if (is_diamond) {
                            block = BlockType::Stone_DiamondOre;
                        }

                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            block;
                    } else {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Air;
                    }
                }
            }
        }
    }
}

void ChunkGenerator::generateCaves(Chunk &chunk, const float heights[16][16]) {
    // Iterate through each sub-chunk
    for (int32_t i_ch = 0; i_ch < 24; i_ch++) {
        ChunkSmol &chunk_smol = chunk.chunk_smols[i_ch];

        // For each y level
        for (int32_t i_y = 0; i_y < 16; i_y++) {
            int32_t absolute_y = i_y + i_ch * 16;

            // Then z coordinate
            for (int32_t i_z = 0; i_z < 16; i_z++) {
                // Then x coordinate
                for (int32_t i_x = 0; i_x < 16; i_x++) {
                    if (shouldGenerateCave(chunk.x + i_x, absolute_y,
                                           chunk.z + i_z, heights)) {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Air;
                    }
                }
            }
        }
    }
}

void ChunkGenerator::generateSurface(Chunk &chunk,
                                     const float heights[16][16]) {
    const float MAYBE_SAND_LEVEL = 129.5;
    const float SAND_LEVEL = 129;
    const float GRAVEL_LEVEL = 124;

    auto sand = SimplexNoise(0.01f);

    for (int32_t i_z = 0; i_z < 16; i_z++) {
        for (int32_t i_x = 0; i_x < 16; i_x++) {
            float height = heights[i_x][i_z] - 1;

            int32_t i_ch = static_cast<int32_t>(height) / 16;
            ChunkSmol &chunk_smol = chunk.chunk_smols[i_ch];
            int32_t relative_y = static_cast<int32_t>(height) % 16;

            if (chunk_smol.blocks[relative_y][i_z][15 - i_x].block_type ==
                BlockType::Stone) {
                auto noise = sand.fractal(2, chunk.x + i_x, chunk.z + i_z);

                if (height < GRAVEL_LEVEL) {
                    chunk_smol.blocks[relative_y][i_z][15 - i_x].block_type =
                        noise < 0.0 ? BlockType::Sand : BlockType::Gravel;
                } else if (height < SAND_LEVEL) {
                    chunk_smol.blocks[relative_y][i_z][15 - i_x].block_type =
                        BlockType::Sand;
                } else if (height < MAYBE_SAND_LEVEL && noise < 0.5) {
                    chunk_smol.blocks[relative_y][i_z][15 - i_x].block_type =
                        BlockType::Sand;
                } else {
                    chunk_smol.blocks[relative_y][i_z][15 - i_x].block_type =
                        BlockType::Grass;
                }
            }
        }
    }
}

void ChunkGenerator::generateWater(Chunk &chunk, const float heights[16][16]) {
    const int32_t WATER_LEVEL = 129;

    // Iterate through each sub-chunk
    for (int32_t i_ch = 0; i_ch < 24; i_ch++) {
        if (i_ch * 16 > WATER_LEVEL) {
            continue; // Skip sub-chunks above water level
        }

        ChunkSmol &chunk_smol = chunk.chunk_smols[i_ch];

        // For each y level
        for (int32_t i_y = 0; i_y < 16; i_y++) {
            int32_t absolute_y = i_y + i_ch * 16;

            // Skip if this entire y-layer is above water level
            if (absolute_y > WATER_LEVEL) {
                continue;
            }

            // Then z coordinate
            for (int32_t i_z = 0; i_z < 16; i_z++) {
                // Then x coordinate
                for (int32_t i_x = 0; i_x < 16; i_x++) {
                    float height = heights[i_x][i_z] - 1;

                    if (height < absolute_y &&
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type ==
                            BlockType::Air) {
                        chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                            BlockType::Water;
                    }
                }
            }
        }
    }
}

void ChunkGenerator::generateVegetation(Chunk &chunk,
                                        const float heights[16][16],
                                        const bool grass_flags[16][16],
                                        const bool kelp_flags[16][16]) {
    // Iterate through each sub-chunk
    for (int32_t i_ch = 0; i_ch < 24; i_ch++) {
        ChunkSmol &chunk_smol = chunk.chunk_smols[i_ch];

        // For each y level
        for (int32_t i_y = 0; i_y < 16; i_y++) {
            int32_t absolute_y = i_y + i_ch * 16;

            // Then z coordinate
            for (int32_t i_z = 0; i_z < 16; i_z++) {
                // Then x coordinate
                for (int32_t i_x = 0; i_x < 16; i_x++) {
                    float height = heights[i_x][i_z];
                    bool generate_grass = grass_flags[i_x][i_z];
                    bool generate_kelp = kelp_flags[i_x][i_z];

                    // Generate shortgrass on grass blocks
                    if (chunk_smol.blocks[i_y][i_z][15 - i_x].block_type ==
                            BlockType::Grass &&
                        absolute_y < height && generate_grass) {

                        // Place shortgrass one block above grass
                        if (i_y + 1 < 16) {
                            if (chunk_smol.blocks[i_y + 1][i_z][15 - i_x]
                                    .block_type != BlockType::Air) {
                                continue; // Skip if not air
                            }
                            chunk_smol.blocks[i_y + 1][i_z][15 - i_x]
                                .block_type = BlockType::Shortgrass;
                        } else if (i_ch + 1 < 24) {
                            if (chunk.chunk_smols[i_ch + 1]
                                    .blocks[0][i_z][15 - i_x]
                                    .block_type != BlockType::Air) {
                                continue; // Skip if not air
                            }
                            chunk.chunk_smols[i_ch + 1]
                                .blocks[0][i_z][15 - i_x]
                                .block_type = BlockType::Shortgrass;
                        }
                    }

                    // Generate kelp in water areas
                    if (chunk_smol.blocks[i_y][i_z][15 - i_x].block_type ==
                            BlockType::Water &&
                        absolute_y < 129 && generate_kelp) {

                        int kelp_boundary =
                            std::min(128, static_cast<int>(height + 5));

                        if (absolute_y > kelp_boundary - 1 &&
                            absolute_y <= kelp_boundary) {
                            chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                                BlockType::Kelp;
                        } else if (absolute_y <= kelp_boundary - 1) {
                            chunk_smol.blocks[i_y][i_z][15 - i_x].block_type =
                                BlockType::KelpPlant;
                        }
                    }
                }
            }
        }
    }
}

void ChunkGenerator::generateTrees(Chunk &chunk, const float heights[16][16]) {
    // Check forestness of this chunk
    float forestness = 0.0f;

    // Use a slightly higher frequency than biome blending (0.0005)
    auto forest_noise = SimplexNoise(0.0006f);

    // Sample at chunk center for better continuity
    float center_x = chunk.x + 8.0f;
    float center_z = chunk.z + 8.0f;

    // Calculate forestness value between 0 and 1
    forestness = forest_noise.fractal(3, center_x, center_z) * 0.5f + 0.5f;

    // Determine number of trees to generate (0-4 based on forestness)
    int max_trees = 4;
    int num_trees = static_cast<int>(forestness * (max_trees + 1));

    // Generate each tree
    for (int i = 0; i < num_trees; i++) {
        // Generate random position in chunk (away from borders)
        int32_t tree_x = 3 + (rand() % 10); // 3-12
        int32_t tree_z = 3 + (rand() % 10); // 3-12

        // Get height at this position
        int32_t height = static_cast<int32_t>(heights[tree_x][tree_z]);

        // Check if we can place a tree here
        if (canPlaceTree(chunk, tree_x, tree_z, height, heights)) {
            placeTree(chunk, tree_x, tree_z, height);
        }
    }
}

bool ChunkGenerator::canPlaceTree(Chunk &chunk, int32_t x, int32_t z,
                                  int32_t height, const float heights[16][16]) {
    // Check if there's grass below
    int32_t ground_y = height - 1;
    int32_t i_ch = ground_y / 16;
    int32_t relative_y = ground_y % 16;

    // Skip if out of bounds
    if (i_ch < 0 || i_ch >= 24)
        return false;

    // Check if there's grass block below
    if (chunk.chunk_smols[i_ch].blocks[relative_y][z][15 - x].block_type !=
        BlockType::Grass) {
        return false;
    }

    // Check if there's enough space above (6 blocks for a small tree)
    for (int y_offset = 0; y_offset < 6; y_offset++) {
        int32_t check_y = height + y_offset;

        // Check a 3x3 area around the tree trunk for each level
        for (int check_z = z - 1; check_z <= z + 1; check_z++) {
            for (int check_x = x - 1; check_x <= x + 1; check_x++) {
                // Skip if coordinates are outside chunk
                if (check_x < 0 || check_x >= 16 || check_z < 0 ||
                    check_z >= 16) {
                    continue;
                }

                // Calculate the proper subchunk and relative coordinates
                int32_t check_ch = check_y / 16;
                int32_t rel_y = check_y % 16;

                // Make sure we don't exceed chunk boundary
                if (check_ch < 0 || check_ch >= 24) {
                    return false;
                }

                // Make sure the space is air
                if (chunk.chunk_smols[check_ch]
                        .blocks[rel_y][check_z][15 - check_x]
                        .block_type != BlockType::Air) {
                    return false;
                }
            }
        }
    }

    return true;
}

void ChunkGenerator::placeTree(Chunk &chunk, int32_t x, int32_t z,
                               int32_t height) {
    // Tree structure - a simple oak tree with a 1-block trunk and leaves around

    // Height of the tree (4-6 blocks)
    int tree_height = 4 + (rand() % 3);

    // Place trunk
    for (int y_offset = 0; y_offset < tree_height; y_offset++) {
        int32_t y = height + y_offset;
        // Place log at the trunk position using the helper method
        setBlockInChunk(chunk, x, y, z, BlockType::OakLog);
    }

    // Place leaves (in a 5x5x3 pattern around the top of the tree)
    for (int y_offset = tree_height - 3; y_offset < tree_height + 1;
         y_offset++) {
        int32_t y = height + y_offset;

        // Size of the leaf layer depends on the height
        int leaf_radius = (y_offset == tree_height)       ? 0
                          : (y_offset == tree_height - 1) ? 1
                                                          : 2;

        for (int leaf_z = z - leaf_radius; leaf_z <= z + leaf_radius;
             leaf_z++) {
            for (int leaf_x = x - leaf_radius; leaf_x <= x + leaf_radius;
                 leaf_x++) {
                // Skip if coordinates are outside chunk
                if (leaf_x < 0 || leaf_x >= 16 || leaf_z < 0 || leaf_z >= 16) {
                    continue;
                }

                // For the outer corners, add some randomness
                if (leaf_radius == 2 &&
                    (std::abs(leaf_x - x) == 2 && std::abs(leaf_z - z) == 2) &&
                    (rand() % 2 == 0)) {
                    continue;
                }

                // Don't place leaves at the trunk
                if (leaf_x == x && leaf_z == z && y_offset < tree_height) {
                    continue;
                }

                // Place leaves - safely handling subchunk boundaries
                setBlockInChunk(chunk, leaf_x, y, leaf_z, BlockType::OakLeaves);
            }
        }
    }
}

// Helper method to safely set blocks across subchunk boundaries
void ChunkGenerator::setBlockInChunk(Chunk &chunk, int32_t x, int32_t y,
                                     int32_t z, BlockType block_type) {
    // Ensure we're within the world height bounds
    if (y < 0 || y >= 24 * 16)
        return;

    // Calculate subchunk and relative coordinates
    int32_t i_ch = y / 16;
    int32_t relative_y = y % 16;

    // Skip if out of valid subchunk range
    if (i_ch < 0 || i_ch >= 24)
        return;

    // Skip if outside chunk boundaries in x/z
    if (x < 0 || x >= 16 || z < 0 || z >= 16)
        return;

    // Set the block
    chunk.chunk_smols[i_ch].blocks[relative_y][z][15 - x].block_type =
        block_type;
}

bool ChunkGenerator::shouldGenerateForest(float x, float z) {
    // Use noise at slightly higher frequency than biome blending
    auto forest = SimplexNoise(0.0006f).fractal(3, x, z) * 0.5f + 0.5f;

    // Return true if the noise is above a threshold
    return forest > 0.7f;
}

float ChunkGenerator::getBaseTerrainHeight(float x, float z) {
    auto mountains = SimplexNoise(0.01f).fractal(3, x, z);
    auto planes = SimplexNoise(0.001f).fractal(3, x, z);
    auto biome_blending = SimplexNoise(0.0005).fractal(3, x, z) * 0.5 + 0.5;

    auto mountains_height = (mountains + 1) * 32 + 64; // 64 - 128
    auto planes_height = (planes + 1) * 16 + 48;       // 48 - 80

    auto ocean_mask = SimplexNoise(0.00025).fractal(1, x, z) * 0.5 + 0.5;
    auto ocean_height = SimplexNoise(0.001).fractal(2, x, z) * 16 + 48;

    auto surface_height = biome_blending * planes_height +
                          (1 - biome_blending) * mountains_height;

    auto height = surface_height * (1 - ocean_mask) + ocean_height * ocean_mask;

    return std::clamp(height + 64.0, 0.0, 320.0);
}

bool ChunkGenerator::shouldGenerateCave(float x, float y, float z,
                                        const float heights[16][16]) {
    float distance_to_surface =
        heights[static_cast<uint32_t>(x) % 16][static_cast<uint32_t>(z) % 16] -
        y;

    if (distance_to_surface < 0) {
        return false;
    }

    float noodle1 = SimplexNoise(0.01).fractal(2, x, y, z);
    float noodle2 = SimplexNoise(0.02).fractal(2, x, y, z);

    float cavern = SimplexNoise(0.01).fractal(3, x, y, z);

    float noodle1_probability =
        std::max(1.f - std::abs(noodle1 - 0.2f) * 10.0f, 0.f);
    float noodle2_probability =
        std::max(1.f - std::abs(noodle2 - 0.2f) * 10.0f, 0.f);

    float cavern_probability = std::max(0.f, cavern - 0.6f);

    float height_based_probability =
        std::min(distance_to_surface * 0.0625f + 0.125f, 1.f) *
        std::max(std::min((y - 4.f) * 0.125f, 1.f), 0.f);

    return (noodle1_probability * noodle2_probability + cavern_probability) *
               height_based_probability >
           0.1f;
}

bool ChunkGenerator::shouldGenerateGrass(float x, float z) {
    return SimplexNoise(0.1).fractal(2, x, z) > 0.6f;
}

bool ChunkGenerator::shouldGenerateKelp(float x, float z) {
    return SimplexNoise(0.1).fractal(2, x, z) > 0.5f;
}

// Legacy function that uses the new ChunkGenerator
Chunk generate_chunk(int32_t x, int32_t z) {
    static ChunkGenerator generator;
    return generator.generate(x, z);
}
