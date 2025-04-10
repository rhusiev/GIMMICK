#ifndef INCLUDE_CHUNK_DOOM_GEN_HPP_
#define INCLUDE_CHUNK_DOOM_GEN_HPP_

#include "../libraries/simplex_noise/SimplexNoise.h"
#include "./blocks.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

/*#define DEBUG_HEIGHTS*/

struct FloatCoord2 {
    float x;
    float z;
};

struct FloatCoord3 {
    float x;
    float y;
    float z;
};
// BlockType enum has been moved to blocks.hpp

class Block {
  public:
    BlockType block_type; // TODO: change to private with methods?
    Block();
    explicit Block(BlockType block_type);
};

class ChunkSmol {
  public:
    Block blocks[16][16][16]; // TODO: change to private with methods?
};

class Chunk {
  public:
    int32_t x;
    int32_t z;
    ChunkSmol chunk_smols[24]; // TODO: change to private with methods?
#ifdef DEBUG_HEIGHTS
    int32_t debug_heights[16][16];
#endif

    Chunk(int32_t x, int32_t z);
};

// New generation system that uses SimplexNoise directly
class ChunkGenerator {
  public:
    Chunk generate(int32_t x, int32_t z);

    // Individual generation stages
    void generateBaseStructure(Chunk &chunk, const float heights[16][16]);
    void generateCaves(Chunk &chunk, const float heights[16][16]);
    void generateSurface(Chunk &chunk, const float heights[16][16]);
    void generateWater(Chunk &chunk, const float heights[16][16]);
    void generateVegetation(Chunk &chunk, const float heights[16][16],
                            const bool grass_flags[16][16],
                            const bool kelp_flags[16][16]);

    // Density/noise functions
    float getBaseTerrainHeight(float x, float z);
    bool shouldGenerateCave(float x, float y, float z,
                            const float heights[16][16]);
    bool shouldGenerateGrass(float x, float z);
    bool shouldGenerateKelp(float x, float z);
};

Chunk generate_chunk(int32_t x, int32_t z);

#endif // INCLUDE_CHUNK_DOOM_GEN_HPP_
