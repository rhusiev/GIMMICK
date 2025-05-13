#ifndef INCLUDE_CHUNK_DOOM_GEN_HPP_
#define INCLUDE_CHUNK_DOOM_GEN_HPP_

#include "../libraries/simplex_noise/SimplexNoise.h"
#include "./blocks.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>

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

class BlockRegistry {
  private:
    static constexpr size_t MAX_BLOCKS = 256;
    Block blocks[MAX_BLOCKS];
    size_t block_count = 0;
    
  public:
    BlockRegistry();
    
    uint8_t addBlock(const Block& block);
    
    size_t getBlockCount() const;
    const Block& getBlock(uint8_t id) const;
};

class ChunkSmol {
  private:
    uint8_t block_ids[16][16][16];
    BlockRegistry registry;
    
  public:
    ChunkSmol();
    
    void setBlock(int32_t y, int32_t z, int32_t x, const Block& block);

    const BlockRegistry& getRegistry() const;
    uint8_t getBlockId(int32_t y, int32_t z, int32_t x) const;
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

// Simplified generation system that only creates stone terrain
class ChunkGenerator {
  public:
    Chunk generate(int32_t x, int32_t z);

    // Individual generation stages (now only using base structure)
    void generateBaseStructure(Chunk &chunk, const float heights[16][16]);
    
    // Density/noise function
    float getBaseTerrainHeight(float x, float z);
};

Chunk generate_chunk(int32_t x, int32_t z);

#endif // INCLUDE_CHUNK_DOOM_GEN_HPP_
