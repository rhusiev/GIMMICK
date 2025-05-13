#ifndef INCLUDE_CHUNK_DOOM_GEN_HPP_
#define INCLUDE_CHUNK_DOOM_GEN_HPP_

#include "./SimplexNoise.hpp"
#include "./block_template.hpp"
#include "./block_registry.hpp"
#include "./nbt.hpp"
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

class ChunkSmol {
  private:
    uint8_t block_ids[16][16][16];
    BlockRegistry registry;
    
  public:
    ChunkSmol();
    
    template<typename Block>
    void setBlock(int32_t y, int32_t z, int32_t x, const Block& block) {
        if (y >= 0 && y < 16 && z >= 0 && z < 16 && x >= 0 && x < 16) {
            uint8_t id = registry.addBlock(block);
            block_ids[y][z][x] = id;
        }
    }

    uint8_t getBlockId(int32_t y, int32_t z, int32_t x) const;
    
    // Add new methods for palette encoding
    void serializeBlockStates(NBTSerializer* serializer) const;
    void encodeBlockData(NBTSerializer* serializer) const;
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
