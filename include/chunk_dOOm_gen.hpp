#ifndef INCLUDE_CHUNK_DOOM_GEN_HPP_
#define INCLUDE_CHUNK_DOOM_GEN_HPP_

#include <cstdint>
#include <tuple>
#include <vector>

#define DEBUG_HEIGHTS

struct FloatCoord2 {
    float x;
    float z;
};

enum class BlockType : uint32_t {
    Air,
    Stone,
};

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
    ChunkSmol chunk_smols[16]; // TODO: change to private with methods?
#ifdef DEBUG_HEIGHTS
    int32_t debug_heights[16][16];
#endif

    Chunk(int32_t x, int32_t z);
};

struct PerlinParam {
    float scale;
    float weight_percent; // because better float ig
    FloatCoord2 offset;
};

class Generator {
  public: // TODO: change to private with methods?
    std::vector<PerlinParam> perlin_scales_offsets;

    float generate_octave(float x, float z);
};

Chunk generate_chunk(int32_t x, int32_t z);

#endif // INCLUDE_CHUNK_DOOM_GEN_HPP_
