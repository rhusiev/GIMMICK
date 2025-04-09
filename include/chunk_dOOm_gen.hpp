#ifndef INCLUDE_CHUNK_DOOM_GEN_HPP_
#define INCLUDE_CHUNK_DOOM_GEN_HPP_

#include "./blocks.hpp"
#include <cstdint>
#include <vector>
#include <cmath>

#define DEBUG_HEIGHTS

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

struct PerlinParam {
    float scale;
    float weight_percent; // because better float ig
    FloatCoord2 offset;
};

struct PerlinParam3D {
    float scale;
    float weight_percent; // because better float ig
    FloatCoord3 offset;
};
class Generator {
  public: // TODO: change to private with methods?
    std::vector<PerlinParam> perlin_scales_offsets;

    float generate_octave(float x, float z);
};

class Generator3D {
  public: 
    std::vector<std::vector<PerlinParam3D>> perlin_scales_offsets;
    float generate_octave(float x, float y, float z,int n);
    bool generate_cave(float x, float y, float z);
    bool compare(float a, float b, float epsilon) {
        return (std::fabs(a - b) < epsilon);
    };
};


Chunk generate_chunk(int32_t x, int32_t z);

#endif // INCLUDE_CHUNK_DOOM_GEN_HPP_
