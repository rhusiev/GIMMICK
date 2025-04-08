#ifndef INCLUDE_CHUNK_DOOM_GEN_HPP_
#define INCLUDE_CHUNK_DOOM_GEN_HPP_

#include <cstdint>

enum class BlockType : uint32_t {
    Air,
    Stone,
};

struct Block {
    BlockType block_type;
};

struct ChunkSmol {
    Block blocks[16][16][16];
};

struct Chunk {
    ChunkSmol chunk_smols[16];
};


#endif // INCLUDE_CHUNK_DOOM_GEN_HPP_
