#ifndef BLOCKS_HPP
#define BLOCKS_HPP

#include "./nbt.hpp"
#include <cstdint>

// MUST BE SEQUENTIAL FOR NOW
// (uniqueness computation)
enum class BlockType : uint32_t {
    Air,
    Stone,
    Grass,
};
constexpr uint32_t BLOCK_TYPE_COUNT = 3;

void write_block_description(NBTSerializer *serializer, BlockType block_type);

#endif /* end of include guard: BLOCKS_HPP */
