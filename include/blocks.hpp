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
    Water,
    Sand,
    Gravel,
    Shortgrass,
    Kelp,
    KelpPlant,
};
constexpr uint32_t BLOCK_TYPE_COUNT = 9;

void write_block_description(NBTSerializer *serializer, BlockType block_type);

#endif /* end of include guard: BLOCKS_HPP */
