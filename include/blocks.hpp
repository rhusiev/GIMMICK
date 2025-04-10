#ifndef BLOCKS_HPP
#define BLOCKS_HPP

#include "./nbt.hpp"
#include <cstdint>

// MUST BE SEQUENTIAL FOR NOW
// (uniqueness computation)
enum class BlockType : uint32_t {
    Air,
    Stone,
    Bedrock,
    Deepslate,
    Grass,
    Water,
    Sand,
    Gravel,
    Shortgrass,
    Kelp,
    KelpPlant,

    Stone_CoalOre,
    Stone_IronOre,
    Stone_CopperOre,
    Stone_GoldOre,
    Stone_LapisOre,
    Stone_RedstoneOre,
    Stone_EmeraldOre,
    Stone_DiamondOre,

    Deepslate_CoalOre,
    Deepslate_IronOre,
    Deepslate_CopperOre,
    Deepslate_GoldOre,
    Deepslate_LapisOre,
    Deepslate_RedstoneOre,
    Deepslate_EmeraldOre,
    Deepslate_DiamondOre,
    
    OakLog,
    OakLeaves,
};
constexpr uint32_t BLOCK_TYPE_COUNT = 29;

void write_block_description(NBTSerializer *serializer, BlockType block_type);

#endif /* end of include guard: BLOCKS_HPP */
