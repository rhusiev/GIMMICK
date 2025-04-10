#include "./blocks.hpp"
#include "./nbt.hpp"

void write_block_description(NBTSerializer *serializer, BlockType block_type) {
    serializer->writeTagHeader("Name", NBT_TagType::TAG_String);
    switch (block_type) {
    case BlockType::Air:
        serializer->writeString("minecraft:air");
        break;
    case BlockType::Stone:
        serializer->writeString("minecraft:stone");
        break;
    case BlockType::Bedrock:
        serializer->writeString("minecraft:bedrock");
        break;
    case BlockType::Deepslate:
        serializer->writeString("minecraft:deepslate");
        break;
    case BlockType::Grass:
        serializer->writeString("minecraft:grass_block");
        serializer->writeTagHeader("Properties", NBT_TagType::TAG_Compound);
        serializer->writeTagHeader("snowy", NBT_TagType::TAG_String);
        serializer->writeString("false");
        serializer->writeTagEnd();
        break;
    case BlockType::Water:
        serializer->writeString("minecraft:water");
        serializer->writeTagHeader("Properties", NBT_TagType::TAG_Compound);
        serializer->writeTagHeader("level", NBT_TagType::TAG_Byte);
        serializer->writeByte(0);
        serializer->writeTagEnd();
        break;
    case BlockType::Sand:
        serializer->writeString("minecraft:sand");
        break;
    case BlockType::Gravel:
        serializer->writeString("minecraft:gravel");
        break;
    case BlockType::Shortgrass:
        serializer->writeString("minecraft:short_grass");
        break;
    case BlockType::Kelp:
        serializer->writeString("minecraft:kelp");
        serializer->writeTagHeader("Properties", NBT_TagType::TAG_Compound);
        serializer->writeTagHeader("age", NBT_TagType::TAG_Int);
        serializer->writeInt(25);
        serializer->writeTagEnd();
        break;
    case BlockType::KelpPlant:
        serializer->writeString("minecraft:kelp_plant");
        break;
    case BlockType::Stone_CoalOre:
        serializer->writeString("minecraft:coal_ore");
        break;
    case BlockType::Stone_IronOre:
        serializer->writeString("minecraft:iron_ore");
        break;
    case BlockType::Stone_CopperOre:
        serializer->writeString("minecraft:copper_ore");
        break;
    case BlockType::Stone_GoldOre:
        serializer->writeString("minecraft:gold_ore");
        break;
    case BlockType::Stone_LapisOre:
        serializer->writeString("minecraft:lapis_ore");
        break;
    case BlockType::Stone_RedstoneOre:
        serializer->writeString("minecraft:redstone_ore");
        break;
    case BlockType::Stone_EmeraldOre:
        serializer->writeString("minecraft:emerald_ore");
        break;
    case BlockType::Stone_DiamondOre:
        serializer->writeString("minecraft:diamond_ore");
        break;
    case BlockType::Deepslate_CoalOre:
        serializer->writeString("minecraft:deepslate_coal_ore");
        break;
    case BlockType::Deepslate_IronOre:
        serializer->writeString("minecraft:deepslate_iron_ore");
        break;
    case BlockType::Deepslate_CopperOre:
        serializer->writeString("minecraft:deepslate_copper_ore");
        break;
    case BlockType::Deepslate_GoldOre:
        serializer->writeString("minecraft:deepslate_gold_ore");
        break;
    case BlockType::Deepslate_LapisOre:
        serializer->writeString("minecraft:deepslate_lapis_ore");
        break;
    case BlockType::Deepslate_RedstoneOre:
        serializer->writeString("minecraft:deepslate_redstone_ore");
        break;
    case BlockType::Deepslate_EmeraldOre:
        serializer->writeString("minecraft:deepslate_emerald_ore");
        break;
    case BlockType::Deepslate_DiamondOre:
        serializer->writeString("minecraft:deepslate_diamond_ore");
        break;
    case BlockType::OakLog:
        serializer->writeString("minecraft:oak_log");
        serializer->writeTagHeader("Properties", NBT_TagType::TAG_Compound);
        serializer->writeTagHeader("axis", NBT_TagType::TAG_String);
        serializer->writeString("y");
        serializer->writeTagEnd();
        break;
    case BlockType::OakLeaves:
        serializer->writeString("minecraft:oak_leaves");
        serializer->writeTagHeader("Properties", NBT_TagType::TAG_Compound);
        serializer->writeTagHeader("persistent", NBT_TagType::TAG_String);
        serializer->writeString("false");
        serializer->writeTagHeader("distance", NBT_TagType::TAG_String);
        serializer->writeString("3");
        serializer->writeTagEnd();
        break;
    default:
        break;
    }
    serializer->writeTagEnd();
}
