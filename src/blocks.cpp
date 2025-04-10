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
    default:
        break;
    }
    serializer->writeTagEnd();
}
