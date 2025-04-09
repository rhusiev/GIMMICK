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
    default:
        break;
    }
    serializer->writeTagEnd();
}
