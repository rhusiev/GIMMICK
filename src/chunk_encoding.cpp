#include "./chunk_encoding.hpp"
#include "./blocks.hpp"
#include "./nbt.hpp"

void encode_subchunk(ChunkSmol &chunk, NBTSerializer *serializer) {
    // Use the new ChunkSmol method to serialize block states
    chunk.serializeBlockStates(serializer);
}

void write_chunk(OutputBuffer *buf, Chunk &chunk) {
    NBTSerializer serializer(buf);
    serializer.writeTagHeader("", NBT_TagType::TAG_Compound);

    serializer.writeTagHeader("DataVersion", NBT_TagType::TAG_Int);
    serializer.writeInt(4325);

    serializer.writeTagHeader("xPos", NBT_TagType::TAG_Int);
    serializer.writeInt(chunk.x / 16);
    serializer.writeTagHeader("zPos", NBT_TagType::TAG_Int);
    serializer.writeInt(chunk.z / 16);
    serializer.writeTagHeader("yPos", NBT_TagType::TAG_Int);
    serializer.writeInt(-4);

    serializer.writeTagHeader("Status", NBT_TagType::TAG_String);
    serializer.writeString("minecraft:full");
    serializer.writeTagHeader("LastUpdate", NBT_TagType::TAG_Long);
    serializer.writeLong(0);

    serializer.writeListTagHeader("sections", NBT_TagType::TAG_Compound, 24);

    for (int8_t y = -4; y < 20; y++) {
        serializer.writeTagHeader("Y", NBT_TagType::TAG_Byte);
        serializer.writeByte(y);

        {
            serializer.writeTagHeader("block_states",
                                      NBT_TagType::TAG_Compound);
            encode_subchunk(chunk.chunk_smols[y + 4], &serializer);
            serializer.writeTagEnd();
        }

        serializer.writeTagEnd();
    }

    serializer.writeTagEnd();
}
