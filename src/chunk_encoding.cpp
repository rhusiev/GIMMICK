#include "./chunk_encoding.hpp"
#include "./blocks.hpp"
#include "./nbt.hpp"

void encode_subchunk(ChunkSmol &chunk, NBTSerializer *serializer) {
    uint32_t unique_blocks[BLOCK_TYPE_COUNT] = {0};

    for (auto y = 0; y < 16; y++) {
        for (auto z = 0; z < 16; z++) {
            for (auto x = 0; x < 16; x++) {
                auto block = chunk.blocks[y][z][x];
                unique_blocks[static_cast<uint32_t>(block.block_type)]++;
            }
        }
    }

    auto n_blocks = 0;
    for (uint32_t i = 0; i < BLOCK_TYPE_COUNT; i++) {
        if (unique_blocks[i] > 0) {
            n_blocks++;
        }
    };

    serializer->writeListTagHeader("palette", NBT_TagType::TAG_Compound,
                                   n_blocks);
    for (uint32_t i = 0; i < BLOCK_TYPE_COUNT; i++) {
        if (unique_blocks[i] > 0) {
            // Use the block description serialization function
            write_block_description(serializer, static_cast<BlockType>(i));
        }
    }

    if (n_blocks < 2) {
        return;
    }

    auto n_bits = 0;
    while ((1 << n_bits) < n_blocks) {
        n_bits++;
    }
    if (n_bits < 4) {
        n_bits = 4;
    };

    auto blocks_per_byte = 64 / n_bits;
    auto n_bytes = 4096 / blocks_per_byte;

    serializer->writeTagHeader("data", NBT_TagType::TAG_Long_Array);
    serializer->writeInt(n_bytes);

    for (auto i = 0; i < n_bytes; i++) {
        auto first_block = i * blocks_per_byte;
        uint64_t item = 0;
        for (auto block = first_block; block < first_block + blocks_per_byte;
             block++) {
            auto y = block / 256;
            auto z = (block % 256) / 16;
            auto x = block % 16;

            auto block_type = chunk.blocks[y][z][x].block_type;
            uint64_t block_id = 0;
            for (uint32_t i = 0; i < static_cast<uint32_t>(block_type); i++) {
                if (unique_blocks[i] > 0) {
                    block_id++;
                }
            }

            auto shift = 64 - ((block - first_block) * n_bits) - n_bits;
            item |= block_id << shift;
        }

        serializer->writeLong(item);
    }
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
            auto subchunk = chunk.chunk_smols[y + 4];
            serializer.writeTagHeader("block_states",
                                      NBT_TagType::TAG_Compound);
            encode_subchunk(subchunk, &serializer);
            serializer.writeTagEnd();
        }

        serializer.writeTagEnd();
    }

    serializer.writeTagEnd();
}
