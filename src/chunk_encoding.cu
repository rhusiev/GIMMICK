#include "./blocks.hpp"
#include "./chunk_encoding.hpp"
#include "./nbt.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// Define a global CUDA kernel for subchunk encoding

__global__ void chunk_writer(OutputBuffer *buf, ChunkSmol *chunk_smols,
                             int32_t x, int32_t z) {
    NBTSerializer serializer(buf);
    serializer.writeTagHeader("", NBT_TagType::TAG_Compound);

    serializer.writeTagHeader("DataVersion", NBT_TagType::TAG_Int);
    serializer.writeInt(4325);

    serializer.writeTagHeader("xPos", NBT_TagType::TAG_Int);
    serializer.writeInt(x / 16);
    serializer.writeTagHeader("zPos", NBT_TagType::TAG_Int);
    serializer.writeInt(z / 16);
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

            // Launch the encoding kernel with a single thread
            // Assuming chunk.chunk_smols[y + 4] is already on the device
            chunk_smols[y + 4].encodeBlockData(&serializer);

            serializer.writeTagEnd();
        }

        serializer.writeTagEnd();
    }

    serializer.writeTagEnd();
}

void write_chunk(OutputBuffer *buf, Chunk &chunk) {
    chunk_writer<<<1, 1>>>(buf, chunk.chunk_smols, chunk.x, chunk.z);
}
