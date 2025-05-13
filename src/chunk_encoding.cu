#include "./chunk_encoding.hpp"
#include "./nbt.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// Full device function for chunk encoding
__device__ void write_chunk_device(OutputBuffer *buf, ChunkSmol *chunk_smol, int32_t x, int32_t z) {
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

            // Serialize the block states from the ChunkSmol
            chunk_smol[y + 4].serializeBlockStates(&serializer);

            serializer.writeTagEnd();
        }

        serializer.writeTagEnd();
    }

    serializer.writeTagEnd();
}

// Host function to write multiple chunks in parallel using Thrust
void write_chunks_parallel(std::vector<OutputBuffer*> &buffers, std::vector<Chunk> &chunks) {
    if (buffers.size() != chunks.size() || buffers.empty()) return;
    
    // Create device vectors to hold our data
    thrust::device_vector<OutputBuffer*> d_buffers(buffers.size());
    thrust::device_vector<ChunkSmol*> d_chunk_smols(chunks.size());
    thrust::device_vector<int32_t> d_xs(chunks.size());
    thrust::device_vector<int32_t> d_zs(chunks.size());
    
    // Copy data to device
    for (size_t i = 0; i < chunks.size(); i++) {
        d_buffers[i] = buffers[i];
        d_chunk_smols[i] = chunks[i].chunk_smols.get();
        d_xs[i] = chunks[i].x;
        d_zs[i] = chunks[i].z;
    }
    
    // Use a simple lambda instead of a functor or kernel
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            d_buffers.begin(), d_chunk_smols.begin(), d_xs.begin(), d_zs.begin()
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            d_buffers.end(), d_chunk_smols.end(), d_xs.end(), d_zs.end()
        )),
        [=] __device__ (const thrust::tuple<OutputBuffer*, ChunkSmol*, int32_t, int32_t>& t) {
            write_chunk_device(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), thrust::get<3>(t));
        }
    );
    
    // Ensure all kernels are complete
    cudaDeviceSynchronize();
}
