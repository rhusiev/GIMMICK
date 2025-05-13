#ifndef INCLUDE_BLOCK_REGISTRY_HPP_
#define INCLUDE_BLOCK_REGISTRY_HPP_

#include "./block_template.hpp"
#include "./buffer.hpp"
#include "./nbt.hpp"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

// Custom string class for efficient storage and comparison
class BlockString {
  private:
    const uint8_t *data;
    size_t length;

  public:
    __host__ __device__ BlockString() : data(nullptr), length(0) {};
    __device__ BlockString(const uint8_t *data, size_t length)
        : data(data), length(length) {}

    // Compare with a Block<N>
    template <size_t N>
    __device__ bool operator==(const Block<N> &block) const {
        if (length != N)
            return false;
        for (size_t i = 0; i < length; ++i) {
            if (data[i] != block.data[i])
                return false;
        }
        return true;
    }

    __device__ const uint8_t *getData() const { return data; }
    __device__ size_t getLength() const { return length; }
};

class BlockRegistry {
  private:
    static constexpr size_t MAX_SERIALIZED_LENGTH = 32;
    static constexpr size_t MAX_BLOCKS = 16; // TODO change
    BlockString serialized_blocks[MAX_BLOCKS];
    uint8_t raw_output[MAX_SERIALIZED_LENGTH];
    OutputBuffer serialization_buffer;
    size_t block_count;

    template <typename Block>
    __device__ int findBlockByString(const Block &block) const {
        for (size_t i = 0; i < block_count; i++) {
            if (serialized_blocks[i] == block) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

  public:
    __device__ BlockRegistry()
        : serialization_buffer(nullptr, 0), block_count(0) {
        serialization_buffer = OutputBuffer(raw_output, MAX_SERIALIZED_LENGTH);
    }

    __device__ uint32_t getRequiredBitsPerBlock() const {
        if (block_count < 2) {
            return 0;
        }

        uint32_t n_bits = 0;
        while ((1U << n_bits) < block_count) {
            n_bits++;
        }

        return (n_bits < 4) ? 4 : n_bits;
    }

    template <typename Block> __device__ uint8_t addBlock(const Block &block) {
        if (block_count >= MAX_BLOCKS) {
            return 0;
        }

        int existing_index = findBlockByString(block);
        if (existing_index >= 0) {
            return static_cast<uint8_t>(existing_index);
        }

        size_t start_pos = serialization_buffer.getOffset();

        serialization_buffer.write(block.data, block.getSize());

        serialized_blocks[block_count] = BlockString(
            serialization_buffer.getData() + start_pos, block.getSize());

        return block_count++;
    }

    __device__ void serializePalette(NBTSerializer *serializer) const {
        if (block_count == 0) {
            return;
        }

        serializer->writeListTagHeader("palette", NBT_TagType::TAG_Compound,
                                       block_count);

        for (size_t i = 0; i < block_count; i++) {
            const BlockString &block_str = serialized_blocks[i];

            serializer->writeTagHeader("Name", NBT_TagType::TAG_String);
            serializer->getBuffer()->write(block_str.getData(),
                                           block_str.getLength());
            serializer->writeTagEnd();
        }
    }
};

#endif // INCLUDE_BLOCK_REGISTRY_HPP_
