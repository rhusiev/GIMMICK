#ifndef INCLUDE_BLOCK_REGISTRY_HPP_
#define INCLUDE_BLOCK_REGISTRY_HPP_

#include "./nbt.hpp"
#include "./buffer.hpp"
#include "./block_template.hpp"
#include <cstdint>
#include <cstring>

// Custom string class for efficient storage and comparison
class BlockString {
private:
    const uint8_t* data;
    size_t length;

public:
    BlockString() : data(nullptr), length(0) {}
    BlockString(const uint8_t* str, size_t len) : data(str), length(len) {}
    
    bool operator==(const BlockString& other) const {
        if (length != other.length) return false;
        return memcmp(data, other.data, length) == 0;
    }
    
    // Compare with a C-style array
    template<size_t N>
    bool operator==(const uint8_t (&other)[N]) const {
        if (length != N) return false;
        return memcmp(data, other, length) == 0;
    }
    
    // Compare with a Block<N>
    template<size_t N>
    bool operator==(const Block<N>& block) const {
        if (length != N) return false;
        return memcmp(data, block.data, length) == 0;
    }
    
    const uint8_t* getData() const { return data; }
    size_t getLength() const { return length; }
};

class BlockRegistry {
  private:
    static constexpr size_t MAX_BLOCKS = 256;
    OutputBuffer serialization_buffer;
    BlockString serialized_blocks[MAX_BLOCKS];
    size_t block_count = 0;
    
    template <typename Block>
    int findBlockByString(const Block& block) const {
        for (size_t i = 0; i < block_count; i++) {
            if (serialized_blocks[i] == block) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
    uint32_t getRequiredBitsPerBlock() const;
    
  public:
    BlockRegistry();
    
    template<typename Block>
    uint8_t addBlock(const Block& block) {
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
            serialization_buffer.getData() + start_pos, 
            block.getSize()
        );
        
        return block_count++;
    }
    
    void serializePalette(NBTSerializer* serializer) const;
};

#endif // INCLUDE_BLOCK_REGISTRY_HPP_ 