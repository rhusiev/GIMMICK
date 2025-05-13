#include "./block_registry.hpp"
#include "./block_template.hpp"
#include <sstream>

BlockRegistry::BlockRegistry() 
    : serialization_buffer(1024)
    , block_count(0) {
    addBlock(make_block("minecraft:air"));
}

void BlockRegistry::serializePalette(NBTSerializer* serializer) const {
    if (block_count == 0) {
        return;
    }

    serializer->writeListTagHeader("palette", NBT_TagType::TAG_Compound, block_count);
    
    for (size_t i = 0; i < block_count; i++) {
        const BlockString& block_str = serialized_blocks[i];

        serializer->writeTagHeader("Name", NBT_TagType::TAG_String);
        serializer->getBuffer()->write(block_str.getData(), block_str.getLength());
        serializer->writeTagEnd();
    }
}

uint32_t BlockRegistry::getRequiredBitsPerBlock() const {
    if (block_count < 2) {
        return 0;
    }
    
    uint32_t n_bits = 0;
    while ((1U << n_bits) < block_count) {
        n_bits++;
    }
    
    return (n_bits < 4) ? 4 : n_bits;
} 