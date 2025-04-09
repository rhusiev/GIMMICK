#include "./anvil.hpp"
#include <zlib.h>

OutputBuffer *McAnvilWriter::getBufferFor(int8_t x, int8_t z) {
    unsigned int ux = static_cast<unsigned int>(x) % REGION_SIZE;
    unsigned int uz = static_cast<unsigned int>(z) % REGION_SIZE;

    if (chunkBuffers[uz][ux].has_value()) {
        return &chunkBuffers[uz][ux].value();
    } else {
        chunkBuffers[uz][ux].emplace(1024 * 1024);
        return &chunkBuffers[uz][ux].value();
    }
}

std::vector<char> McAnvilWriter::serialize() const {
    std::optional<std::vector<char>> compressedBuffers[REGION_SIZE]
                                                      [REGION_SIZE];

    for (int z = 0; z < REGION_SIZE; z++) {
        for (int x = 0; x < REGION_SIZE; x++) {
            if (chunkBuffers[z][x].has_value()) {
                const auto &buffer = chunkBuffers[z][x].value();

                // Compress the buffer
                uLongf compressedSize = compressBound(buffer.getOffset());
                std::vector<char> compressedData(compressedSize);
                if (compress(reinterpret_cast<Bytef *>(compressedData.data()),
                             &compressedSize,
                             reinterpret_cast<const Bytef *>(buffer.getData()),
                             buffer.getOffset()) == Z_OK) {
                    compressedData.resize(compressedSize);
                    compressedBuffers[z][x].emplace(std::move(compressedData));
                }
            }
        }
    }

    size_t region_size = 8192;
    for (int z = 0; z < REGION_SIZE; z++) {
        for (int x = 0; x < REGION_SIZE; x++) {
            if (compressedBuffers[z][x].has_value()) {
                region_size +=
                    5 + compressedBuffers[z][x]->size(); // 5 bytes for header
                // Round up to the nearest 4096 bytes
                region_size += (4096 - (region_size % 4096)) % 4096;
            }
        }
    }
    std::vector<char> region(region_size, 0);

    size_t offset = 2;
    for (size_t index = 0; index < REGION_SIZE * REGION_SIZE; index++) {
        auto x = index % REGION_SIZE;
        auto z = index / REGION_SIZE;

        if (!compressedBuffers[z][x].has_value()) {
            continue;
        }

        auto &compressedData = compressedBuffers[z][x].value();

        // Save data size
        // Big endian size+1
        region[offset * 4096] = (compressedData.size() + 1) >> 24;
        region[offset * 4096 + 1] = (compressedData.size() + 1) >> 16;
        region[offset * 4096 + 2] = (compressedData.size() + 1) >> 8;
        region[offset * 4096 + 3] = (compressedData.size() + 1);
        // Compression type
        region[offset * 4096 + 4] = 2;
        // Compressed data
        std::copy(compressedData.begin(), compressedData.end(),
                  region.begin() + offset * 4096 + 5);

        // Save location
        region[index * 4] = (offset >> 16) & 0xFF;
        region[index * 4 + 1] = (offset >> 8) & 0xFF;
        region[index * 4 + 2] = offset & 0xFF;
        // Save size
        int size = -((-compressedData.size() - 5) / 4096);
        region[index * 4 + 3] = size;

        offset += size;
    }

    return region;
}
