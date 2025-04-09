#ifndef NBT_HPP
#define NBT_HPP

#include "./chunk_dOOm_gen.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <unordered_map>
#include <vector>
#include <zlib.h>

enum class NBT_TagType : uint8_t {
    TAG_End = 0,
    TAG_Byte = 1,
    TAG_Short = 2,
    TAG_Int = 3,
    TAG_Long = 4,
    TAG_Float = 5,
    TAG_Double = 6,
    TAG_Byte_Array = 7,
    TAG_String = 8,
    TAG_List = 9,
    TAG_Compound = 10,
    TAG_Int_Array = 11,
    TAG_Long_Array = 12,
};

// TODO RAII with CUDA memory
class OutputBuffer {
  public:
    OutputBuffer(uint8_t *ptr, size_t capacity)
        : data(ptr), capacity(capacity), offset(0) {}

    bool write(const void *src, size_t bytes) {
        if (offset + bytes > capacity)
            return false;
        const uint8_t *srcBytes = static_cast<const uint8_t *>(src);
        for (size_t i = 0; i < bytes; ++i) {
            data[offset + i] = srcBytes[i];
        }
        offset += bytes;
        return true;
    }

    bool writeByte(uint8_t v) { return write(&v, sizeof(v)); }

    uint8_t *getData() const { return data; }

    size_t getOffset() const { return offset; }

    std::string asBase64() const {
        std::string result;
        result.reserve((offset + 2) / 3 * 4);
        static const char base64_chars[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < offset; i += 3) {
            uint32_t value = (data[i] << 16) +
                             ((i + 1 < offset) ? (data[i + 1] << 8) : 0) +
                             ((i + 2 < offset) ? data[i + 2] : 0);
            result.push_back(base64_chars[(value >> 18) & 0x3F]);
            result.push_back(base64_chars[(value >> 12) & 0x3F]);
            result.push_back(
                (i + 1 < offset) ? base64_chars[(value >> 6) & 0x3F] : '=');
            result.push_back((i + 2 < offset) ? base64_chars[value & 0x3F]
                                              : '=');
        }
        return result;
    }

  private:
    uint8_t *data;
    size_t capacity;
    size_t offset;
};

class NBTSerializer {
  private:
    OutputBuffer *buffer;

    bool writeTagType(NBT_TagType type) {
        return buffer->writeByte(static_cast<uint8_t>(type));
    }

  public:
    NBTSerializer(OutputBuffer *buf) : buffer(buf) {}

    template <std::size_t N>
    bool writeTagHeader(const char (&name)[N], NBT_TagType type) {
        if (!writeTagType(type))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) >> 8)))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) & 0xFF)))
            return false;
        if (!buffer->write(name, N - 1))
            return false;
        return true;
    }

    template <std::size_t N>
    bool writeListTagHeader(const char (&name)[N], NBT_TagType type,
                            int32_t length) {
        if (!writeTagType(NBT_TagType::TAG_List))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) >> 8)))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) & 0xFF)))
            return false;
        if (!buffer->write(name, N - 1))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>(type)))
            return false;
        return writeInt(length);
    }

    bool writeTagEnd() { return writeTagType(NBT_TagType::TAG_End); }

    bool writeByte(int8_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v));
    }

    bool writeShort(int16_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v >> 8)) &&
               buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
    }

    bool writeInt(int32_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v >> 24)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 16) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 8) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
    }

    bool writeLong(int64_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v >> 56)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 48) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 40) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 32) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 24) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 16) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 8) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
    }

    template <std::size_t N> bool writeString(const char (&str)[N]) {
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) >> 8)))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) & 0xFF)))
            return false;
        return buffer->write(str, N - 1);
    }
};

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
            serializer->writeTagHeader("Name", NBT_TagType::TAG_String);
            switch (static_cast<BlockType>(i)) {
            case BlockType::Air:
                serializer->writeString("minecraft:air");
                break;
            case BlockType::Stone:
                serializer->writeString("minecraft:stone");
                break;
            case BlockType::Grass:
                serializer->writeString("minecraft:grass_block");
                serializer->writeTagHeader("Properties",
                                           NBT_TagType::TAG_Compound);
                serializer->writeTagHeader("snowy", NBT_TagType::TAG_String);
                serializer->writeString("false");
                serializer->writeTagEnd();
                break;
            default:
                break;
            }
            serializer->writeTagEnd();
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

        // item = 0;

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

struct TupleHash {
    std::size_t operator()(const std::tuple<int8_t, int8_t> &t) const {
        int8_t a, b;
        std::tie(a, b) = t;
        // A basic combination of hash values for each element.
        return std::hash<int>{}(a) * 31 + std::hash<int>{}(b);
    }
};

class McAnvilWriter {
  private:
    std::unordered_map<std::tuple<int8_t, int8_t>, OutputBuffer, TupleHash>
        chunkBuffers;

  public:
    OutputBuffer *getBufferFor(int8_t x, int8_t z) {
        auto key = std::make_tuple(x, z);
        auto it = chunkBuffers.find(key);
        if (it != chunkBuffers.end()) {
            return &it->second;
        } else {
            // Create a new buffer if it doesn't exist
            uint8_t *data =
                new uint8_t[1024 * 1024]; // Allocate memory for the buffer
            OutputBuffer buffer(data, 1024 * 1024);
            // Use insert with std::move instead of operator[]
            auto result = chunkBuffers.insert({key, std::move(buffer)});
            return &result.first->second;
        }
    }

    std::vector<char> serialize() const {
        std::unordered_map<std::tuple<int8_t, int8_t>, std::vector<char>,
                           TupleHash>
            compressedBuffers(chunkBuffers.size());

        for (const auto &pair : chunkBuffers) {
            const auto &key = pair.first;
            const auto &buffer = pair.second;

            // Compress the buffer
            uLongf compressedSize = compressBound(buffer.getOffset());
            std::vector<char> compressedData(compressedSize);
            if (compress(reinterpret_cast<Bytef *>(compressedData.data()),
                         &compressedSize,
                         reinterpret_cast<const Bytef *>(buffer.getData()),
                         buffer.getOffset()) == Z_OK) {
                compressedData.resize(compressedSize);
                compressedBuffers[key] = std::move(compressedData);
            }
        }

        size_t region_size = 8192;
        for (const auto &pair : compressedBuffers) {
            region_size += 5 + pair.second.size();
            // Round up to the nearest 4096 bytes
            region_size += (4096 - (region_size % 4096)) % 4096;
        }
        std::vector<char> region(region_size, 0);

        size_t offset = 2;
        for (size_t index = 0; index < 32 * 32; index++) {
            auto x = index % 32;
            auto z = index / 32;

            auto it = compressedBuffers.find(std::make_tuple(x, z));
            if (it == compressedBuffers.end()) {
                continue;
            }

            auto &compressedData = it->second;

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
};

#endif /* end of include guard: NBT_HPP */
