#ifndef NBT_HPP
#define NBT_HPP

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
      result.push_back((i + 1 < offset) ? base64_chars[(value >> 6) & 0x3F]
                                        : '=');
      result.push_back((i + 2 < offset) ? base64_chars[value & 0x3F] : '=');
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

void write_chunk(OutputBuffer *buf) {
  NBTSerializer serializer(buf);
  serializer.writeTagHeader("", NBT_TagType::TAG_Compound);

  serializer.writeTagHeader("DataVersion", NBT_TagType::TAG_Int);
  serializer.writeInt(4325);

  serializer.writeTagHeader("xPos", NBT_TagType::TAG_Int);
  serializer.writeInt(0);
  serializer.writeTagHeader("zPos", NBT_TagType::TAG_Int);
  serializer.writeInt(0);
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
      serializer.writeTagHeader("block_states", NBT_TagType::TAG_Compound);
      {
        serializer.writeListTagHeader("palette", NBT_TagType::TAG_Compound, 1);
        serializer.writeTagHeader("Name", NBT_TagType::TAG_String);
        if (y % 2 == 0) {
          serializer.writeString("minecraft:stone");
        } else {
          serializer.writeString("minecraft:air");
        }
        serializer.writeTagEnd();
      }
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
    std::unordered_map<std::tuple<int8_t, int8_t>, std::vector<char>, TupleHash>
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

      std::cout << "Chunk at (" << static_cast<int>(x) << ", "
                << static_cast<int>(z) << ") has size: " << size
                << " (full size: " << compressedData.size()
                << ") and offset: " << offset << std::endl;

      offset += size;
    }

    return region;
  }
};

#endif /* end of include guard: NBT_HPP */
