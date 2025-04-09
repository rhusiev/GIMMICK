#ifndef NBT_HPP
#define NBT_HPP

#include "./buffer.hpp"
#include <cstdint>
#include <cstring>

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

class NBTSerializer {
  private:
    OutputBuffer *buffer;
    bool writeTagType(NBT_TagType type);

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

    bool writeTagEnd();
    bool writeByte(int8_t v);
    bool writeShort(int16_t v);
    bool writeInt(int32_t v);
    bool writeLong(int64_t v);

    template <std::size_t N> bool writeString(const char (&str)[N]) {
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) >> 8)))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) & 0xFF)))
            return false;
        return buffer->write(str, N - 1);
    }
};

#endif /* end of include guard: NBT_HPP */
