#ifndef NBT_HPP
#define NBT_HPP

#include "./buffer.hpp"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

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
    
    __host__ __device__ bool writeTagType(NBT_TagType type) {
        return buffer->writeByte(static_cast<uint8_t>(type));
    }

  public:
    __host__ __device__ NBTSerializer(OutputBuffer *buf) : buffer(buf) {}

    // Access the underlying buffer
    __host__ __device__ OutputBuffer *getBuffer() const { return buffer; }

    template <std::size_t N>
    __host__ __device__ bool writeTagHeader(const char (&name)[N],
                                            NBT_TagType type) {
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
    __host__ __device__ bool writeListTagHeader(const char (&name)[N],
                                                NBT_TagType type,
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

    __host__ __device__ bool writeTagEnd() { 
        return writeTagType(NBT_TagType::TAG_End); 
    }
    
    __host__ __device__ bool writeByte(int8_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v));
    }
    
    __host__ __device__ bool writeShort(int16_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v >> 8)) &&
               buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
    }
    
    __host__ __device__ bool writeInt(int32_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v >> 24)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 16) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 8) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
    }
    
    __host__ __device__ bool writeLong(int64_t v) {
        return buffer->writeByte(static_cast<uint8_t>(v >> 56)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 48) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 40) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 32) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 24) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 16) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>((v >> 8) & 0xFF)) &&
               buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
    }

    template <std::size_t N> 
    __host__ __device__ bool writeString(const char (&str)[N]) {
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) >> 8)))
            return false;
        if (!buffer->writeByte(static_cast<uint8_t>((N - 1) & 0xFF)))
            return false;
        return buffer->write(str, N - 1);
    }
};

#endif /* end of include guard: NBT_HPP */
