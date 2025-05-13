#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <cstdint>
#include <cuda_runtime.h>
#include <string>

class OutputBuffer {
  public:
    __host__ __device__ OutputBuffer(uint8_t *data, size_t capacity)
        : data(data), capacity(capacity), offset(0) {}

    __host__ __device__ bool write(const void *src, size_t bytes) {
        if (offset + bytes > capacity)
            return false;
        // CUDA kernel code cannot use std::memcpy; do it byte by byte
        for (size_t i = 0; i < bytes; ++i) {
            data[offset + i] = static_cast<const uint8_t *>(src)[i];
        }

        offset += bytes;
        return true;
    }

    __device__ bool writeByte(uint8_t v) { return write(&v, sizeof(v)); }

    __device__ uint8_t *getData() const { return data; }
    __device__ size_t getOffset() const { return offset; }

    std::string asBase64() const;

  private:
    uint8_t *data;
    size_t capacity;
    size_t offset;
};

#endif /* end of include guard: BUFFER_HPP */
