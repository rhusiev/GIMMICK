#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <thrust/copy.h>

class OutputBuffer {
  public:
    __host__ __device__ OutputBuffer(uint8_t *data, size_t capacity)
        : data(data), capacity(capacity), offset(0) {}

    __host__ __device__ bool write(const void *src, size_t bytes) {
        if (offset + bytes > capacity)
            return false;

#ifdef __CUDA_ARCH__
        __builtin_memcpy(data + offset, src, bytes);
#else
        std::memcpy(data + offset, src, bytes);
#endif

        offset += bytes;
        return true;
    }

    __device__ bool writeByte(uint8_t v) {
        if (offset + 1 > capacity)
            return false;

        data[offset] = v;
        offset += 1;
        return true;
    }

    __device__ uint8_t *getData() const { return data; }
    __host__ __device__ size_t getOffset() const { return offset; }
    __host__ __device__ size_t getCapacity() const { return capacity; }

    std::string asBase64() const;

  private:
    uint8_t *data;
    size_t capacity;
    size_t offset;
};

#endif /* end of include guard: BUFFER_HPP */
