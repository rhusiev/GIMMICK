#ifndef ANVIL_HPP
#define ANVIL_HPP

#include "buffer.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <optional>
#include <thread>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call)                                                 \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << " at line "           \
                      << __LINE__ << ": " << cudaGetErrorString(err) << " ("   \
                      << err << ")" << std::endl;                              \
            exit(1);                                                           \
        }                                                                      \
    }

class BufferContainer {
  private:
    uint8_t *buffer;
    OutputBuffer *outputBuffer;

  public:
    BufferContainer(size_t size) : outputBuffer(nullptr) {
        CHECK_CUDA_ERROR(cudaMalloc(&buffer, size));
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&outputBuffer, sizeof(OutputBuffer)));
        new (outputBuffer) OutputBuffer(buffer, size);
    }
    ~BufferContainer() {
        CHECK_CUDA_ERROR(cudaFree(buffer));
        CHECK_CUDA_ERROR(cudaFree(outputBuffer));
    }

    OutputBuffer *getOutputBuffer() { return outputBuffer; }

    std::vector<uint8_t> getData() {
        std::vector<uint8_t> hostBuffer(outputBuffer->getOffset());
        cudaMemcpy(hostBuffer.data(), buffer, outputBuffer->getOffset(),
                   cudaMemcpyDeviceToHost);
        return hostBuffer;
    }
};

class McAnvilWriter {
  private:
    static constexpr uint32_t REGION_SIZE = 32; // 32x32 chunks per region
    std::optional<BufferContainer> chunkBuffers[REGION_SIZE][REGION_SIZE];

  public:
    OutputBuffer *getBufferFor(uint32_t x, uint32_t z);
    std::vector<char> serialize();
};

#endif /* end of include guard: ANVIL_HPP */
