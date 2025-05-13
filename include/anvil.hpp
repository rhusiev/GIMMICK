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

class BufferContainer {
  private:
    uint8_t *buffer;
    OutputBuffer *outputBuffer;

  public:
    BufferContainer(size_t size) : outputBuffer(nullptr) {
        cudaMalloc(&buffer, size);

        // Create the OutputBuffer on the device
        OutputBuffer tmpBuffer(buffer, size);
        cudaMalloc(&outputBuffer, sizeof(OutputBuffer));
        cudaMemcpy(&outputBuffer, &tmpBuffer, sizeof(OutputBuffer),
                   cudaMemcpyHostToDevice);
    }
    ~BufferContainer() {
        std::cout << "Freeing buffer" << (void *)buffer << std::endl;
        cudaFree(buffer);
        cudaFree(outputBuffer);
    }

    OutputBuffer *getOutputBuffer() { return outputBuffer; }

    std::vector<uint8_t> getData() {
        std::cout << "Retrieving data from buffer" << (void *)buffer
                  << std::endl;
        // Copy the data from the device to the host
        OutputBuffer outputBufferHost(nullptr, 0);
        cudaMemcpy(&outputBufferHost, outputBuffer, sizeof(OutputBuffer),
                   cudaMemcpyDeviceToHost);

        std::vector<uint8_t> hostBuffer(outputBufferHost.getOffset());
        cudaMemcpy(hostBuffer.data(), buffer, outputBufferHost.getOffset(),
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
