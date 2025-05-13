#include "./anvil.hpp"
#include "./chunk_dOOm_gen.hpp"
#include "./chunk_encoding.hpp"
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iostream>
#include <thread>
#include <thrust/device_vector.h>

void sample_write_chunk(int region_x, int region_z) {
    McAnvilWriter writer;

    std::vector<OutputBuffer *> buffers;
    std::vector<Chunk> chunks;

    for (auto x = region_x * 32; x < region_x * 32 + 32; x++) {
        for (auto z = region_z * 32; z < region_z * 32 + 32; z++) {
            auto buffer = writer.getBufferFor(x, z);
            buffers.push_back(buffer);
        }
    }

    static ChunkGenerator generator;
    auto chunk_array = generator.generate_all(region_x, region_z);

    // Convert the array to a vector for easier handling
    for (int i = 0; i < buffers.size(); i++) {
        chunks.push_back(std::move(chunk_array[i]));
    }

    cudaDeviceSynchronize();

    write_chunks_parallel(buffers, chunks);

    cudaDeviceSynchronize();

    std::vector<char> data = writer.serialize();

    {
        std::ofstream file(std::format("r.{}.{}.mca", region_x, region_z),
                           std::ios::binary);
        file.write(data.data(), data.size());
        file.close();
    }
}

int main() {
    constexpr auto regions = 1;

    for (auto x = 0; x < regions; x++) {
        for (auto z = 0; z < regions; z++) {
            sample_write_chunk(x, z);
        }
    }

    return 0;
}
