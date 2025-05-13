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
        std::cout << "Added chunk at "
                  << std::format("({},{})", chunks[i].x, chunks[i].z)
                  << std::endl;
    }

    cudaDeviceSynchronize();

    // Write all chunks in parallel using Thrust
    std::cout << "Writing " << chunks.size() << " chunks in parallel..."
              << std::endl;
    write_chunks_parallel(buffers, chunks);

    cudaDeviceSynchronize();

    std::cout << "Serializing region..." << std::endl;
    std::vector<char> data = writer.serialize();
    std::cout << "Region size: " << data.size() << std::endl;

    {
        std::ofstream file(std::format("r.{}.{}.mca", region_x, region_z),
                           std::ios::binary);
        file.write(data.data(), data.size());
        file.close();
    }
    std::cout << "Region written to r." << region_x << "." << region_z << ".mca"
              << std::endl;

    // Clean up is handled by smart pointers now
    std::cout << "Cleanup complete." << std::endl;
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
