#include "./anvil.hpp"
#include "./chunk_dOOm_gen.hpp"
#include "./chunk_encoding.hpp"
#include "./generator.hpp"
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iostream>
#include <thread>
#include <thrust/device_vector.h>

void sample_write_chunk(int region_x, int region_z) {
    ChunkGenerator generator;

    McAnvilWriter writer;
    auto buffers = writer.allBuffers();

    auto chunks = generator.generate_all(region_x, region_z);

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
