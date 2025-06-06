#include "./anvil.hpp"
#include "./chunk_dOOm_gen.hpp"
#include "./chunk_encoding.hpp"
#include <format>
#include <fstream>
#include <iostream>
#include <thread>

void sample_write_chunk(int region_x, int region_z) {
    McAnvilWriter writer;

    for (auto x = region_x * 32; x < region_x * 32 + 32; x++) {
        std::vector<std::thread> threads;
        for (auto z = region_z * 32; z < region_z * 32 + 32; z++) {
            auto buffer = writer.getBufferFor(x, z);
            threads.emplace_back(
                [](OutputBuffer *buffer, int x, int z) {
                    Chunk chunk = generate_chunk(x * 16, z * 16);
                    write_chunk(buffer, chunk);
                },
                buffer, x, z);
        }
        for (auto &thread : threads) {
            thread.join();
        }
    }

    std::vector<char> data = writer.serialize();

    {
        std::ofstream file(std::format("r.{}.{}.mca", region_x, region_z),
                           std::ios::binary);
        file.write(data.data(), data.size());
        file.close();
    }
}

int main() {
#ifdef DEBUG_HEIGHTS
    Chunk chunk = generate_chunk(-16, -16);
    for (size_t i_x = 0; i_x < 16; i_x++) {
        for (size_t i_z = 0; i_z < 16; i_z++) {
            std::cout << chunk.debug_heights[i_x][i_z] << " ";
        }
        std::cout << std::endl;
    }
#endif
    constexpr auto regions = 1;

    for (auto x = -regions; x < regions; x++) {
        for (auto z = -regions; z < regions; z++) {
            sample_write_chunk(x, z);
        }
    }

    return 0;
}
