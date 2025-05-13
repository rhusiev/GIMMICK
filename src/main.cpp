#include "./anvil.hpp"
#include "./chunk_dOOm_gen.hpp"
#include "./chunk_encoding.hpp"
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <iostream>

void sample_write_chunk(int region_x, int region_z) {
    McAnvilWriter writer;

    for (auto x = region_x * 32; x < region_x * 32 + 4; x++) {
        for (auto z = region_z * 32; z < region_z * 32 + 4; z++) {
            std::cout << "Writing chunk " << x << ", " << z << std::endl;
            auto buffer = writer.getBufferFor(x, z);
            std::cout << "Buffer retrieved" << std::endl;
            Chunk chunk = generate_chunk(x * 16, z * 16);
            std::cout << "Chunk generated" << std::endl;
            write_chunk(buffer, chunk);
            std::cout << "Chunk written" << std::endl;
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
