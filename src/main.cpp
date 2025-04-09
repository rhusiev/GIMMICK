#include "./anvil.hpp"
#include "./chunk_dOOm_gen.hpp"
#include "./chunk_encoding.hpp"
#include <format>
#include <fstream>
#include <iostream>

void sample_write_chunk(int region_x, int region_z) {
    McAnvilWriter writer;

    for (auto x = region_x * 32; x < region_x * 32 + 32; x++) {
        for (auto z = region_z * 32; z < region_z * 32 + 32; z++) {
            Chunk chunk = generate_chunk(x * 16, z * 16);
            write_chunk(writer.getBufferFor(x, z), chunk);
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
    Chunk chunk = generate_chunk(-16, -16);

#ifdef DEBUG_HEIGHTS
    for (size_t i_x = 0; i_x < 16; i_x++) {
        for (size_t i_z = 0; i_z < 16; i_z++) {
            std::cout << chunk.debug_heights[i_x][i_z] << " ";
        }
        std::cout << std::endl;
    }
#endif

    for (auto x = -1; x < 1; x++) {
        for (auto z = -1; z < 1; z++) {
            sample_write_chunk(x, z);
        }
    }

    return 0;
}
