#include "./chunk_dOOm_gen.hpp"
#include "./nbt.hpp"
#include <fstream>
#include <iostream>

int sample_write_chunk() {
    McAnvilWriter writer;

    for (auto x = 0; x < 16; x++) {
        for (auto z = 0; z < 16; z++) {
            Chunk chunk = generate_chunk(x * 16, z * 16);
            write_chunk(writer.getBufferFor(x, z), chunk);
        }
    }

    {
        std::ofstream file("output.b64", std::ios::binary);
        auto base64 = writer.getBufferFor(0, 0)->asBase64();
        file.write(base64.data(), base64.size());
        file.close();
    }

    {
        std::ofstream file("output.nbt", std::ios::binary);
        file.write(static_cast<const char *>(static_cast<const void *>(
                       writer.getBufferFor(0, 0)->getData())),
                   writer.getBufferFor(0, 0)->getOffset());
        file.close();
    }

    std::vector<char> data = writer.serialize();

    {
        std::ofstream file("output.dat", std::ios::binary);
        file.write(data.data(), data.size());
        file.close();
    }

    return 0;
}

int main() {
    Chunk chunk = generate_chunk(0, 0);

#ifdef DEBUG_HEIGHTS
    for (size_t i_x = 0; i_x < 16; i_x++) {
        for (size_t i_z = 0; i_z < 16; i_z++) {
            std::cout << chunk.debug_heights[i_x][i_z] << " ";
        }
        std::cout << std::endl;
    }
#endif

    sample_write_chunk();

    return 0;
}
