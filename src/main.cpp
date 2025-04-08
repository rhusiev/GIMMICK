#include "./chunk_dOOm_gen.hpp"
#include <iostream>

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

    return 0;
}
