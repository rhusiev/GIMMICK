#ifndef INCLUDE_GENERATOR_HPP_
#define INCLUDE_GENERATOR_HPP_

#include "./chunk_dOOm_gen.hpp"
#include <cuda_runtime.h>

class ChunkGenerator {
  public:
    std::vector<Chunk> generate_all(int32_t region_x, int32_t region_z);

    __device__ static float getBaseTerrainHeight(float x, float z);
};

#endif // INCLUDE_GENERATOR_HPP_
