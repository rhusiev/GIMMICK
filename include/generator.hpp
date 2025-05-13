#ifndef INCLUDE_GENERATOR_HPP_
#define INCLUDE_GENERATOR_HPP_

#include "./chunk_dOOm_gen.hpp"
#include <cuda_runtime.h>

class FlatInfo {
  public:
    float height;
    float shatter;
};

class ChunkGenerator {
  private:
    uint32_t seed;

  public:
    ChunkGenerator(uint32_t seed) : seed(seed) {};

    std::vector<Chunk> generate_all(int32_t region_x, int32_t region_z);

    __device__ static FlatInfo get_flat_info(int32_t seed, int32_t x,
                                             int32_t z);
    __device__ static void generateSmolChunk(ChunkSmol *chunk_smol,
                                             int32_t seed, int32_t chunk_x,
                                             int32_t chunk_y, int32_t chunk_z,
                                             const FlatInfo *flat_info);
};

#endif // INCLUDE_GENERATOR_HPP_
