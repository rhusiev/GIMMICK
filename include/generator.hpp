#ifndef INCLUDE_GENERATOR_HPP_
#define INCLUDE_GENERATOR_HPP_

#include "./chunk_dOOm_gen.hpp"
#include <cuda_runtime.h>

class FlatInfo {
  public:
    float height;
    float shatter;
};

class VolumetricInfo {
  public:
    float density;
    bool cave;
};

class ChunkWrapper {
  private:
    ChunkSmol *smols;
    const FlatInfo *flat_info;

  public:
    __device__ ChunkWrapper(ChunkSmol *smols, const FlatInfo *flat_info)
        : smols(smols), flat_info(flat_info) {};

    template <typename Block>
    __device__ void setBlock(int32_t y, int32_t z, int32_t x,
                             const Block &block) {
        x = x % 16;
        z = z % 16;

        int32_t subchunk_id = y / 16 % 24;
        int32_t subchunk_y = y % 16;

        (smols + subchunk_id)->setBlock(subchunk_y, z, x, block);
    };

    template <typename Block>
    __device__ bool isSameBlock(int32_t y, int32_t z, int32_t x,
                                const Block &block) {
        x = x % 16;
        z = z % 16;

        int32_t subchunk_id = y / 16 % 24;
        int32_t subchunk_y = y % 16;

        return (smols + subchunk_id)->isSameBlock(subchunk_y, z, x, block);
    };

    __device__ const FlatInfo get_flat_info(int32_t x, int32_t z) {
        return flat_info[z * 16 + x];
    };
};

class ChunkGenerator {
  private:
    uint32_t seed;

  public:
    ChunkGenerator(uint32_t seed) : seed(seed) {};

    std::vector<Chunk> generate_all(int32_t region_x, int32_t region_z);

    __device__ static FlatInfo get_flat_info(int32_t seed, int32_t x,
                                             int32_t z);
    __device__ static void
    generateSmolChunk(ChunkSmol *chunk_smol, int32_t seed, int32_t chunk_x,
                      int32_t chunk_y, int32_t chunk_z,
                      const FlatInfo *flat_info,
                      const VolumetricInfo *volumetric_info);
    __device__ static void replaceSurface(ChunkWrapper &chunk, int32_t seed);
};

#endif // INCLUDE_GENERATOR_HPP_
