#ifndef INCLUDE_CHUNK_DOOM_GEN_HPP_
#define INCLUDE_CHUNK_DOOM_GEN_HPP_

#include "./block_registry.hpp"
#include "./block_template.hpp"
#include "./nbt.hpp"
#include "anvil.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <thrust/device_vector.h>
#include <vector>
/*#define DEBUG_HEIGHTS*/

struct FloatCoord2 {
    float x;
    float z;
};

struct FloatCoord3 {
    float x;
    float y;
    float z;
};
// BlockType enum has been moved to blocks.hpp

class ChunkSmol {
  private:
    uint8_t block_ids[16][16][16];
    BlockRegistry registry;

  public:
    template <typename Block>
    __device__ uint8_t setBlock(int32_t y, int32_t z, int32_t x,
                                const Block &block) {
        uint8_t id = registry.addBlock(block);
        block_ids[y][z][x] = id;
        return id;
    }

    __device__ void setBlock(int32_t y, int32_t z, int32_t x, uint8_t id) {
        block_ids[y][z][x] = id;
    }

    __device__ uint8_t getBlockId(int32_t y, int32_t z, int32_t x) const {
        return block_ids[y][z][x];
    };

    // Add new methods for palette encoding
    __device__ void serializeBlockStates(NBTSerializer *serializer) const {
        // Serialize the palette directly from registry
        registry.serializePalette(serializer);

        // Get bit count required for encoding and encode the block data
        encodeBlockData(serializer);
    }

    __device__ void encodeBlockData(NBTSerializer *serializer) const {
        // First determine the number of bits needed per block
        uint32_t n_bits = registry.getRequiredBitsPerBlock();

        // Skip data section entirely if palette is empty or only has air
        if (n_bits == 0) {
            return;
        }

        uint32_t blocks_per_long = 64 / n_bits;
        uint32_t longs_needed = 4096 / blocks_per_long;
        if (4096 % blocks_per_long != 0) {
            longs_needed++;
        }

        serializer->writeTagHeader("data", NBT_TagType::TAG_Long_Array);
        serializer->writeInt(longs_needed);

        for (uint32_t i = 0; i < longs_needed; i++) {
            uint32_t first_block = i * blocks_per_long;
            uint64_t packed_data = 0;

            for (uint32_t j = 0;
                 j < blocks_per_long && (first_block + j) < 4096; j++) {
                uint32_t block_index = first_block + j;
                uint32_t y = block_index / 256;
                uint32_t z = (block_index % 256) / 16;
                uint32_t x = block_index % 16;

                uint64_t block_id = getBlockId(y, z, x);

                // Minecraft requires most significant bits first
                uint32_t shift = (blocks_per_long - 1 - j) * n_bits;
                packed_data |= (block_id << shift);
            }

            serializer->writeLong(packed_data);
        }
    };
};

struct CudaDeleter {
    void operator()(void *ptr) const noexcept {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err)
                          << std::endl;
            }
        }
    }
};

class Chunk {
  public:
    int32_t x;
    int32_t z;
    std::unique_ptr<ChunkSmol, CudaDeleter> chunk_smols;

    Chunk(int32_t x, int32_t z);
};

// Simplified generation system that only creates stone terrain
class ChunkGenerator {
  public:
    Chunk *generate_all(int32_t region_x, int32_t region_z);

    // Density/noise function
    __device__ static float getBaseTerrainHeight(float x, float z);
};

#endif // INCLUDE_CHUNK_DOOM_GEN_HPP_
