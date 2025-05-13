#ifndef CHUNK_ENCODING_HPP
#define CHUNK_ENCODING_HPP

#include "./chunk_dOOm_gen.hpp"
#include "./nbt.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// Device function for writing a single chunk
__device__ void write_chunk_device(OutputBuffer *buf, ChunkSmol *chunk_smol, int32_t x, int32_t z);

// Host function to write chunks in parallel using Thrust
void write_chunks_parallel(std::vector<OutputBuffer*> &buffers, std::vector<Chunk> &chunks);

#endif /* end of include guard: CHUNK_ENCODING_HPP */
