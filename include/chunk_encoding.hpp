#ifndef CHUNK_ENCODING_HPP
#define CHUNK_ENCODING_HPP

#include "./chunk_dOOm_gen.hpp"
#include "./nbt.hpp"

void encode_subchunk(ChunkSmol &chunk, NBTSerializer *serializer);
void write_chunk(OutputBuffer *buf, Chunk &chunk);

#endif /* end of include guard: CHUNK_ENCODING_HPP */
