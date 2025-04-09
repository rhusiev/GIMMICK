#ifndef ANVIL_HPP
#define ANVIL_HPP

#include "buffer.hpp"
#include <cstdint>
#include <optional>
#include <vector>

class McAnvilWriter {
  private:
    static constexpr int REGION_SIZE = 32; // 32x32 chunks per region
    std::optional<OutputBuffer> chunkBuffers[REGION_SIZE][REGION_SIZE];

  public:
    OutputBuffer *getBufferFor(uint32_t x, uint32_t z);
    std::vector<char> serialize() const;
};

#endif /* end of include guard: ANVIL_HPP */
