#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <cstdint>
#include <memory>
#include <string>

class OutputBuffer {
  public:
    explicit OutputBuffer(size_t capacity)
        : data(std::make_unique<uint8_t[]>(capacity)), capacity(capacity),
          offset(0) {}

    bool write(const void *src, size_t bytes);
    bool writeByte(uint8_t v);

    uint8_t *getData() const { return data.get(); }
    size_t getOffset() const { return offset; }
    std::string asBase64() const;

  private:
    std::unique_ptr<uint8_t[]> data;
    size_t capacity;
    size_t offset = 0;
};

#endif /* end of include guard: BUFFER_HPP */
