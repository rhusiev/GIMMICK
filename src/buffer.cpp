#include "./buffer.hpp"
#include <cstring>

bool OutputBuffer::write(const void *src, size_t bytes) {
    if (offset + bytes > capacity)
        return false;
    std::memcpy(data.get() + offset, src, bytes);
    offset += bytes;
    return true;
}

bool OutputBuffer::writeByte(uint8_t v) { return write(&v, sizeof(v)); }

std::string OutputBuffer::asBase64() const {
    std::string result;
    result.reserve((offset + 2) / 3 * 4);
    static const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    uint8_t *raw_data = data.get();
    for (size_t i = 0; i < offset; i += 3) {
        uint32_t value = (raw_data[i] << 16) +
                         ((i + 1 < offset) ? (raw_data[i + 1] << 8) : 0) +
                         ((i + 2 < offset) ? raw_data[i + 2] : 0);
        result.push_back(base64_chars[(value >> 18) & 0x3F]);
        result.push_back(base64_chars[(value >> 12) & 0x3F]);
        result.push_back((i + 1 < offset) ? base64_chars[(value >> 6) & 0x3F]
                                          : '=');
        result.push_back((i + 2 < offset) ? base64_chars[value & 0x3F] : '=');
    }
    return result;
}
