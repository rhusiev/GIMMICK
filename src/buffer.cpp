#include "./buffer.hpp"
#include <cuda_runtime.h>

std::string OutputBuffer::asBase64() const {
    std::string result;
    result.reserve((offset + 2) / 3 * 4);
    static const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    for (size_t i = 0; i < offset; i += 3) {
        uint32_t value = (data[i] << 16) +
                         ((i + 1 < offset) ? (data[i + 1] << 8) : 0) +
                         ((i + 2 < offset) ? data[i + 2] : 0);
        result.push_back(base64_chars[(value >> 18) & 0x3F]);
        result.push_back(base64_chars[(value >> 12) & 0x3F]);
        result.push_back((i + 1 < offset) ? base64_chars[(value >> 6) & 0x3F]
                                          : '=');
        result.push_back((i + 2 < offset) ? base64_chars[value & 0x3F] : '=');
    }
    return result;
}
