#ifndef BLOCK_TEMPLATE_HPP
#define BLOCK_TEMPLATE_HPP

#include "./nbt.hpp"

#include <cstdint>
#include <cstring>
#include <utility>
#include <string_view>
#include <type_traits>
#include <array>
#include <iostream>

template <size_t N>
class Block {
public:
    uint8_t data[N];

    std::string_view as_string() const {
        return std::string_view(reinterpret_cast<const char*>(data), N - 1);
    }

    constexpr Block(const uint8_t* buffer) {
        std::memcpy(data, buffer, N);
    }

    constexpr size_t getSize() const {
        return N;
    }
};

template <typename... Args>
constexpr size_t compute_size(size_t name_len, const Args&... args) {
    constexpr size_t n = sizeof...(args);
    static_assert(n % 2 == 0, "Number of properties must be even");
    static_assert(n == 0, "Properties are not supported yet");

    size_t total = name_len + 2; // length -> string

    // total += 3; // compound tag + length
    // total += 10; // "Properties"
    // total += 5 + key + value;
    // total += 1; // end tag

    return total;
}

template <size_t name_len>
constexpr auto make_block(const char(&blockName)[name_len]) {
    constexpr size_t N = compute_size(name_len - 1); // TODO add properties
    
    OutputBuffer buffer(N);
    NBTSerializer serializer(&buffer);
    
    // Write the block name
    serializer.writeString(blockName);
    
    // Add properties if there are any
    if constexpr (false) {
        // static_assert(sizeof...(propertyArgs) % 2 == 0, "Properties must be key-value pairs");
        
        serializer.writeTagHeader("Properties", NBT_TagType::TAG_Compound);
        // TODO create
        // add_properties(serializer, propertyArgs...);
        serializer.writeTagEnd();
    }

    return Block<N>(buffer.getData());
}

#endif // BLOCK_TEMPLATE_HPP 