#ifndef BLOCK_TEMPLATE_HPP
#define BLOCK_TEMPLATE_HPP

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <type_traits>

#include "./nbt.hpp"

template <size_t N> class Block {
  public:
    uint8_t data[N];

    __host__ __device__ constexpr Block(const uint8_t *buffer) {
        for (size_t i = 0; i < N; ++i) {
            data[i] = buffer[i];
        }
    }

    __host__ __device__ constexpr size_t getSize() const { return N; }
};

template <size_t key_str_len, size_t val_str_len> class KeyValue {
  public:
    const char (&key)[key_str_len];
    const char (&val)[val_str_len];
    constexpr size_t size() const { return key_str_len + val_str_len - 2; }
};

template <size_t N, size_t M>
constexpr KeyValue<N, M> make_kv(const char (&k)[N], const char (&v)[M]) {
    return {k, v};
}
#ifndef MAKE_KV
#define MAKE_KV(key, value) \
    [&]() { \
        static constexpr char _k_##__LINE__[] = key; \
        static constexpr char _v_##__LINE__[] = value; \
        return make_kv(_k_##__LINE__, _v_##__LINE__); \
    }()
#endif

template <typename T> struct is_key_value : std::false_type {};

template <size_t K, size_t V>
struct is_key_value<KeyValue<K, V>> : std::true_type {};

template <typename... Args>
constexpr size_t compute_size(Args... args) {
    constexpr size_t n = sizeof...(args);
    /*static_assert(n == 0, "Properties are not supported yet");*/
    static_assert((... && is_key_value<std::decay_t<Args>>::value),
                  "All args must be KeyValue<K, V>");

    size_t total = 0;

    if constexpr (n > 0) {
        total += 3;  // compound tag + length
        total += 10; // "Properties"
        size_t info_about_property = 5;
        size_t args_sizes = (args.size() + ...);
        total += info_about_property * n + args_sizes;
        total += 1; // end tag
    }

    return total;
}

template <typename... Args>
constexpr void add_properties(NBTSerializer &serializer, Args... args) {
    static_assert((... && is_key_value<std::decay_t<Args>>::value),
                  "All propertyArgs must be KeyValue<K, V>");
    (..., (serializer.writeTagHeader(args.key, NBT_TagType::TAG_String),
           serializer.writeString(args.val)));
}

template <auto... propertyArgs, size_t name_len>
__host__ __device__ constexpr auto make_block(const char (&blockName)[name_len]) {
    static_assert(
        (... &&
         is_key_value<std::remove_cvref_t<decltype(propertyArgs)>>::value),
        "All propertyArgs must be KeyValue<K,V>");
    constexpr size_t property_size =
        compute_size(propertyArgs...);
    constexpr size_t N = name_len + 1 + property_size; // length -> string

    uint8_t raw_buffer[N];
    OutputBuffer buffer(raw_buffer, N);
    NBTSerializer serializer(&buffer);

    // Write the block name
    serializer.writeString(blockName);

    // Add properties if there are any
    if constexpr (property_size > 0) {
        serializer.writeTagHeader("Properties", NBT_TagType::TAG_Compound);
        add_properties(serializer, propertyArgs...);
        serializer.writeTagEnd();
    }

    return Block<N>(raw_buffer);
}

#endif // BLOCK_TEMPLATE_HPP
