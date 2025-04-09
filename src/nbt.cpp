#include "./nbt.hpp"

bool NBTSerializer::writeTagType(NBT_TagType type) {
    return buffer->writeByte(static_cast<uint8_t>(type));
}

bool NBTSerializer::writeTagEnd() { return writeTagType(NBT_TagType::TAG_End); }

bool NBTSerializer::writeByte(int8_t v) {
    return buffer->writeByte(static_cast<uint8_t>(v));
}

bool NBTSerializer::writeShort(int16_t v) {
    return buffer->writeByte(static_cast<uint8_t>(v >> 8)) &&
           buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
}

bool NBTSerializer::writeInt(int32_t v) {
    return buffer->writeByte(static_cast<uint8_t>(v >> 24)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 16) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 8) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
}

bool NBTSerializer::writeLong(int64_t v) {
    return buffer->writeByte(static_cast<uint8_t>(v >> 56)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 48) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 40) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 32) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 24) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 16) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>((v >> 8) & 0xFF)) &&
           buffer->writeByte(static_cast<uint8_t>(v & 0xFF));
}
