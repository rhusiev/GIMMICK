#include "./nbt.hpp"
#include <fstream>
#include <iostream>

int main() {
  McAnvilWriter writer;

  write_chunk(writer.getBufferFor(0, 0));

  {
    std::ofstream file("output.b64", std::ios::binary);
    auto base64 = writer.getBufferFor(0, 0)->asBase64();
    file.write(base64.data(), base64.size());
    file.close();
  }

  {
    std::ofstream file("output.nbt", std::ios::binary);
    file.write(static_cast<const char *>(static_cast<const void *>(
                   writer.getBufferFor(0, 0)->getData())),
               writer.getBufferFor(0, 0)->getOffset());
    file.close();
  }

  std::vector<char> data = writer.serialize();

  {
    std::ofstream file("output.dat", std::ios::binary);
    file.write(data.data(), data.size());
    file.close();
  }

  return 0;
}
