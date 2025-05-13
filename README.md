# GIMMICK: Generator of Ingame Mojang Minecraft worlds In Cuda - Kool

GIMMICK is a project focused on creating a high-performance procedural world generator for Mojang's Minecraft (Java Edition). Leveraging the parallel processing power of NVIDIA CUDA, this project aims to significantly accelerate the computationally intensive task of generating large and complex Minecraft worlds. The primary motivation is an academic exploration of GPGPU techniques applied to procedural generation, investigating effective parallelization strategies and optimization methods within the CUDA framework.

The current stage involves a baseline sequential generator implemented in C++ on the CPU, capable of producing basic terrain using Simplex noise and serializing it into the standard Minecraft Anvil file format for compatibility with the latest game version. The ultimate goal is to develop a CUDA-accelerated version that demonstrably outperforms the vanilla Minecraft world generator in terms of speed while maintaining a comparable level of environmental complexity, thereby showcasing the potential of GPU acceleration for this application.

# Requirements

- clang, llvm
- cmake
- cuda
- zlib

`sudo apt install clang cmake nvidia-cuda-toolkit zlib1g-dev` should make it for ubuntu
