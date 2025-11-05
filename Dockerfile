FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

RUN apt update -y && apt install clang cmake zlib1g-dev -y

ENTRYPOINT ["/bin/bash"]
