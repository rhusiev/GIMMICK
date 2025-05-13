cmake_minimum_required(VERSION 3.24)
project(gimmick C CXX)

if (NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/bin/clang++")
endif()
set(CMAKE_CUDA_ARCHITECTURES all)
enable_language(CUDA)

add_executable(${PROJECT_NAME})

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.c"
)

target_sources(${PROJECT_NAME}
    PRIVATE
    ${SRC_FILES}
)

add_library(cuda_noise INTERFACE)
target_include_directories(cuda_noise INTERFACE
    libraries/cuda_noise
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} ZLIB::ZLIB cuda_noise)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES all
        COMPILE_FLAGS "-xcuda --cuda-path=/usr/local/cuda"
    )
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES all
    )
endif()
