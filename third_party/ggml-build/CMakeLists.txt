# third_party/ggml-build - builds the vendored ggml (third_party/llama.cpp/ggml) the way llama.cpp's top-level
# CMakeLists did, without the rest of llama.cpp: ggml as a subdirectory (not ggml's standalone mode, which wants
# tests/, examples/ and ggml.pc.in that llama.cpp's copy does not carry), with the ggml settings llama.cpp passes down.
# The output layout is the one STRATA_GGML_PREBUILT reads: <build>/ggml/src/libggml*.a, <build>/ggml/src/ggml-hip/.
cmake_minimum_required(VERSION 3.14...3.28)
project("ggml-vendored" C CXX)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(BUILD_SHARED_LIBS "build shared libraries" OFF)
# what llama.cpp sets for ggml (its LLAMA_ALL_WARNINGS ON, LLAMA_FATAL_WARNINGS OFF; llamafile and graphs on)
set(GGML_ALL_WARNINGS   ON)
set(GGML_FATAL_WARNINGS OFF)
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()
if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml ggml)
