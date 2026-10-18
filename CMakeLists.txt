# ds41e - the DeepSeek-V4.1-Flash engine (tools/v41/engine) and the few Strata libraries it links.
#
# This tree used to be a fork of Strata (a Qwen engine with a DeepSeek-V4.1 port on top).  Only the DeepSeek engine
# is left; of Strata's own sources, exactly the objects ds41_generate links remain (traced with -Wl,--trace on
# 2026-10-18): the VRAM expert cache and VMM, pinned host memory, the CPU expert kernels, the HIP MMVQ / IQ kernels and
# ggml's MMQ instances for the prompt path.  The flags below are the ones those files were built with in the full
# tree, so the engine binary is the same.
#
# Build (AMD, ROCm; ggml comes prebuilt from third_party/llama.cpp/build-hip, see docs/ds4/AMD_HIP_NOTES.md):
#   cmake -S . -B build-hip-fused -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
#     -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_GGML_PREBUILT=$PWD/third_party/llama.cpp/build-hip \
#     -DSTRATA_DSV41_HC_MIX=ON -DSTRATA_DSV41_FQ_OP=ON
#   cmake --build build-hip-fused --target ds41_generate -j
cmake_minimum_required(VERSION 3.24)
project(ds41e VERSION 0.1.41 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_HIP "Build for wave32 AMD GPUs (gfx1100 validated)" ON)
option(STRATA_PREFILL_MMQ "ggml's MMQ kernels for the prompt path's experts" ON)
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
set(STRATA_GGML_DIR "" CACHE PATH "the llama.cpp checkout ggml comes from (third_party/llama.cpp, with tools/v41/ggml patches)")
set(STRATA_GGML_PREBUILT "" CACHE PATH "an existing ggml build directory of STRATA_GGML_DIR: link its ggml")
if(NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "ds41e builds the HIP engine only (the CUDA and CPU-only paths left with the Strata tree)")
endif()
if(NOT STRATA_GGML_DIR OR NOT STRATA_GGML_PREBUILT)
  message(FATAL_ERROR "set STRATA_GGML_DIR and STRATA_GGML_PREBUILT (third_party/llama.cpp and its build-hip)")
endif()

include(cmake/hip_backend.cmake)
set(_strata_gpu_runtime_target strata_hip_runtime)
set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})

# ggml's MMQ host API spelling (see src/prefill/moe_mmq.cu: STRATA_MMQ_PREC_SRC1)
function(strata_mmq_api_version target mmq_dir)
  if(EXISTS "${mmq_dir}/mmq.cuh")
    file(READ "${mmq_dir}/mmq.cuh" _strata_mmq_h)
    if(_strata_mmq_h MATCHES "const ggml_prec prec_src1")
      target_compile_definitions(${target} PRIVATE STRATA_MMQ_PREC_SRC1=1)
    endif()
  endif()
endfunction()

add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall> $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror> $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# the GGUF reader and scalar dequantizers (header-only)
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

# ---- ggml (prebuilt from third_party/llama.cpp) ----
enable_language(C)
find_package(Threads REQUIRED)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/cmake/ggml_prebuilt.cmake)

# ---- strata_core: pinned host memory, the memory probe ----
add_library(strata_core STATIC src/core/pinned.cu src/platform/memory.cpp)
target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

# ---- strata_kernels: the HIP MMVQ and IQ expert kernels ----
add_library(strata_kernels STATIC src/kernels/cuda/iq_kernels.cu src/kernels/cuda/native_mmvq.cu)
target_include_directories(strata_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(strata_kernels PRIVATE third_party/ggml)
target_link_libraries(strata_kernels PUBLIC strata_artifact strata_core strata_warnings ${_strata_gpu_runtime_target})

# ---- strata_kernels_cpu: the CPU expert kernels and their thread pool ----
add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp
            src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
            src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/ptq1_0.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include PRIVATE third_party/ggml)
set(_strata_avx512 "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(src/kernels/cpu/expert.cpp src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${_strata_avx512}")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp
                            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# AVX-VNNI copies of the AVX-2 kernels (target("avxvnni") per function, taken where cpu_avxvnni_ok())
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
check_cxx_source_compiles([=[
#include <immintrin.h>
__attribute__((target("avxvnni")))
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set_property(SOURCE src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_AVXVNNI=1)
endif()
target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)

# ---- strata_engine: the VRAM expert cache and VMM ----
add_library(strata_engine STATIC src/core/expert_cache.cpp src/core/vmm.cpp)
target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})

# ---- strata_mmq: ggml's MMQ kernels for the prompt path's experts (pinned ggml tree, compiled as HIP) ----
if(STRATA_PREFILL_MMQ)
  set(_strata_gg "${STRATA_GGML_DIR}/ggml")
  set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
  set(_strata_mmq_sources src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu "${_strata_mmq}/quantize.cu")
  foreach(_t IN ITEMS q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
    list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
  endforeach()
  set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
  add_library(strata_mmq STATIC ${_strata_mmq_sources})
  target_include_directories(strata_mmq PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
                                        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  strata_mmq_api_version(strata_mmq "${_strata_mmq}")
  target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
  target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
endif()

# ---- the DeepSeek-V4.1 engine ----
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/engine/v41_engine.cmake)
