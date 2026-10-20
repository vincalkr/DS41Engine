#!/usr/bin/env bash
# build.sh - builds ggml (third_party/llama.cpp/build-hip) and the engine (build-hip-fused/ds41_generate), HIP only.
# The same commands as the README's "2. Build".  ggml is built once (tens of minutes: many HIP kernels); a second run
# only rebuilds what changed.  Environment:
#   ROCM=/opt/rocm      the ROCm install
#   GFX=gfx1100         the GPU architecture (RX 7900 XT / XTX); others are not tested
#   JOBS=$(nproc)       parallel compile jobs
#   REBUILD_GGML=1      configure and build ggml again even if build-hip exists
set -euo pipefail
cd "$(dirname "$0")"
ROCM=${ROCM:-/opt/rocm}
GFX=${GFX:-gfx1100}
JOBS=${JOBS:-$(nproc)}
CXX_HIP="$ROCM/lib/llvm/bin/clang++"
for need in cmake gcc "$CXX_HIP" "$ROCM/bin/hipcc"; do
    command -v "$need" >/dev/null 2>&1 || [ -x "$need" ] || { echo "build.sh: $need not found (ROCm with hipcc and clang++, cmake >= 3.24, gcc)" >&2; exit 1; }
done
COMMON=(-DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES="$GFX" -DCMAKE_HIP_COMPILER="$CXX_HIP")

# a build-hip configured from another source tree (llama.cpp's own CMakeLists, before third_party/ggml-build) cannot
# regenerate here: it is moved aside and built again
B=third_party/llama.cpp/build-hip
if [ -f "$B/CMakeCache.txt" ] && ! grep -q "^CMAKE_HOME_DIRECTORY:INTERNAL=$PWD/third_party/ggml-build$" "$B/CMakeCache.txt"; then
    echo "build.sh: $B was configured from another source tree; moving it to $B.old and building ggml again"
    rm -rf "$B.old"; mv "$B" "$B.old"
fi
if [ ! -f "$B/ggml/src/libggml-base.a" ] || [ -n "${REBUILD_GGML:-}" ]; then
    echo "=== ggml (third_party/llama.cpp/ggml -> third_party/llama.cpp/build-hip)"
    cmake -S third_party/ggml-build -B third_party/llama.cpp/build-hip "${COMMON[@]}" \
        -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=OFF
fi
cmake --build third_party/llama.cpp/build-hip -j "$JOBS"

echo "=== engine (build-hip-fused)"
cmake -S . -B build-hip-fused "${COMMON[@]}" \
    -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" -DSTRATA_GGML_PREBUILT="$PWD/third_party/llama.cpp/build-hip" \
    -DSTRATA_DSV41_HC_MIX=ON -DSTRATA_DSV41_FQ_OP=ON
cmake --build build-hip-fused -j "$JOBS"
echo "=== done: build-hip-fused/ds41_generate"
