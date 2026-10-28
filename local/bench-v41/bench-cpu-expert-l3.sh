#!/usr/bin/env bash
# bench-cpu-expert-l3.sh - build and run bench-cpu-expert-l3 (the CPU expert kernels from L3 against DRAM) on the
# pack's three gate/up classes.  Needs ./build.sh first (libstrata_kernels_cpu, ggml-cpu).  Results:
# local/sweep-cpu-l3/summary.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$HERE/local/sweep-cpu-l3; mkdir -p "$OUT"
BIN=$OUT/bench-cpu-expert-l3
G=$HERE/third_party/llama.cpp/build-hip/ggml/src
g++ -O2 -std=c++17 -I"$HERE/include" "$HERE/local/bench-v41/bench-cpu-expert-l3.cpp" -o "$BIN" \
    "$HERE/build-hip-fused/libstrata_kernels_cpu.a" "$G/libggml-cpu.a" "$G/libggml-base.a" -fopenmp -lpthread
SHARD=$HERE/models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf
for c in "29 29" "16 29" "17 29" "16 16"; do
    for t in 12 6; do "$BIN" "$SHARD" $c $t "${E:-2}"; done
done | tee "$OUT/summary.txt"
