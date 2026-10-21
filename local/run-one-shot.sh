#!/usr/bin/env bash
# local/run-one-shot.sh - one-shot DeepSeek-V4.1-Flash-1017 (ds4 Q2) run on THIS machine (AMD gfx1100, 2x RX 7900 XTX, 128 GiB RAM).
#
#   local/run-one-shot.sh --ids 1,2,3 -n 16 [extra ds41_generate args...]
#   local/run-one-shot.sh --ids-file local/prompts/code_review.i32 -n 96
#   EXPERTS=cpu ARENA_GIB=8 local/run-one-shot.sh --ids-file p.i32 -n 8     (the disk-bound fallback)
#
# Build this used: build-hip-pre (STRATA_ENABLE_HIP=ON, ggml-hip, the DS4_MOE_CUDA tier compiled as HIP) - dense half
# and routed experts on the GPU, the ranked experts in a 60 GiB pinned RAM arena, the rest read from the GGUF.
# Everything lives in this checkout: docs/ds4/AMD_HIP_NOTES.md has the port, the measured numbers and what is still
# missing (the Engram tables above all: without them the engine warns and skips the add, so it runs but it is not the
# real model).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-$PWD/models/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf}"
RANK="${RANK:-$PWD/local/rank-q2.bin}"
EXE="${EXE:-$PWD/build-hip-pre/ds41_generate}"
ARENA="${ARENA_GIB:-60}"
THREADS="${THREADS:-$(nproc)}"
EXPERTS="${EXPERTS:-gpu}"
# 4 KiB pages pin the 60 GiB arena in ~17 s; with THP it takes minutes
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
export GGML_NO_BACKTRACE="${GGML_NO_BACKTRACE:-1}"
if [ "$EXPERTS" = cpu ]; then
    # The layer-major prompt batch hands the experts' routed rows to the tier as a DEVICE tensor whenever the dense
    # half runs on a GPU; the CPU tier has no device input (run_chunk_dev returns false -> "tier run refused"), so
    # the batch path is kept correct by leaving the routed sums on the host.  The GPU tier takes the device path.
    export STRATA_ROUTED_HOST="${STRATA_ROUTED_HOST:-1}"
    # and its blobs come faster through the page cache than through the O_DIRECT twin
    export STRATA_BLOB_MMAP="${STRATA_BLOB_MMAP:-1}"
fi

[ -x "$EXE" ] || { echo "run-one-shot: $EXE missing - build it: make -C build-hip-pre -j$(nproc) ds41_generate" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "run-one-shot: $MODEL missing - run local/fetch-model.sh" >&2; exit 1; }

exec "$EXE" -m "$MODEL" \
    --backend cuda --experts "$EXPERTS" \
    --arena-gib "$ARENA" --threads "$THREADS" \
    --rank "$RANK" \
    "$@"
