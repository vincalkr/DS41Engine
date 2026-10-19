# Upstream llama.cpp fixes ported to our vendored ggml (APPLIED 2026-10-14; already in the vendored third_party/llama.cpp/ggml - do not re-apply)

Patch: `tools/v41/ggml/upstream-fixes.patch` — `-p1` relative to `third_party/llama.cpp`.
Source PRs (ggml-org/llama.cpp):
- **#30103** "CUDA/HIP: fix race in flash_attn_ext_f16_process_tile when nbatch_combine != DKQ/2"
  merge `1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3` (2026-10-10); 1 file, +1/-1.
- **#29953** "CUDA: fix MMQ out-of-bounds reads"
  merge `fc9ce6b9d52a8504edcb262abc92737c2289f96c` (2026-10-08); `mmq.cu` +54/-16, `mmq.cuh` +11/-41.

Our vendored tree (`third_party/llama.cpp`, no `.git`) predates BOTH: it still has
`ggml_cuda_mmq_get_J_max` / `mmq_args::ncols_opt` (removed by #29953) and the bare
`if (np > 1)` sync (changed by #30103). It also predates the `prec_src1` refactor
(`ggml_cuda_mmq_get_config` takes no `prec_src1`), so the #29953 hunks are ported
against our API — the `prec_src1` argument is dropped from the J_best loop. Our local
`dedup_bcast` (gate/up broadcast) path in `mmq.cu` is untouched.

## #30103 — flash-attn tile race
Bug: at the end of the VKQ combine loop the barrier guarding the shared tile buffer was
`if (np > 1)`. When a config has `nbatch_combine != DKQ/2`, the combine writes only part
of the DV buffer per pass and warps can still race on `tile_Q` even with `np == 1` →
a thread reads a half-written tile.
Our path: **D=512 prompt (prefill) flash-attn**, fp16 KV, MMA kernel `fattn-mma-f16.cuh`.
Every DKQ=512 config in our table has `nbatch_combine` 64 or 128 (never DKQ/2 = 256), so
the race is live for all of them. Fix: `if (np > 1 || nbatch_combine != DKQ/2)`.
**md5 drift: PLAUSIBLE primary cause.** A missing barrier in the combine is a genuine
read-after-write race on shared memory — output varies run-to-run at fixed input, exactly
the prefill md5 drift signature.
Rebuild: `fattn.cu.o` + all `template-instances/fattn-mma-f16-instance-*.cu.o` (~22 objects).

## #29953 — MMQ out-of-bounds reads
Bug: MMQ tile loads read up to `J-1` columns past the end of `src1_q8_1` / `ids_dst`. The
old buffers were sized with `ggml_cuda_mmq_get_J_max(...)*sizeof(block_q8_1_mmq)` (a J-size
guess) and `ids_dst` had zero slack. The fix computes `J_best` (and `nthreads_best`) once on
the host, derives an explicit `src1_q8_1_padding`, pads `ids_dst` by `J_best-1`, and drops
the J_max guess. It also caps `J <= 128` in the config macro and de-inverts
`mul_mat_q_case`'s fallback selection (readability — same semantics).
Our path: every CUDA MMQ matmul — our **q8_0, q5_K, q4_K and MXFP4** dense/MoE products.
**md5 drift: WEAKER.** These are OOB *reads* (usually stale scratch, not a value that
changes every run); our `mmq-deterministic.patch` already fixed the stream-k split, which is
the drift cause we saw. Still worth taking: it removes reads of uninitialised scratch and
makes `J_best` deterministic.
Rebuild: `mmq.cu.o` plus every TU that includes `mmq.cuh` — `quantize.cu.o`, `mmvq.cu.o`.

## Residuals
- Not applied — dry-run only. No build, no GPU, nothing under `third_party/` touched.
- Stacking: apply AFTER `dsv41-fuse.patch` (already applied) and alongside
  `mmq-deterministic.patch` (already applied). No shared files with fuse/fake-quant/rank-argsort;
  shares only `mmq.cuh` with mmq-deterministic (separate region — both dry-run clean).
- Verify after applying: rebuild + forced-token md5 A/B, prefill D=512, run-to-run.

## Applied (2026-10-14, session 9)
- Applied with `/usr/bin/patch -p1` (anaconda's `patch` binary is corrupt on this box).
- **Required companion change (80476a58):** #29953 turned `mmq_args`' last field from `ncols_opt` into `J_best`;
  `src/prefill/moe_mmq.cu` builds `mmq_args` positionally, so it must compute `J_best` itself (it now does, same rule
  as ggml's host code) - without it: `GGML_ABORT` "J_best=75" in the MoE prefill.  Identity id buffers +128 ints.
- Verified: 512-token prefill logits bit-identical before/after AND run to run; DSpark decode 256 tokens identical.

## kv-fp4.patch (2026-10-15, session 11) - APPLIED in the local tree
- The `--kv-fp4` caches: CUDA set_rows quantize-on-write for NVFP4 (`quantize_f32_nvfp4_ds41_block`) and MXFP4
  (`quantize_f32_mxfp4_ds41_block`) in `cpy-utils.cuh` - EXACT packers for values already on the DS4.1 fake-quant
  grids (they try the scales the grid admits and verify by multiplication; ggml's own MXFP4/NVFP4 rules differ);
  NVFP4 get_rows (`dequantize_nvfp4`, getrows.cu); MXFP4 K in the lightning indexer on the fp32 vec kernel (never the
  fp16 wmma kernel, so the scores equal the BF16 path's); supports_op entries.  Files: cpy-utils.cuh, set-rows.cu,
  getrows.cu, dequantize.cuh, fattn-common.cuh, lightning-indexer.cu, ggml-cuda.cu.  Apply after the others.
- Verified: `test_ds41_kvfp4 --cuda` (5.2M fake-quantized values through set_rows/get_rows, 0 differ) and the engine
  gate (8000-token prompt, exact routing, `STRATA_PRESTAGE_GIB=1.5` both arms): last-row logits bitwise identical.
