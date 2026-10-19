# DeepSeek-V4.1-Flash GSQ on AMD (gfx1100, HIP) - how this checkout runs it

This page records how the V4.1 engine was brought up on **2x Radeon RX 7900 XTX (gfx1100, wave32) + Ryzen 9 9900X +
128 GiB RAM**, ROCm 7.14, NVMe (model on `/dev/nvme0n1p4`, one Engram table per drive), and what the numbers are.
The engine (`tools/v41/engine/`) was written for CUDA; here **the whole three-tier expert path runs on HIP**: the
dense half and the routed experts are on the GPU, the RAM arena holds the ranked experts, the file tier reads the
rest, and the Engram tables are read from two files.

## What runs where

| Half | Where | Notes |
| --- | --- | --- |
| Dense half (attention, router, shared expert, Engram add, head) | RX 7900 XTX via `ggml-hip` | `--backend cuda` picks the GPU by type, so the HIP build keeps the same flag |
| Routed experts (PTQ1_0 / Q2_0 / Q3_K, per-expert formats) | RX 7900 XTX, through the `DS4_MOE_CUDA` tier compiled as HIP | `--experts gpu`: VRAM cache (~2100 experts) + 60 GiB pinned RAM arena (6708) + file tier (8652) |
| Engram tables (202.8 GB) | two files, one per NVMe, read at run time | `--engram A:B`; the raw-device fast path needs `/dev/nvme*n1` access this user does not have, so it uses filesystem preads (see Engram below) |
| CPU expert pool | unused with `--experts gpu` | `--experts cpu` still works (disk-bound, ~1.4 tok/s) |

## Build

The engine's own sources and `strata_kernels` compile in minutes. ggml's HIP backend is the expensive part (the
flash-attention template instances did not finish in >40 min at 20 jobs on this box), so the build reuses a ggml that
the same llama.cpp checkout already built:

```sh
# 1. ggml with the DSV4 hyper-connection ops (third_party/llama.cpp/ggml), built once for gfx1100
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...
cmake --build /path/to/llama.cpp/build -j

# 2. this engine, linking that ggml instead of recompiling it
cmake -S . -B build-hip-pre -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DSTRATA_GGML_DIR=/path/to/llama.cpp -DSTRATA_GGML_PREBUILT=/path/to/llama.cpp/build \
      -DBUILD_SHARED_LIBS=ON -DSTRATA_BUILD_TESTS=OFF
make -C build-hip-pre -j ds41_generate

# or, without STRATA_GGML_PREBUILT, the same configure compiles ggml from STRATA_GGML_DIR itself
```

`STRATA_GGML_PREBUILT` is a dev shortcut (empty by default): headers and archives come from the same source tree, so
nothing can drift, and the normal CUDA/CPU paths are untouched.

### The port changes in this tree

1. `CMakeLists.txt`: with `STRATA_ENABLE_HIP` the `BUILD_SHARED_LIBS OFF` / `GGML_STATIC ON` FORCEs are skipped
   (ggml-hip refuses a static link and wants its own shared build). CUDA and CPU builds are unchanged.
2. `CMakeLists.txt` + `tools/v41/cmake/ggml_prebuilt.cmake` (+ `ggml_prebuilt_stubs.cpp`): the prebuilt-ggml option
   above. The stub answers the one backend registration the prebuilt `libggml.a` references but this engine does not
   link (Vulkan on this box); ggml's registry ignores a NULL registration, so Vulkan stays out of the run.
3. `tools/v41/engine/v41_engine.cmake`: `ds41_dense` links `ggml-hip` and keeps the `DS41_HAVE_CUDA` compile path
   (ggml's HIP backend is the CUDA backend's sources compiled as HIP, so it exports `ggml_backend_cuda_*`); the expert
   tier gets a `ds41_moe_hip` flavour - the same `DS4_MOE_CUDA` source, `strata_hip_runtime` force-including the
   compat header - and `ds41_generate` prefers it (HIP tier, else CUDA tier, else CPU tier). With `strata_mmq` present
   the tier also takes the MMQ prompt path (`DS4_MOE_MMQ`), and the DSpark gate takes `DS41_HAVE_CUDA`.
4. `tools/v41/engine/ds41_moe.cpp`: `set_cache_enabled()` touched `im_->gpu->cache`, a member of the CUDA-only `Gpu`
   struct, so the CPU-only flavour did not compile at all; the access is now behind `#if defined(DS4_MOE_CUDA)`.
5. `include/strata/hip_compat/cuda_profiler_api.h`: the one CUDA header the tier includes that ROCm has no
   counterpart for (`cudaProfilerStart/Stop` -> `hipProfilerStart/Stop`).
6. `src/prefill/moe_mmq.cu` + `CMakeLists.txt` (`strata_mmq_api_version()`): the MMQ host API moved after the pinned
   commit (`ggml_cuda_mmq_get_config`'s `prec_src1`, `mul_mat_q_case<type, prec_src1>`); the file carries both
   spellings behind `STRATA_MMQ_PREC_SRC1`, detected from the ggml tree in use.
7. `src/prefill/ggml_cuda_host.cu`: its host symbols are now weak. The V4.1 engine links `ggml-hip`, which defines the
   same `ggml_cuda_error` / `ggml_cuda_info` / `ggml_backend_cuda_context::*`, and the link failed with duplicate
   symbols; weak lets the real backend's copy win while a build without it (the HIP MMQ parity tests) still links.

The tier itself needed no source change: it carries **no device code** (the kernels live in `strata_kernels`, which the
HIP backend compiles for gfx1100), and every CUDA runtime call it makes was already in the compat shim.

Local scripts: `local/run-one-shot.sh` (one-shot), `local/ds41-engine-local.sh` + `local/ds41-local.json` (the
API server), `local/encode_prompt.py` / `local/detok.py` (tokenize a prompt into `--ids-file` and back),
`local/engram_index.py` (merge the per-layer Engram indexes).

## One-shot run

```sh
./local/run-one-shot.sh --ids-file local/prompts/code_review.i32 -n 96   # experts gpu by default, EXPERTS=cpu to fall back
```

Two env settings matter on this machine:

- `STRATA_NO_LARGEPAGES=1` - pin the RAM arena with 4 KiB pages (~15 s for 60 GiB); with THP it takes minutes.
- `--experts cpu` additionally needs `STRATA_ROUTED_HOST=1` (the CPU tier has no device input for the layer-major
  prompt batch: `run_chunk_dev` returns false, "tier run refused") and reads blobs better through the page cache
  (`STRATA_BLOB_MMAP=1` or `STRATA_BLOB_DIRECT=0`). Neither applies to the GPU tier.

## The Engram tables (202.8 GB)

DeepSeek-V4.1 reads an n-gram memory at run time and the GSQ repo does not host it: its two
`model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt` markers name the source. What was done here, and
what it needs:

```sh
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 model.safetensors.index.json config.json \
    model-00047-of-00048.safetensors --local-dir SRCDIR     # ~101.5 GB, holds layers.1.engram.*
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 model-00048-of-00048.safetensors --local-dir SRCDIR
# layers.1 -> shard 47, layers.14 -> shard 48 (the index says which)
python3 tools/v41/extract_engram.py SRCDIR --to layers.1=/home/vincenzo/v41-engram/layers1 \
                                         --to layers.14=/run/media/vincenzo/Dati/v41-engram/layers14
python3 local/engram_index.py  <both dirs>      # one index naming all four tensors; the engine reads the first it finds
```

Notes from doing it here: the extraction needs ~101 GB of free space per layer and checks every tensor's size plus
2000 random rows; the source shards can be deleted right after their layer is extracted. `extract_engram.py` insists
on a `--to` for every engram layer it finds in the index, so one shard at a time is easiest with a two-tensor
index.json in a scratch directory (that is what was done; `local/engram-src*`).

At run time `Ds41Engram::open()` reads the first `model.safetensors.index.json` among the `--engram` directories and
resolves each tensor's filename against all of them, which is why the merged index (distinct filenames per layer) is
what makes the two-drive split work. The engine prefers raw block-device reads when a file has ≥1024 extents and the
device node is readable; `/dev/nvme0n1p1` is `root:disk` here, so it prints
`1973 extents, /dev/nvme0n1p1 not readable - filesystem reads` and uses preads - measured cost ~1 ms/token, which is
why it is left alone.

## Measured (2026-10-15, this box)

Load with the full configuration (256K context, Engram, DSpark): **16.5 s** (arena: 6708 experts in RAM, 8652 file
tier, seeded in 15.8 s).

| Run (server, 256K ctx, Engram, DSpark) | Prompt | Decode |
| --- | ---: | ---: |
| Code-review prompt, 113 tokens in / 96 out | 6.6 tok/s (cold first request) | **18.0-21.5 tok/s** |
| 1536-token prompt / 16 out | **142 tok/s** | 23.4 tok/s |
| 3072-token prompt (one-shot, quiet box, no DSpark) | 165-172 tok/s | - |

Decode A/B with everything else equal (one-shot, 96 tokens, Engram on): **10.35 / 12.68 tok/s without DSpark,
20.98 / 20.55 tok/s with it** - DSpark is worth ~1.7x here.

Where the time goes (one-shot, 96 tokens, DSpark, `pass_v` split over 41 verify passes of 7.05 tokens):
**dense 99.9 ms, tier 365.1 ms**, taps 0.4 ms, logits 1.1 ms per pass. The expert tier - VRAM-hit expert kernels plus
the PCIe/file misses - is ~78% of the decode; 93-97% of the expert reads are VRAM hits at ~2100 slots (14 GiB), with
~29 PCIe misses and ~4.6 file reads per token.

Quality: greedy, thinking off, the code-review prompt gets the missing error handling (and the dropped line
separators / the unbounded `s +=` growth) in order, with the Engram tables enabled the answer points at the
`std::ifstream` open check first. The CPU tier and the HIP tier produce **identical tokens** on the same prompt.

## The KV cache here: 8-bit, not 4-bit

The compressed KV and the indexer keys are the context-sized state (`L.comp` = `{d_head 512, ctx/ratio}` per
KV-source layer, `L.idxk` = `{idx_dim, ctx/ratio}`). The engine's choices:

| Setting | Type | Notes |
| --- | --- | --- |
| default | BF16 (or F32 with `--no-actquant`) | what the server ran before this |
| `--kv-q8` (added here; `--kv-type N` for any ggml type) | Q8_0 | 8 bits; `ggml_set_rows` quantizes on write, `ggml_flash_attn_ext` takes Q8_0 K/V (it is in ggml's supported KV list), the window ring's `ggml_cast` dequantizes on read |
| `--kv-fp4` | NVFP4 compressed / MXFP4 indexer | the 4-bit grids the author's 4090 used for 512K. **Not usable on this tree**: NVFP4/MXFP4 are not in `ggml_cuda_fattn_kv_type_supported()` (`fattn.cu`), so the attention would have no kernel for them - `tools/v41/ggml/kv-fp4.patch` adds the set_rows packers, not attention kernels |

Measured at 256K with everything else equal (one-shot, Engram, DSpark off for the logits):

| | expert slots after the dense half | logits vs BF16 | prompt 1536 tok | decode 48 tok |
| --- | --- | --- | --- | --- |
| BF16 (default) | 13.71 GiB | - | 115.8 / 119.3 tok/s | 15.86 tok/s |
| `--kv-q8` | **14.08 GiB** | top-1 identical, cosine **0.998987**, mean abs diff 0.152 (logits -11.2..31.0), top-5 overlap 5/5 | 100.5 / 107.1 tok/s | 16.53 tok/s |

So the 8-bit cache is effectively lossless here and buys 0.37 GiB of expert slots, but the quantized-KV attention
kernel costs ~10% on the prompt path (the vector kernel instead of the f16 MMA one). This checkout's server runs
`--kv-q8` (the choice for 256K and the setting that scales to 512K); BF16 is one flag away if prompt speed matters
more than the 0.37 GiB. 4-bit is out of the question both for quality and because the attention kernels for it are not
in this ggml tree.

## Tuning on one card (measured with `local/bench_ds41.py`, medians of 3-4 runs)

The bench runs configurations round robin and reports medians, because a single run on this box is worth nothing:
decode varies with the *content* DSpark drafts (accepted drafts per pass change how many tokens one verify buys, and
the same prompt can land on a different stream at a tie). Spreads of 25.2 / 27.2 / 24.9 and 26.3 / 23.8 / 16.0 tok/s
were measured for **the same** configuration.

| Knob | Result |
| --- | --- |
| `--arena-gib 60 -> 75` | **+40% decode** (16.4-17.4 -> 23.8-25.2 tok/s), file wait on the prompt 16.6 -> 11.0 s. Consistently reproduced in four batches. 80 GiB was worse (20.9-21.8) and unstable, 85 bimodal (22.4/14.3/14.3), 95-100 left the box at 5 GiB of RAM available; 75 is the adopted value |
| `--arena-gib` sweep 60/65/70/75 | 16.4 / 21.7 / 18.6 / 24.7 - not monotonic (the arena holds a different ranked head at each size, and the page cache left for the Engram tables changes with it) |
| `STRATA_ENGRAM_CACHE_ROWS=20000000` (in-process row cache, default 2M) | neutral: 16.7 / 23.8 vs 17.4 / 24.1 without |
| `--pcie 1.0` (every miss to the CPU pool) | neutral (24.7 vs 25.2) |
| `--route-bias-file 1.5` (routing avoids the file tier: the miss-free ceiling) | tier per pass 634.6 -> 604.3 ms only, decode 23.0 vs 25.2 - **the tier is not dominated by the misses** but by the resident-expert kernels; it also changes the answer |
| `STRATA_PROM=1` (prompt-batch expert kernels) | neutral on decode (23.7 vs 23.8), worse on the prompt (43.1 vs 52.4 tok/s) |
| `STRATA_DSPARK_DEPTH=5` | 4.364 tokens/pass but a 160.7 ms verify vs 3.692 / 128.8 at depth 4: net neutral (25.4 vs 25.2); depth 6 is refused by design (1..5) |
| `--vram-margin 0.25`, `--pf-b 2.0`, `--pcie 0.50`, `--pcie 0.85` | inside the noise |
| `STRATA_TIER_GRAPH=1` (tier graph capture), 1536-token prompt + DSpark, 3 runs a side | inside the noise: base median 19.68 [22.40 19.36 19.68] vs 20.21 [20.21 19.37 20.92]; the first 2-run screen suggested +13.6%, 3 runs did not hold it up (the author measured the same capture as bit-identical and neutral on CUDA) |
| `--vram-margin 0.25` / `STRATA_DMA_OVERLAP=1` / `STRATA_D3_PF=2`, same 3-run A/B | 20.83 / ~20.1 / decode 18.8 with prompt 51.8 -> 57.8 tok/s: nothing outside the spread, `STRATA_D3_PF=2` only moves the prompt |
| serve `--prompt-batch 2048 -> 12288` on a 5.8K-token prompt | 229.8 -> 232.8 tok/s: no measurable difference (the engine sizes the batch from free VRAM anyway) |

Adopted: `--arena-gib 75`, `--kv-q8`, `--ctx 262144`, Engram tables (L1 and L14, both read from the same NVMe: / is nvme0n1p1 and Dati is nvme0n1p4), DSpark depth 4 with the 64-expert
prune and no confidence trim, `--slots auto`.

## Where a decode token goes (and why the second card cannot help)

`STRATA_DEC_PROF=1` (prints every 64 steps), one-shot, no DSpark, 96-token reply, arena 75 GiB, ctx 256K:

```
decode step ms/token: begin 5.49  predict 0.38  attn_router 51.70  tier 19.44  finish 5.77   -> 82.8 ms = 12.1 tok/s
```

For comparison, the author's RTX 4090 breakdown for the same model (`V41_START_HERE.md`) is 37.5 ms/token:
`attn_router 17.8 · tier 16.6 · finish 1.9 · begin 0.9`. So the tiers are close (19.4 vs 16.6 ms) while **our dense
half is ~2.9x slower** (51.7 vs 17.8 ms) - the gap to CUDA on this box is the dense half, not the expert service.

The context size is part of that, with identical tokens (same prompt, greedy; only `--ctx` changes):

| `--ctx` | decode (no DSpark, 32 tok) |
| --- | ---: |
| 4096 | 12.70 tok/s |
| 32768 | 14.10 tok/s |
| 262144 | 9.80 tok/s |

Since the generated tokens are the same, the difference is the attention path itself: the CSA/indexer work grows with
the number of candidate blocks, so a 256K context costs ~30% of decode against 32K. That is the price of the context
the user asked for, and it is the thing to attack next (the indexer/attention kernels, or the op count of the dense
graphs - the fork's fused ops are the cheap experiment, `local/build-ggml-fused.sh`).

A caution measured today: comparing a run **with and without `--engram`** is not a clean A/B. Removing the Engram
changes what the model writes (it is part of the model), so the decode path differs and the numbers move for that
reason (9.4-12.1 tok/s with, 10.4-14.1 without, on the same prompt) - not because the Engram add costs that much.
Contexts (`--ctx`) are the clean comparison above: same tokens, different state size.

The **peer expert tier** (the second card) was implemented, verified bit-identical and measured in this state: it
took 35% of the routed rows off card 0 and still lost ~5%, because the tier it is helping is only 23% of the step.
Design, measurements and the saved 901-line patch: `docs/ds4/V41_AMD_PEER_TIER.md`.

## DSpark on this box (the draft stack runs on HIP)

The Q2_K sidecar loads as a HIP graph (`2.91 GB of weights on the card` with the 64-experts-per-layer prune, 5.14 GB
without) and the one-shot `STRATA_DSPARK=` path works. The draft stack itself is healthy on HIP: the repo's own gate,
`test_ds41_dspark` (now built for the HIP backend too), scores **2.423 accepted drafts/pass = 3.423 tokens/pass,
first draft 92.3%** over 26 steps of a recorded trunk stream - the same range as the CUDA number in
`V41_PROGRESS.md` (3.27). In the server with Engram on, DSpark runs at **3.17-3.20 tokens/pass**.

**The confidence trim is what is not safe here.** `STRATA_DSPARK_CONF=0.6` narrows the verify width to the leading
drafts the confidence head is sure about, which - by the engine's own comment - only changes the verify width and is
lossless in principle. On this model it is not: at depth 4 with the prune it produced a *different and wrong* answer
on the code-review prompt (a `std::string` repetition loop), deterministically, while depth 4 without the trim, depth
2 and the trim without the prune all answered correctly. The greedy stream moves with the verify width at near-ties
(the engine's own notes say as much: "DSpark output = plain greedy output holds up to ties only"), and a narrowed
width landed on a worse stream. So this checkout runs DSpark **without** `STRATA_DSPARK_CONF` (see
`local/ds41-local.json`); the trim stays a knob to re-measure, not a default.

## The MMQ prompt path on HIP

`-DSTRATA_PREFILL_MMQ=ON` builds `strata_mmq` for HIP and the tier takes the CUDA path (`ds41_transcode.cu` compiled
as HIP, `DS4_MOE_MMQ=1`); changes 6 and 7 above are what it took. Measured on a 3072-token prompt, quiet box,
alternating runs (three each): **`--chunk-mmq` 172.4 / 168.8 / 155.9 tok/s (median 168.8) vs `--no-chunk-mmq`
165.1 / 164.6 / 161.0 (median 164.6)** - +2.5%, inside the spread. The PTQ1_0 -> Q2_0 transcode plus the MoE-MMQ
kernels are a wash on RDNA3; they are left on (the engine's default) but they are not where the prompt time is.

## The fork's fused dense ops: the win that matters on this box

The dense half is the majority of a decode step here, and it is launch-bound: ~3300 tiny ggml ops a token. The fork
ships two patches that collapse the worst of them into one node each - `tools/v41/ggml/dsv41-fake-quant.patch`
(the activation quantize-dequantize) and `dsv41-fuse.patch` (the two hyper-connection mixes) - and the engine takes
them with `-DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON` against a ggml built from the patched tree.
`local/build-ggml-fused.sh` does both (the ggml build must be **static**: the engine calls ggml internals that a
shared build hides).

| (same prompt, same flags, 3 runs a side) | decode tok/s |
| --- | ---: |
| stock ops | 12.3 [12.3 13.8 12.3] |
| fused ops | **13.9 [13.9 13.9 13.8]** (+13%) |
| stock ops, with DSpark | 16.6 [16.2 16.6 17.0] |
| **fused ops, with DSpark** | **20.5 [19.1 22.4 20.5]** (+23%) |

The last-token logits are the same model: top-1 identical, mean |delta| 0.156 on logits in -11..31 (the fused op is
the same arithmetic as the composite it replaces, in one node). The server runs this build
(`local/ds41-local.json` sets `DS41_EXE` to `build-hip-fused/ds41_generate`).

With it, the serve numbers on this box are: a 5.8K-token prompt at **230 tok/s** (fused engine, batch 2048 or 12288:
229.8 vs 232.8, i.e. no measurable difference here - the author's one-sweep win at 12288 is on a 20K prompt) and
decode **26-30 tok/s** with DSpark.

## Prompt streaming: layer prestage (the DMA overlap)

`STRATA_PRESTAGE_GIB=<GiB>` keeps a device region where the NEXT routed layer's arena experts are DMA'd on a second
stream while the current layer's dense half runs (`chunk_pstage_issue` / `Gpu::p_buf`); `run_chunk` then launches
from there and streams only the rest. Measured through the server (`--serve`, the path it belongs to), 5787-token
prompt, one request a side, fresh server each time, batch 12288 + `STRATA_HIP_WMMA=1`:

| `STRATA_PRESTAGE_GIB` | prompt tok/s |
| --- | ---: |
| unset (baseline) | 231.2 |
| 1.5 (adopted) | **244.5** |
| 3.0 | 236.2 |

It only moves data (no arithmetic), so the tokens are unchanged. One request a side, so read the +5.8% as the
direction; 1.5 (the author's value) beats 3.0 here. With `STRATA_HIP_WMMA=1` the served prompt on this box goes from
~220 to ~245 tok/s on a 5787-token prompt (both opt-ins are in `local/ds41-local.json`).

## `STRATA_HIP_WMMA` does not apply to this engine (and a measurement that was noise)

There is an int8-KV prompt-attention path on matrix cores in `src/kernels/cuda/qsa_prompt_attn.cu`
(`STRATA_HIP_WMMA=1`, guarded for gfx11/gfx12). A one-run-a-side A/B on the 5787-token prompt first looked like
+5.4% (239.3 -> 252.1 tok/s, same 8 tokens), but that was noise: **the V4.1 engine never enters that code**. Evidence:
0 `prompt_attn` symbols in `ds41_generate` against 752 `flash_attn*` ones (this engine's prompt attention is its own
`native_flash_attn`/`flash_attn_tile`, not the parent engine's `qsa_prompt_attn`), no `strata: STRATA_HIP_WMMA: ...
matrix cores` line in a whole real session's engine log, and no engine source referencing `qsa_prompt_attn`. The env
was removed from `local/ds41-local.json`. Same shape of trap as the hipBLASLt table: an opt-in from the parent engine
that this engine does not link. **A one-run-a-side A/B is not enough to adopt a knob - this is the concrete reason
why.**

## Prompt streaming: layer prestage (the DMA overlap)

`STRATA_PRESTAGE_GIB=<GiB>` keeps a device region where the NEXT routed layer's arena experts are DMA'd on a second
stream while the current layer's dense half runs (`chunk_pstage_issue` / `Gpu::p_buf`); `run_chunk` then launches
from there and streams only the rest. Measured through the server (`--serve`, the path it belongs to), 5787-token
prompt, one request a side, fresh server each time, batch 12288 + `STRATA_HIP_WMMA=1`:

| `STRATA_PRESTAGE_GIB` | prompt tok/s |
| --- | ---: |
| unset (baseline) | 231.2 |
| 1.5 (adopted) | **244.5** |
| 3.0 | 236.2 |

It only moves data (no arithmetic), so the tokens are unchanged. One request a side, so read the +5.8% as the
direction; 1.5 (the author's value) beats 3.0 here. With `STRATA_HIP_WMMA=1` the served prompt on this box goes from
~220 to ~245 tok/s on a 5787-token prompt (both opt-ins are in `local/ds41-local.json`).

## Prompt attention on the matrix cores (gfx11)

`STRATA_HIP_WMMA=1` switches the int8-KV prompt attention to the WMMA path (`src/kernels/cuda/qsa_prompt_attn.cu`,
written for gfx12 but guarded for gfx11 too, which is this card). Measured on the 5787-token prompt, one run a side,
same flags (`--kv-q8`, arena 75, 256K, Engram):

| | prompt tok/s |
| --- | ---: |
| off | 239.3 |
| `STRATA_HIP_WMMA=1` | **252.1** (+5.4%) |

The eight generated tokens are identical with and without it. One run a side, so treat the +5.4% as the direction,
not a settled number; it is in `local/ds41-local.json` (the server) as an opt-in. `STRATA_PA_FAST=1` halves the matrix
work again by dropping the FP16 hi/lo split (rounding-level accuracy, not the FP32-level path) and is **not** adopted:
it trades the model's numerics for speed, which is out of scope here.

## hipBLASLt on the prompt: measured, and it cannot pay here

`tools/hip/` ships hipBLASLt tuning tables per (architecture, library version) - for this card the pair is gfx1100 +
hipBLASLt 1.4.1 = `tools/hip/gfx1100-hipblaslt-100401.txt`, and `tools/hip/tune_hipblaslt.cpp` calibrates more
(`cmake --build build-hip-fused --target tune_hipblaslt`). The **V4.1 engine never links that code**: `ds41_generate`
has zero `strata::prefill::Gemm` symbols (the reader of the table is the Qwen engine's prefill), so the table changes
nothing for DeepSeek.

The tuner on this card, our prompt shapes, f16 (what ggml feeds a BLAS path at a large batch), best hipBLASLt
solution against the tuner's own rocBLAS baseline:

| T | N | K | rocBLAS ms | best hipBLASLt ms | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1536 | 1280 | 5120 | 3.40 | 0.68 | 5.0x |
| 1536 | 32768 | 1280 | 15.90 | 2.78 | 5.7x |
| 1536 | 4608 | 5120 | 11.33 | 0.95 | 11.9x |
| 1536 | 5120 | 2304 | 3.84 | 0.57 | 6.7x |
| 1536 | 129280 | 5120 | 157.22 | 28.09 | 5.6x |
| 5787 | 4608 | 5120 | 24.73 | 4.19 | 5.9x |

So the table idea is real (5-12x over plain rocBLAS) - and still not worth wiring: **the prompt is not GEMM-bound**.
`rocprofv3` on a 5787-token prompt (21.42 s of kernel time):

| family | s | % |
| --- | ---: | ---: |
| expert tier (`native_gu` + `native_down`) | 11.34 | 53.0 |
| flash attention (`flash_attn_tile` + `combine_results`) | 4.77 | 22.3 |
| Q8_0 MMQ (`mul_mat_q<8,128>`) | 0.85 | 4.0 |
| rocBLAS/Tensile GEMM (all of it) | 0.75 | 3.5 |
| quant/dequant, transcode, copies | 0.54 | 2.5 |

A perfect GEMM path would save at most those 0.75 s of 21.4 (3.5%), and the engine's dense prompt does not even use
rocBLAS for most of its matmuls (the Q8_0 MMQ kernels above). The prompt's levers are the expert tier and flash
attention, in that order.

## The tier's ternary decode: every existing alternative measured, none adopted

The tier is the single biggest kernel family (53% of a 5787-token prompt, 37% of a decode step), and its weights are
ternary PTQ1_0 blocks (28 bytes per 128 weights, base-3 digits). Four ways to make it cheaper exist in the tree, and
all four were measured on this card:

| alternative | result |
| --- | --- |
| AMD expert layouts (`STRATA_EXP_MODE=2/4`, what gfx906 picks) | 10.4 / 12.0 vs **13.8** tok/s: wave64 tuning, loses on wave32 |
| `STRATA_PTQ1_QUARTER` (quarter-block dot) | 12.0 vs 13.8 medians: no gain |
| `STRATA_PROM=1` (row decoded once into LDS, dp4a per entry) | 154.2 vs 219.6 prompt tok/s: much worse |
| PTQ1_0 -> Q2_0 (the lossless transcode the prompt MMQ path already uses, `STRATA_MMQ_PTQ1`) | 244.5 vs 239.7 prompt tok/s: +2%, inside the noise |

The last one closes the structural idea too: the Q2_0 kernels are not 3-4x cheaper on this engine's real work (the
per-call difference in the profile is different work per call, not a cheaper op), so transcoding the decode store
would buy a couple of percent at best while costing ~28% more memory for those experts and a large change in
`ds41_moe.cpp`. Not done, on this evidence. What is left is the ternary decode arithmetic itself, which is what the
format costs.

## What `attn_router` really is: the decode is host/sync-bound, not GPU-bound

`rocprofv3 --runtime-trace --kernel-trace` on a decode with a **17K-token state filled** (32 tokens, one-shot, batch
4096). Under the profiler the wall inflates to 139 ms/token (unprofiled ~75), so read the **split**, not the absolute:

| | per token |
| --- | ---: |
| wall (profiled) | 139.0 ms |
| **GPU kernels** | **32.2 ms (23%)** in 4,238 launches |
| therefore outside the kernels | **106.9 ms (77%)** |
| HIP API calls | 2,213 (their summed duration = the whole wall) |

Kernel time by family: `mmvq` 11.7 ms, tier 4.8, `mmvq_f` 3.0, elementwise 2.7, copies/fill 2.2, argsort 1.2,
quant 0.95. The API calls, per token: `hipGetLastError` **685**, `hipLaunchKernel` **520**, `hipMemcpyAsync`
**392**, `hipStreamSynchronize` **215**, `hipGetDevice` **206**, `hipEventRecord` 97, `hipGraphLaunch` 77.

So per layer (~40 a token) the engine issues ~5.4 blocking stream synchronizations and ~13 launches, and the GPU sits
idle while the host walks that sequence. Removing ggml-stream syncs only moves the wait (measured on CUDA too), and on HIP the syncs cost more, and here they are 57-77% of the token: the
dense half is not slow because the kernels are slow, it is slow because of how many times the host waits.

**Consequence for the plan:** the levers on the decode are, in order, (1) fewer synchronizations per layer (an engine
pipeline change, not a kernel), (2) fewer API calls per token - the 685 error queries and the 392 async copies are
the cheapest to remove - and only then (3) kernel work. A new attention or indexer kernel cannot pay while the GPU is
idle three quarters of the time.

### Follow-up: the error checks are noise, the synchronizations are everything

Same trace, cost of each HIP call summed over the decode window (per token):

| call | calls/token | ms/token |
| --- | ---: | ---: |
| `hipStreamSynchronize` | 215 | **105.0** |
| `hipMemcpyAsync` | 392 | 9.7 |
| `hipGraphLaunch` | 77 | 7.2 |
| `hipLaunchKernel` | 520 | 3.2 |
| `hipGetLastError` | 685 | **< 0.3 (not in the top nine)** |

So the "cheap win" I had queued first - making the error checks optional - is worth **nothing** (a third of a
millisecond), and patching the vendored ggml for it would have cost a 20-minute rebuild for noise. Measuring first
saved that. What is left is the thing the numbers point at: 215 stream synchronizations per token, each waiting
~0.49 ms, i.e. the host spends 76% of the token's wall inside a stream sync while the GPU has only ~32 ms of kernel
work. The decode is a per-layer handshake problem, and the fix is in the engine's pipeline (one sync point per layer
instead of ~5.4, a deferred top-k readback), not in any kernel or in ggml.

### Who causes the 215 stream synchronizations (attributed from the trace)

The 6,865 syncs in the decode window, by the call immediately before them:

| call before the sync | per token | what it is |
| --- | ---: | --- |
| `hipMemcpyAsync` | **95.3** | a device->host readback, ~2.4 per layer (routed ids, state) |
| `hipGraphLaunch` | **76.7** | the dense graph is launched and **synchronized right after**, ~1.9 per layer |
| `hipGetLastError` | 42.5 | the launch -> error-query -> sync pattern |
| after the sync | -- | `hipGetDevice` **119.8** (a device query in the wrapper of every following call) |

Two per-layer handshakes show up clearly: the engine launches the dense graph and blocks on it (~1.9x a layer), and
it copies something back to the host and blocks for it (~2.4x a layer). The `layer sync` the author already skips on
this path (`layer_synced` false, `ds41_moe.cpp:2682-2691`) is not among them. `hipGetDevice` 120 times a token is the
cheapest of the lot and lives in the wrapper layer, not in the engine's logic.

### Caching `ggml_cuda_get_device` (measured, no gain, reverted)

`ggml_cuda_get_device()` (ggml-cuda.cu:135) calls `cudaGetDevice` and nearly every op calls it: 120 times per decode
token here. It was cached per thread (invalidated in `ggml_cuda_set_device`) and measured on the same decode, 96
tokens, two runs: `attn_router` **43.61 / 43.62 ms** against 43.35 before, decode **13.39 / 13.61 tok/s** against
13.67. No gain: these calls cost ~0.5 us each, so 120 of them are 0.06 ms of a 75 ms token. The patch was reverted
and ggml rebuilt. Same lesson as the error queries: on this engine the host API *call count* is not the problem, the
*waits* are (215 stream syncs, 105 ms/token), and those wait for GPU work that the engine has serialized.

## A fresh long prompt can crash the engine (found 2026-10-16) and the context penalty is smaller than I thought

**The crash.** A 63,613-token prompt, one-shot, `--prompt-batch 12288 --ctx 262144 --arena-gib 75` (the tuned config):

```
chunk prestage: 2.32 GiB device region
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2194.81 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate ROCm0 buffer of size 2301427712
ds41_dense: graph buffer did not fit - freed the tier's prestage region, retrying
Segmentation fault (core dumped)
```

The dense pass graph wants ~2.19 GiB at that batch, does not fit next to a 75 GiB arena's expert cache at 256K, the
retry frees the prestage region and then **segfaults** instead of falling back to a smaller batch. Same prompt with
`--prompt-batch 4096`: **works**, 63,613 tokens in 174.91 s = **363.7 tok/s** prefill. The serve path caps the batch
from free VRAM itself ("serve: VRAM 14.87 GiB free -> prompt batch 10316"), which is why the long sessions in the
server have not crashed, but a *fresh* 60K+ prompt on the one-shot path is a crash, not a slowdown.

**The context penalty, re-measured with the phase timer** (same binary, `STRATA_DEC_PROF`, 64 decode steps):

| prompt | begin | predict | attn_router | tier | finish | total | decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 113 token (4K state) | 4.08 | 0.46 | 43.35 | 18.68 | 5.61 | 72.2 ms | 13.67 tok/s |
| 63,613 token (64K state) | 0.50 | 0.40 | 57.66 | 11.43 | 5.20 | 75.2 ms | **14.00 tok/s** |

So from 4K to 64K the token does **not** get materially slower: `attn_router` grows by 14 ms and the tier and `begin`
give most of it back. The earlier 12.7 / 14.1 / 9.8 tok/s at ctx 4K / 32K / 256K was confounded. What dominates at
every context is `attn_router` (43-58 ms, 57-77% of the token), and it is **not** the indexer scaling with the
context - which weakens the "batched indexer top-k" lead as the next step.

## DSpark's adaptive bypass: implemented, measured, does NOT pay here

An adaptive scheduler probes four speculative cycles and requires an average of four accepted tokens per cycle;
below that it bypasses speculation, which caps the draft's overhead. Our low-acceptance case looked like the same problem (prose 16.3
tok/s against 26.7 on code), so the bypass was implemented in this engine's serve loop (`STRATA_DSPARK_ADAPT`,
threshold `STRATA_DSPARK_ADAPT_MIN`, probe after 4 passes, re-probe every 64) and measured on the same creative
prompt, 400 tokens, fresh server a side:

| | decode tok/s | accepted/pass |
| --- | ---: | ---: |
| adaptive off | **13.5 / 13.9** | 2.19 |
| adaptive on (threshold 2.0) | 12.1 | bypassed (1.50 at the probe) |
| adaptive on (threshold 1.7) | 12.8 | bypassed (1.50 at the probe) |

Two reasons it cannot win here: the probe's four passes are the *least* predictable part of a generation (the
acceptance rises to 2.19 as the text settles - so the bypass fires on bad evidence), and this engine's speculative
pass is cheap enough that it beats the non-speculative one even at 1.5 accepted/pass (74 ms/token against 78
bypassed). The patch was removed; the numbers are the record. Same shape as the confidence trim: a technique that
pays on another stack (DSpark is 1.7-2x here) does not pay on this one.

## The original serve settings, measured on this box (2026-10-16)

`tools/v41/serve/ds41-engine.sh` + `strata-ds41.json` are the engine's original serve settings. Every one that
differs from ours was measured here through `--serve`, same 5787-token prompt + 64-token reply, fresh server a side:

| setting | author | ours | measured |
| --- | --- | --- | --- |
| `--route-bias` / `--route-bias-file` | 0.5 / 0.3 | 1.0 / 0.6 (engine defaults) | **12.1 vs 25.1 tok/s**: the softer bias routes 2.2x more rows to PCIe/file (4,677/511 vs 2,086/138) and the pass goes 140 -> 274 ms. The author's value is routing fidelity, not speed |
| `--arena-adapt --arena-gib 64` | yes | `--arena-gib 75` fixed | 21.5 vs 25.1 tok/s (7,119 vs 8,229 experts in RAM); the adaptive arena did not grow past its base |
| `STRATA_DSPARK_CONF` | 0.6 | off | changes the generated tokens on 2 of 2 prompts **even with `STRATA_SERVE_PASS_BIAS=0`** (+13% on one, -23% on the other): still not adoptable |
| `STRATA_SERVE_PASS_BIAS` | 0 | 0 (adopted) | not a speed knob: it makes the decode-side prompt passes route exactly. Adopted for fidelity |
| `STRATA_SERVE_STEP_MAX` | 96 | 768 (engine default) | untested; the author's own comment puts the batch/pass crossover at ~765 tokens, so the default keeps small new chunks on the fast pass path |
| `STRATA_DSPARK_DEPTH` | 4 | 4 | 4 is right: 25.1 vs 20.3 (depth 5) vs 17.0 tok/s (depth 6) - deeper drafts add work per pass without raising acceptance |
| `STRATA_DSPARK_KEEP` | 64 | 64 | 64 is right: 25.1 vs 17.3 (32) vs 14.2 tok/s (128): fewer experts make the draft guess worse, more make its forward slower |
| `--ctx` / KV | 524288 / `--kv-fp4` | 262144 / `--kv-q8` | the user's call (no 4-bit KV) |
| `--threads` | (unset) | 24 | untested |
| `systemd-run MemoryMax=84G` | yes | no | robustness, not speed |

Conclusion: on this machine the engine's own defaults beat the author's serve tuning almost everywhere - his values
buy routing fidelity and a smaller memory footprint, not tokens per second. The only thing worth taking from his
script was `STRATA_SERVE_PASS_BIAS=0`.

## ggml's HIP graph capture costs more than it saves here (`GGML_CUDA_DISABLE_GRAPHS=1`)

> **Superseded 2026-10-17:** with exact routing, the peer miss tier and `--pf-b 0` the graphs replay (69 launches, 1.3
> captures per token) and pay: prompt 200 -> 245 tok/s, decode +2-10%. The chat config no longer sets the variable;
> numbers in `V41_EXACT_FAST.md`. The measurement below is the biased-routing state of 2026-10-16.

`rocprofv3 --runtime-trace` on a decode: **6,986 HIP runtime calls per token** - 1,822 launches, 951 `hipMemcpyAsync`
(508 of them host->device), 3,130 `hipGetLastError`, 652 `hipGetDevice`, 176 `hipStreamSynchronize`, and **20
`hipStreamBeginCapture` + 20 `hipStreamEndCapture` per token**. That is the ~14 ms/token between the 69.4 ms of GPU
kernel time and the 83 ms a token really takes. ggml re-captures its dense graph instead of replaying it (its key
includes pointers that this engine moves as the context grows), so the capture is pure cost.

| (same flags, `--kv-q8`, arena 75, 256K, Engram) | prompt tok/s | decode tok/s |
| --- | ---: | ---: |
| graphs on (ggml default) | 232.6 | 22.8 |
| **`GGML_CUDA_DISABLE_GRAPHS=1`** | **240.1** | **24.3** |

Through `--serve`, 5787-token prompt + 96-token reply, fresh server a side (wall 29.1 -> 28.1 s); one-shot, 64
tokens, 3 runs a side: 13.9 [13.9 15.3 13.4] -> **15.4 [15.6 15.4 15.0]**. Adopted in `local/ds41-local.json`.

On the numbers it is not bit-identical, and that is worth stating: the last-token logits of a single step are equal
to the last bit (`--dump-last`, max|delta| 0.0000, top-1 19 both ways), and in a 64-token greedy run the first **50**
tokens match and then a near-tie flips (11 of 64 differ). Same class as every other kernel choice here: the same
model, a different summation order - not a different model. Independent confirmation that graphs do not pay on RDNA: hipfire's experiment measured
4.31 us/launch native against 18.25 us/node with a graph (N=200), hipEngine 1.00x on gfx1100, and rocWMMA+graphs
crashes on gfx1201 (llama.cpp #24961) - the workaround is this same env.

## The second card sits on a PCIe 4.0 x2 uplink

`lspci` topology, this box: card 0 is `03:00.0` behind `02:00.0` (**x16**, 16 GT/s); card 1 is `16:00.0` behind
`15:00.0` (x16) which is behind **`14:00.0`: width 2, 16 GT/s = PCIe 4.0 x2** (~4 GB/s against ~32). The card
negotiates x16, so `current_link_width` on the endpoint alone is misleading - the bridge is the bottleneck.

That is a second, hardware reason the peer expert tier could not win (its per-layer hand-off rides that link), on top
of the dense half being 63% of the step. For two cards here, only work with little cross-traffic pays; a layer split
or a peer store served over x2 does not.

## Expert kernels on gfx1100: the AMD layouts are compiled out, and switching them on is slower

`src/kernels/cuda/iq_kernels.cu` carries two families of grouped-expert kernels: the CUDA layout (mode 0,
`native_gu_kernel`, one output row per warp, 8 rows a block) and the AMD layouts (`native_gu_amd_kernel` modes 1/2/4,
`native_gu_lds_kernel` and `native_gu_fused_kernel` modes 5-8) selected by `STRATA_EXP_MODE` (and per weight type by
`STRATA_EXP_MODE_T<type>`). The AMD layouts are only **compiled** for `STRATA_HIP_GFX906` (`STRATA_EXP_LAYOUTS` is 0
for every other HIP architecture), so on this card the tier has always run mode 0 - the layout the author measured at
~136 GB/s on the fork's ternary weights (nsys).

Measured 2026-10-16, same binary, same prompt/flags, 48-token replies, 3 runs a side (fused build, no DSpark):

| `STRATA_EXP_MODE` | decode tok/s |
| --- | ---: |
| 0 (today's default) | **13.8** [14.0 13.8 10.4] |
| 2 (R2: two rows per warp in one loop) | 10.4 [13.2 8.0 10.4] |
| 4 (32 rows a block) | 12.0 [11.4 14.7 12.0] |

Mode 2 is also what gfx906 lands on for these types through `kExpFallback`, so the AMD layout is not a missing win
here: it is tuned for a 64-lane wavefront and loses on RDNA3's wave32. The kernels can be compiled in as an opt-in
(`-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_EXP_LAYOUTS=1`, patch in `iq_kernels.cu`, default mode stays 0) for anyone who wants
to re-measure; the shipped build keeps the flag off.

## DSpark's confidence trim at 0.9: still not adoptable

`STRATA_DSPARK_CONF=0.9` (the trim that verifies only the leading drafts the confidence head is sure about) on three
prompts, 48 tokens each, against the same run without it: `long` identical and +8%, but `code_review` and `capital`
produce **different tokens** (`capital` also slower, 34.5 -> 22.6 tok/s). Changing the answer on 2 of 3 prompts for an
inconsistent gain is not adoptable; the trim stays off (the 0.6 threshold was rejected earlier for worse).

## The second GPU: implemented, measured, not adopted

The project's **parent engine** (the Qwen one) does support several cards: contiguous layer ranges, one per GPU, each
with the expert cache for its own layers (`docs/MULTI_GPU.md`, `src/core/peer_experts.hpp`, layer split /
`--peer-device`). The **V4.1 engine** had none of that, so a peer expert tier was written for it: `--peer-device N`
with a store on the second card, the pack/launch metadata lambdas factored so a launch can target either card, the
peer's rows deposited into the primary's `d_parts` exactly like the CPU share already does (so the routed sum and the
dense half are untouched), plus a startup seed that gives the peer the second half of card 0's ranked head.

It is **correct**: identical tokens with the peer on and off, and `STRATA_PEER_CHECK=1` compared 4299/4299 of the
peer's rows bit-for-bit against card 0's own computation. It moved real work: 91.5 routed rows/token (~35% of the
258 rows a token routes) came off card 0's hits (10959 -> 6713 rows, every other counter identical).

It is also **slower**: 11.8 vs 12.6 tok/s (no DSpark, 3 reps, round robin). The phase profile above says why - the
expert tier is 23% of the step, the dense half 63%, so taking a third of the tier off card 0 buys ~3% and the
per-layer hand-off (activation upload, metadata, event and wait, row copies, 40 layers a token) costs more than that.
The whole attempt is saved as `local/peer-tier-attempt.patch` (901 lines) with its design note
(`docs/ds4/V41_AMD_PEER_TIER.md`); it is the thing to pick up again if the dense half ever stops being the majority.

For reference, the expert cache's own slope on one card: `--slots 1200` 19.30 tok/s (92.5% hits), `--slots 400` 14.98
(80.0%), `--slots 2100` 5.34 (7.3% - an explicit budget larger than the free VRAM thrashes; leave `--slots auto`).

## Not done yet

- **512K context**: the 8-bit cache makes the state affordable, but nothing above 256K has been run here; the second
  thing `--kv-fp4` would buy (a further ~0.4 GiB at 256K) is not available in this ggml tree, see the KV section.
- **The attention/indexer kernels on RDNA3**: the remaining gap to the author's 4090 is here - `attn_router` 51.7
  ms/token against 17.8, while the expert tier is close (19.4 vs 16.6). The CSA indexer's cost also grows with the
  context (12.7 / 14.1 / 9.8 tok/s at ctx 4K / 32K / 256K, identical tokens). This is kernel work (profile, then port
  or tune the gfx1100 paths), not a setting: it is the next real project, and `DSpark` went from 16.6 to 20.5 tok/s
  with the cheap half of the same idea already done (the fused ops).
- **The second card**, as above (design + measured negative in `V41_AMD_PEER_TIER.md`).
- **`--chunk-mmq` tuning**: the Q2_0 MoE-MMQ kernels have no RDNA3-specific config here; the A/B is a wash.
- **A last tuning batch not run**: `STRATA_D3_PF`, `--prefill-chunk`, `STRATA_TIER_GRAPH` on the fused build (the
  earlier runs of the same knobs on the stock build were inside the noise).

### Tool calls over the local server (fixed 2026-10-16)

The model speaks DeepSeek's own DSML (`<｜DSML｜ invoke name="X">` + `<｜DSML｜ parameter name="K" string="true|false">V<...>`);
Strata's `serve/frontend.py` only knows Qwen's XML call form. Overriding `CALL_START`/`CALL_END`, `call_end` and
`parse_tool_call` (what the shim did) is not enough: `frontend.py`'s `_follower` gate looks for `<function=`,
`<parameter=` or `{` right after the opener, so a DSML call was rejected as prose and left in `content` -
`tool_calls: null`, the client ran nothing. Seen on a real harness session: 488 tool calls through
`llama-cpp/Qwen3.8-Flash-Next` had worked, the one turn on this model produced a raw DSML block instead.

`serve_ds41.py` now **translates the stream** (`ds41_translate`: DSML -> `<tool_call><function=NAME><parameter=K>V`),
holding back a partial tag or element (including a closing tag split across deltas - that one leaked raw until the
hold covered `</｜DSML｜ invoke`), which keeps every heuristic of that parser (code fences, a quoted call, the streamed
body) working on the model's own syntax. Verified against the running server with an OpenAI client: non-streaming
`finish_reason: tool_calls` with the parsed arguments, streaming a `tool_calls` delta in the SSE chunks, whole and
byte-split streams, `string="false"` values typed (`30`, not `"30"`), two calls in one reply, and a quoted call
inside a code fence stays text.

The gate for all of it is already in the tree: `tools/v41/serve/test_serve_ds41.py`
(`t_parser` compares the served path against DeepSeek's own parser at **every split 1..39**, thinking on and
off; `t_http` drives the mock server over HTTP for the OpenAI and Anthropic APIs). It was **red** before the
translation - the markup came out as `content`, the same failure the harness session hit - and is green now:
`... test_serve_ds41.py models/V41-GSQ/tokenizer` prints `all passed` (12 checks). Its server subprocess now
runs under the interpreter running the test (`STRATA_TEST_PY` overrides) instead of the author's
`~/AI/Strata/.venv/bin/python`, which is what kept that half unrunnable on other machines.

## API server (256K context)

```sh
STRATA_HOME=$PWD /path/to/venv/bin/python3 tools/v41/serve/serve_ds41.py \
    --engine strata --config local/ds41-local.json --port 8150
# OpenAI /v1/chat/completions, Anthropic /v1/messages, web UI on http://127.0.0.1:8150/
```

`local/ds41-local.json` is this machine's configuration: `--experts gpu`, `--arena-gib 60`, `--ctx 262144` (256K),
`--engram` to the two table directories, `--prompt-batch 2048`, DSpark depth 4 with the 64-expert prune and no
confidence trim, plus `STRATA_NO_LARGEPAGES=1`. `STRATA_HOME` is a new env hook in `serve_ds41.py` (default
`~/AI/Strata`, unchanged): this checkout ships its own `serve/`, so the shim needs no installed Strata. Two signature
drifts against the newer `serve/` were fixed in the shim: `Service.run(..., force=...)` and
`OutputParser.finish(reason)`.
