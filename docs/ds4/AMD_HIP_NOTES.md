# DeepSeek-V4.1-Flash on AMD (gfx1100, HIP) - how this checkout runs it

The engine (`tools/v41/engine/`) was written for CUDA; on this box - two Radeon RX 7900 XTX (gfx1100, wave32), Ryzen 9
9900X, 122 GiB RAM, ROCm 7.14 - the whole expert path runs on HIP. Speed, quality and the configuration are in
[V41_EXACT_FAST.md](V41_EXACT_FAST.md).

## What runs where

| part | where |
| --- | --- |
| dense half (attention, router, shared expert, Engram add, head) | card 0 through `ggml-hip`; `--backend cuda` picks the GPU by type, so the HIP build keeps the same flag |
| routed experts | card 0's VRAM, card 1's VRAM (`--peer-device 1`), the CPU over the pinned RAM arena, and the NVMe - the `DS4_MOE_CUDA` tier compiled as HIP |
| Engram tables | read at run time from the files `--engram` names |

## Build

`./build.sh` builds the vendored ggml (`third_party/llama.cpp/ggml`, with the fork's patches - also kept in
`tools/v41/ggml/`) into `third_party/llama.cpp/build-hip`, statically (the engine calls ggml internals a shared build
hides), then the engine into `build-hip-fused` against it, with `-DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON` (the
fused activation quantize-dequantize and hyper-connection mixes, one node each instead of a chain of small ops) and the
MMQ prompt path (`-DSTRATA_PREFILL_MMQ=ON`). `ROCM`, `GFX`, `JOBS`, `REBUILD_GGML=1` from the environment.

The port changes in this tree:

1. `CMakeLists.txt`: with `STRATA_ENABLE_HIP` the `BUILD_SHARED_LIBS OFF` / `GGML_STATIC ON` FORCEs are skipped.
2. `tools/v41/engine/v41_engine.cmake`: `ds41_dense` links `ggml-hip` and keeps the `DS41_HAVE_CUDA` compile path
   (ggml's HIP backend is the CUDA backend's sources compiled as HIP, so it exports `ggml_backend_cuda_*`); the expert
   tier gets a `ds41_moe_hip` flavour - the same `DS4_MOE_CUDA` source, `strata_hip_runtime` force-including the compat
   header - and `ds41_generate` prefers it.
3. `include/strata/hip_compat/cuda_profiler_api.h`: `cudaProfilerStart/Stop` -> `hipProfilerStart/Stop`.
4. `src/prefill/moe_mmq.cu`: the MMQ host API's two spellings (`prec_src1`) behind `STRATA_MMQ_PREC_SRC1`, detected
   from the ggml tree in use.
5. `src/prefill/ggml_cuda_host.cu`: its host symbols are weak, so the real backend's copy wins when `ggml-hip` is
   linked.

The tier carries no device code of its own: the kernels live in `strata_kernels`, compiled for gfx1100.

## One-shot run

```sh
./local/run-one-shot.sh --ids-file local/prompts/dev-docs.i32 -n 96
```

`STRATA_NO_LARGEPAGES=1` pins the RAM arena with 4 KiB pages (with THP the pinning takes minutes).
`local/encode_prompt.py` / `local/detok.py` turn a text into `--ids-file` ids and back.

## The Engram tables on ext4

The engine prefers raw block-device reads when a table file has >= 1024 extents and the device node is readable;
`/dev/nvme0n1p4` is `root:disk` here, so it prints `... extents, /dev/nvme0n1p4 not readable - filesystem reads` and
uses preads, which on ext4 cost nothing measurable.

## Flash attention at head size 512

V4.1's attention heads are 512 wide, so ggml runs `flash_attn_tile<512, 512, ...>`, not the WMMA kernel RDNA3 has for
heads up to 256: `ggml_cuda_get_best_fattn_kernel` allows AMD WMMA only for heads <= 256 and more than 16 columns, and
the kernel compiles no AMD WMMA device code for `DKQ > 256` (selecting it anyway aborts: "HIP kernel
flash_attn_ext_f16 has no device code"). Using the matrix cores here would take a head-512 kernel written for gfx1100.

## Expert kernels: the AMD layouts are compiled out

`src/kernels/cuda/iq_kernels.cu` has a CUDA layout for the grouped expert kernels (mode 0) and AMD layouts (modes
1-8, `STRATA_EXP_MODE`). The AMD layouts are compiled only for `STRATA_HIP_GFX906` (`STRATA_EXP_LAYOUTS` is 0 on every
other HIP architecture), so on gfx1100 the tier runs mode 0; `-DCMAKE_HIP_FLAGS=-DSTRATA_HIP_EXP_LAYOUTS=1` compiles
them in for a measurement. They are tuned for a 64-lane wavefront.

## The second card sits on a PCIe 4.0 x2 uplink

`lspci`: card 0 is `03:00.0` behind `02:00.0` (x16, 16 GT/s); card 1 is `16:00.0` behind `15:00.0` (x16), which is
behind `14:00.0`: **width 2, 16 GT/s = PCIe 4.0 x2** (~4 GB/s). The card negotiates x16, so the endpoint's
`current_link_width` alone is misleading. HIP reports peer access both ways, but over this path it is no faster than
going through pinned host memory (`hipMemcpyAsync` + sync, median of 300, 2026-10-23):

| bytes | card 1 -> pinned host | card 1 -> card 0, P2P |
| ---: | ---: | ---: |
| 16 KiB | 40.1 us | 46.3 us |
| 48 KiB | 26.1 us | 55.8 us |
| 1 MiB | 335 us (3.1 GB/s) | 372 us (2.8 GB/s) |
| 16 MiB | 5.12 ms (3.3 GB/s) | 5.13 ms (3.3 GB/s) |

So card 1 serves only work with little cross-traffic: the experts it holds, whose rows (a few KB a layer) come back
through pinned host memory.

## API server

`./run.sh` starts `tools/v41/serve/serve_ds41.py` with `local/ds41-local.json` on `127.0.0.1:8150`: OpenAI
`/v1/chat/completions` and `/v1/responses`, Anthropic `/v1/messages`, the web chat and monitor at `/`.
