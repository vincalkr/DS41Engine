# The ds4 Q2 weights on our engine — status (2026-10-17)

Goal: run `DeepSeek-V4.1-Flash-Q2.gguf` (antirez/ds4's Q2, on the external SSD) with this repository's own
`tools/v41` engine. This file records what was verified and measured, so the next person does not have to rediscover
it. The same numbers are in the README's "The ds4 Q2 weights" section.

## Verdict

**It runs.** The geometry is read, the dense half loads, the expert tier seeds, and it generates at 9.0 tok/s with
the default settings and 0.90-0.97 tok/s with exact routing (the GSQ: 10.8 and 3.0-3.2 under the same conditions). It
names the nine circles correctly at exact routing only - the default route bias changes the selection for both models
- and it needs no format work: the expert tier had always computed the right per-expert blob.

## The model

`/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf` — 365,713,686,528 B, arch
`deepseek41`, 1046 tensors. Types: I8 202.8 GB (the two Engram tables are **inside** the gguf, `blk.1.engram_embd`
and `blk.14.engram_embd`), IQ2_XXS 93.4 GB, Q2_K 59.5 GB, Q8_0 7.6 GB, F16 2.2 GB, F32 0.3 GB. Tensor names are the
standard ones (`token_embd.weight`, `output_norm.weight`, `output.weight`, `blk.N.ffn_{gate,up,down}_exps.weight`).

The Engram rows inside the gguf are labelled I8 and their byte count equals the `F8_E4M3` weight + `F8_E8M0` scale
tensors we already hold as `models/engram/layers1|14` (weight `[384,006,168 x 256]` E4M3 + scale
`[384,006,168 x 8]` E8M0 for L1, `384,016,682` rows for L14), so nothing is extracted from the gguf: the engine reads
them from `--engram`, as the GSQ run does.

## What works (measured)

- `./build-hip-fused/test_ds41_geometry "<model>"`: `ok geometry` (D 5120, 40 layers, 64 heads, d_head 512, swa
  128, experts 384/6, ff 2304, idx 32x128 top 512, vocab 129280), `ok V4.1 shape`, `ok engram token map`, `ok full
  tensor map`, `L<n> tiers: IQ2_XXS/Q2_K x384` for all 40 layers, `all passed`.
- The exact blob is right, from `native_fmt` -> `ggml_row_size`/`ggml_blck_size`: gate/up IQ2_XXS at 5120 = 20
  blocks x 66 B = **1320 B/row** x 2304 rows x 2; down Q2_K at 2304 = 9 blocks x 84 B = **756 B/row** x 5120 rows;
  **9,953,280 B** per expert, 384 experts x 40 layers.
- One-shot, `local/prompts/divina.i32`, 200 tokens, greedy, default route bias: **9.01 tok/s** (22.07 s), 1,481 VRAM
  slots, 97.0% VRAM hits; the GSQ on the same binary: 10.81 tok/s, 1,640 slots, 95.2%. Prompt 29 tokens: 12.67 s vs
  5.17 s. Arena seed: 122-142 s for 75 GiB (cold page cache on the external SSD).
- Exact routing (`--route-bias 0 --route-bias-file 0`), chat-encoded `elenca i nove cerchi dell'Inferno, solo i
  nomi` (`local/prompts/circles.i32`), greedy: the Q2 names **all nine** — Limbo, Lussuria, Gola, Avidità, Ira,
  Eresia, Violenza, Frode, Tradimento — in 47 tokens at 0.97 tok/s; the GSQ the same list in 135 tokens at 3.19
  tok/s.
- Through the server (`local/ds41-local-q2-exact.json`, thinking budget 192, `temp 0.8`, `top_p 0.95`,
  `rep 1.1`): `finish=stop`, 263 tokens, the same nine names, **1.1 tok/s** (the 46-token prompt itself took 38.5 s).
- The GSQ's DSpark/MTP sidecar (`models/V41-GSQ/V41-MTP-Vision.Q2_K.gguf`) loads and drafts for the Q2 (1.47
  accepted drafts/pass, 2.47 tokens/pass) but exact-routing decode is I/O-bound and got slower: 0.82 vs 0.97 tok/s.

## The blocker was NOT native_fmt: it was the cache slot count

The reported failure was:

```
tier init: ExpertCache: 152882380800 slots x 1 B = 142.38 GiB, but only 14.94 GiB of VRAM is free
```

The `1 B` is `ExpertCache::open_sized`'s uniform-path parameter (`open(off.back(), ..., 1, err)`), not the expert
blob. 152,882,380,800 = 15,360 slots (40 x 384) x 9,953,280 B. With an automatic `--slots`, `ds41_generate` passes the
free-VRAM budget as `slot_gib` and `slots = n_layers * n_experts`; the init path opened a cache slot for EVERY
expert and `slot_gib` was documented as ignored for uniform geometries. Two fixes:

1. `tools/v41/engine/ds41_moe.cpp`, `uniform_slots()`: when the size is automatic (`slot_gib > 0`), the slot count is
   capped by the budget, `max(1, min(slots, slot_gib * 2^30 / blob))`. An explicit `--slots N` still means N.
2. `tools/v41/engine/ds41_generate.cpp`: `mc.defer_cache = defer_vram` (the ds4 engine's `defer_seed`, which the v41
   fork never set). Without it the init cache holds ~14 GiB, the first serve request measures 0.87 GiB free, releases
   the cache — and a uniform geometry never opens it again, so the decode ran with zero VRAM slots (measured: the server
   decoded at 6.2 tok/s; one-shot was 7.89 before the fix, 9.01 after it).

## The nine circles: route bias, not quantisation

| settings (one-shot, greedy, chat-encoded prompt) | Q2 | GSQ |
|---|---|---|
| default route bias (+1.0 / -0.6) | wrong: "Cerchio della Terra/Cielo/Fiume..." | wrong: "Vestibolo" x8 |
| `STRATA_RB_MARGIN=1` | not run | wrong: "Inferno, Antinferno, Centro, Dis..." at 11.99 tok/s (95.7% hits) |
| `--route-bias 0 --route-bias-file 0` | **all nine**, 0.97 tok/s (45.7% hits) | **all nine**, 3.19 tok/s (45.5% hits) |

So the acceptance test passes for the Q2, but only with the model's own routing; the default bias changes the
selection (and the answer) for both models. Note that `STRATA_RB_MARGIN=1` did **not** reproduce the exact-routing
answer in these runs even though the README's earlier table records it doing so — the margin run was fast (11.99
tok/s, 95.7% VRAM hits), so it is not a routing no-op in this build. Do not use it as a stand-in for `--route-bias 0`
when checking an answer.

Exact routing is expensive because the only `rank.bin` is the GSQ's: the Q2's own routing misses the VRAM cache 54%
of the time and reads ~86 experts/token from the file tier (9.49 MiB each, about 1 GB/s of SSD), 0.97 tok/s against
the GSQ's 3.19. A Q2 `rank.bin` from Q2 routing counts (`tools/v41/engine/v41_rank.py`, the calibration stats from a
Q2 run) is the next speed lever; there is no V4.1 drafter trained for these weights.

## Prose

300 greedy tokens of `raccontami la divina commedia` at exact routing are coherent for both: three cantiche, Virgilio
and Beatrice, Lucifero at the centre, seven cornici, nine heavens, Empireo; the Q2 says 1307-1321, the GSQ 1304-1321.
With the default route bias both degrade (Q2: 1300-1307, "immaginato non morto", repeated sentences). In these
samples the routing explains the difference far better than the 2.06 bpw.

## The two geometry patches (still needed, both uncommitted)

1. `tools/v41/engine/ds41_geometry.hpp` — the tensor-map validation accepts and ignores `blk.N.engram_embd.*`
   (they are not loaded; the engine uses the `--engram` safetensors). The mutation test still rejects every other
   unexpected tensor.
2. `tools/v41/engine/ds41_dense.cpp` — the dense loader skips those same tensors. Without it the dense init tried to
   allocate 197 GiB on the GPU ("VRAM: 15.08 GiB free after the dense half" is what it prints now).

Plus two CMake files from the build work: `tools/ds4/cmake/ggml_cuda.cmake` (new `STRATA_GGML_HIP` option) and
`tools/ds4/cmake/ds4_engine.cmake` (link `ggml-hip` when present).

## How to run it

```
./build-hip-fused/ds41_generate -m "/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf" \
  --ids-file local/prompts/divina.i32 -n 200 --backend cuda --experts gpu --arena-gib 75 --ctx 262144 --kv-q8 \
  --temp 0 --threads 24 --rank models/V41-GSQ/rank.bin --engram models/engram/layers14:models/engram/layers1
```

For a correct answer add `--route-bias 0 --route-bias-file 0` (and expect ~1 tok/s on the Q2). The chat configs are
`local/ds41-local-q2.json` (default bias, no drafter) and `local/ds41-local-q2-exact.json` (exact routing); the GSQ
reference without DSpark is `local/ds41-local-gsq-nodspark.json`.

## Do not waste time here

`tools/ds4/` is a **different** engine: it reads the older `deepseek4.*` metadata keys (V4 Flash) and cannot load this
model ("missing deepseek4.embedding_length"). The model belongs to `tools/v41` (`deepseek41.*`).

## Remaining work

1. A Q2 `rank.bin` (Q2 routing counts -> `v41_rank.py`): the difference between 0.97 and, hopefully, the GSQ's ~3
   tok/s at exact routing, and between 9 and more with the bias.
2. A V4.1 drafter matched to these weights, or a decision to keep the GSQ MTP (it drafts, but I/O-bound decode loses
   by it).
3. The `STRATA_RB_MARGIN` path: it is not equivalent to exact routing in this build, and the README's acceptance
   claim should be re-measured with `--route-bias 0` before being relied on.
