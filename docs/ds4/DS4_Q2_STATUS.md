# The ds4 Q2 weights on our engine — status (2026-10-17)

Goal: run `DeepSeek-V4.1-Flash-Q2.gguf` (antirez/ds4's Q2, on the external SSD) with this repository's own
`tools/v41` engine. This file records what was verified and measured, so the next person does not have to rediscover
it. The same numbers are in the README's "The ds4 Q2 weights" section.

## Verdict

**It runs.** The geometry is read, the dense half loads, the expert tier seeds, and it generates at 9.0 tok/s with
the default settings and **3.1-4.5 tok/s with exact routing** using its own `rank.bin` (0.97 tok/s with the GSQ's
rank; the GSQ: 10.8 and 3.0-3.2 under the same conditions). It names the nine circles correctly at exact routing only
- the default route bias changes the selection for both models - and it needs no format work: the expert tier had
always computed the right per-expert blob.

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
- Exact routing (`--route-bias 0 --route-bias-file 0`) with the **GSQ's** rank, chat-encoded `elenca i nove cerchi
  dell'Inferno, solo i nomi` (`local/prompts/circles.i32`), greedy: the Q2 names **all nine** — Limbo, Lussuria, Gola,
  Avidità, Ira, Eresia, Violenza, Frode, Tradimento — in 47 tokens at 0.97 tok/s; the GSQ the same list in 135 tokens
  at 3.19 tok/s. With the Q2's own rank (`local/rank-q2.bin`, below) the same answer decodes at **4.53 tok/s**.
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

Exact routing is expensive because the only `rank.bin` was the GSQ's: the Q2's own routing misses the VRAM cache 54%
of the time and reads ~86 experts/token from the file tier (9.49 MiB each, about 1 GB/s of SSD), 0.97 tok/s against
the GSQ's 3.19. That is fixed below.

## The Q2's own rank.bin (2026-10-17)

`local/rank-q2.bin`. `ds41_generate --dump-routes DIR` counts how many times the router selects each (layer, expert)
and writes `DIR/LNN.json` in the format `tools/v41/engine/v41_rank.py` reads (`experts[e].n`). The router's `ids`
were already a graph output (`Ds41Dense::Impl::o_ids`) and already copied host-ward for the expert tier, so the
instrumentation changed no arithmetic: the counts are taken where `attn_router` / `batch_ids()` hand the ids to the
tier. The tool's default 9,953,280 B blob is this model's expert exactly, so no `--map` is needed.

Corpus (two calibration runs, exact routing):

- `local/calib-q2/corpus.i32`, 8,154 tokens of mixed text: 2,500 tokens of `docs/DETAILS.md` + 2,500 of
  `ds41_generate.cpp` + 800 of `README.md` + the 656-token `long-it` prompt + `divina` + the 1,669-token Italian
  answer in `local/v41-text.log`.
- `local/prompts/circles.i32` and its 60-token answer, replayed with `STRATA_FORCE_IDS` so the counts are that
  answer's own routing.

Total: **8,296 tokens, 1,991,040 (layer, expert) selections**, 129.6 a pair, in `local/calib-q2/stats-{corpus,circles}`
(merged by `local/calib-q2/merge_stats.py`). `v41_rank.py` then reports: first 8 GiB of experts = 38.0% of the routed
tokens, 16 GiB = 52.4%, 32 GiB = 70.3%, 48 GiB = 81.7%, 64 GiB = 89.4%, 80 GiB = 94.5%, 100 GiB = 98.2%. On the
nine-circles routing (in-sample) the top 1,526 experts (the VRAM seed) cover 47.0%, the top 8,090 (the 75 GiB arena) 88.8%, the
top 11,229 (90 GiB arena + VRAM) 96.6%.

A first rank was built from 6,581 tokens assembled from the prompts already in `local/prompts/` - but 88% of that
corpus was one code-review prompt repeated: it covered 99.0% of *its own* corpus at 80 GiB and only 85.9% of the
nine-circles routing, and decoded at 2.10 tok/s. A repeat-heavy corpus overfits; it was discarded.

| resident set (one-shot, greedy, exact routing, `--ctx 8192`) | decode | file-tier reads |
|---|---|---|
| GSQ `rank.bin`, 75 GiB arena | 0.97 tok/s | ~86/token (36%) |
| Q2 rank, 75 GiB arena (arena duplicates the VRAM head) | 3.13 tok/s | 20/token |
| Q2 rank, `--arena-skip-vram`, 75 GiB arena | 3.49 tok/s | 14.5/token |
| Q2 rank, `--arena-skip-vram`, 90 GiB arena (`memguard.sh 110 82`) | **4.53 tok/s** | 7.4/token |

`--arena-skip-vram` is new and opt-in: the arena holds the ranking past the VRAM cache's head instead of the same
experts again, so the two tiers hold distinct experts (the tier already demotes an evicted VRAM expert into the slot
its replacement vacates; the mode existed without a flag). The nine circles stayed correct in every configuration -
the 47-token "Lussuria, Gola, Avidità..." list in the 75 GiB `--arena-skip-vram` run and the 60-token
"Lussuriosi, Golosi..." list in the other two, both all nine. The measured non-file cost is ~0.13 s/token, so a zero file tier would be ~7.4-7.7 tok/s: to get near 9.01 the
142.4 GiB expert set needs more resident bytes than the 104 GiB this machine can hold (90 GiB arena + 14 GiB VRAM), or a
faster file tier (the reads are serial 9.5 MiB preads at ~0.93 GB/s). Weighting the acceptance prompt's counts 16-64x
reaches 97-100% on it, but that is a rank for that prompt, not for the model; it was not kept.

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

For a correct answer use the Q2's own rank and the model's own routing:

```
bash tools/ds4/memguard.sh 110 82 -- ./build-hip-fused/ds41_generate \
  -m "/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf" \
  --ids-file local/prompts/circles.i32 -n 64 --temp 0 --backend cuda --experts gpu \
  --arena-gib 90 --ctx 8192 --kv-q8 --threads 24 --rank local/rank-q2.bin \
  --engram models/engram/layers14:models/engram/layers1 --route-bias 0 --route-bias-file 0 --arena-skip-vram
```

That is the 4.53 tok/s, all-nine-circles run. 75 GiB with `memguard.sh 80 82` is the safer arena (3.49 tok/s with
`--arena-skip-vram`, 3.13 without); the 90 GiB arena leaves ~20 GiB of RAM free (MemAvailable 21 GiB during the run)
on this 122 GiB machine, which is why
it needs the bigger `memguard` cap. Stop the 8150 chat server first (`fuser -k 8150/tcp`): its 75 GiB arena is the
same RAM, and `./run-v41-gsq.sh` brings it back. The chat configs are `local/ds41-local-q2.json` (default bias, no
drafter) and `local/ds41-local-q2-exact.json` (exact routing); the GSQ reference without DSpark is
`local/ds41-local-gsq-nodspark.json`.

## Rebuilding the Q2 rank

```
# 1. the general corpus (8,154 tokens): the same command as the calibration run in "The Q2's own rank.bin"
STRATA_FORCE_IDS=local/calib-q2/circles-answer.ids ./build-hip-fused/ds41_generate ... \
  --ids-file local/prompts/circles.i32 -n 64 ... --route-bias 0 --route-bias-file 0 --dump-routes local/calib-q2/stats-circles
# 2. merge the stats dirs and rank
python3 local/calib-q2/merge_stats.py --out local/calib-q2/stats-merged local/calib-q2/stats-corpus local/calib-q2/stats-circles
python3 tools/v41/engine/v41_rank.py --stats local/calib-q2/stats-merged --out local/rank-q2.bin
```

## Do not waste time here

`tools/ds4/` is a **different** engine: it reads the older `deepseek4.*` metadata keys (V4 Flash) and cannot load this
model ("missing deepseek4.embedding_length"). The model belongs to `tools/v41` (`deepseek41.*`).

## Remaining work

1. ~~A Q2 `rank.bin`~~ done (2026-10-17): 0.97 -> 3.13 tok/s with the rank alone, 3.49 with `--arena-skip-vram` at
   75 GiB, 4.53 at 90 GiB, all-nine-circles correct. What is left of the gap to 9.01: the file tier is still 3.1% of
   the decode's lookups, and the measured non-file cost alone is 0.127 s/token (~7.9 tok/s ceiling). More resident
   bytes (the expert set is 142.4 GiB) or a file tier faster than ~0.93 GB/s of serial 9.5 MiB preads are the levers.
2. A V4.1 drafter matched to these weights, or a decision to keep the GSQ MTP (it drafts, but I/O-bound decode loses
   by it).
3. The `STRATA_RB_MARGIN` path: it is not equivalent to exact routing in this build, and the README's acceptance
   claim should be re-measured with `--route-bias 0` before being relied on.
