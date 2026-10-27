# DeepSeek-V4.1-Flash-1017 with exact routing on this box

The model is DeepSeek-V4.1-Flash-1017 as the
[ISTA-DASLab GSQ-RCO IQ2_XXS pack](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), run with
the model's own routing (`--route-bias 0 --route-bias-file 0`): every expert the router picks is computed, wherever it
is. The box: two RX 7900 XTX (`gfx1100`, 24 GB; card 1 behind a PCIe 4.0 x2 chipset link), Ryzen 9 9900X (12 cores,
AVX-512), 122 GiB RAM, the pack on a PCIe 5.0 NVMe (ext4), ROCm 7.14. Every number below was measured on this box with
this pack; one-shot runs are greedy and, unless they say otherwise, forced on one reference token stream
(`STRATA_FORCE_IDS`, `STRATA_DEC_MARGIN=1` counting the positions whose argmax differs) so that the arms of a
comparison decode the same text.

## The pack

Four shards, 138.2 GB: per layer, gate/up IQ1_M, IQ2_XXS or IQ2_XS and down IQ1_M, IQ2_XXS or IQ2_XS - eight
(gate, down) classes over the 40 layers (14 all IQ1_M, 9 IQ2_XXS/IQ1_M, 5 IQ2_XS/IQ1_M, 5 IQ2_XXS/IQ2_XXS,
3 IQ2_XXS/IQ2_XS, 2 IQ1_M/IQ2_XXS, 1 IQ2_XS/IQ2_XS, 1 IQ2_XS/IQ2_XXS); experts 132.1 GB, a mean expert blob of
8,598,528 bytes; the dense half 5.2 GB, Q5_K except `attn_q_a` and `attn_kv` (Q8_0) and the BF16 router, indexer and
Engram q/k.

Its GGUF carries llama.cpp metadata, not the `deepseek41.*` keys this engine reads. `tools/v41/ds41_pack_meta.py`
writes them into a 511 KB overlay (`local/rco-iq2xxs-meta/meta.gguf`, read through `DS41_META`): geometry and the
config JSON from `local/rco-iq2xxs-meta/geometry-keys.json` (68 keys of the checkpoint, 8 KB, in the repo), compress
ratios and kv / index source layers derived from which compressor tensors the pack holds, and the Engram token map,
primes and multipliers read from the pack's own I32/I64 tensors (129,280 map ids; the primes must tile the two Engram
tables' 384,006,168 and 384,016,682 rows exactly, so a wrong read fails at once). Rebuilt from the JSON alone it is
byte-identical to the one in use. In the engine: expert blobs are read from whichever shard holds their stacked
tensor (one buffered and one `O_DIRECT` descriptor per shard); the Engram tensors `engram_q` / `engram_k` /
`engram_wkv` are stored BF16 and Q5_K and re-typed to F32 at load (on the CPU path too: ggml-cpu has no f32 x bf16
binary op); `attn_output_a` is stored flattened; the embedding rows are Q5_K and go through the type traits; the VL
router bias `exp_probs_b_vl.bias` is only required of a pack that has it, and this one has none, so images are off.
`test_ds41_geometry <shard 1>` passes with the overlay.

## Where an expert is computed

| tier | what | experts |
| --- | --- | ---: |
| card 0's VRAM | the ranked head, then an LRU cache with background promotion (`STRATA_VRAM_BG`) | ~2,100-2,200 |
| card 1's VRAM | `--peer-device 1`: a 22 GiB store (`STRATA_PEER_GIB=22`) of ranked experts the arena does not hold (`STRATA_PEER_DISJOINT=1`), adaptive (`STRATA_PEER_ADAPT=1`); a miss it holds is computed there and only n_embd floats cross the link | ~2,750 |
| CPU | the RAM arena: `--arena-gib 80` of pinned RAM with the ranking past the VRAM head (`--arena-skip-vram`); a file-tier read takes a slot (`--arena-adapt`), an expert card 1 holds first (`STRATA_ARENA_EVICT_PEER=1`); 12 threads compute those experts' rows (`--threads 12`, the physical cores) | ~10,000 |
| NVMe | everything else, read when routed | ~3,100-3,400 |

The order the tiers are filled in is the rank (`local/rank.bin`; `local/update-rank.sh` rebuilds it, below).

## The configuration (`local/ds41-local.json`)

Flags: `--backend cuda --experts gpu --threads 12 --ctx 262144 --kv-q8 --prompt-batch 8192 --arena-gib 80
--arena-skip-vram --arena-adapt --vram-margin 0.5 --peer-device 1 --pf-b 0 --pcie 0 --route-bias 0
--route-bias-file 0`, the Engram tables, the rank, `--dump-routes local/routes-rco` and the conversation pool and
disk store (below). `--pf-b 0` turns the predicted-expert prefetch off and `--pcie 0` the PCIe-staged compute of
misses: a miss outside the cards is computed by the CPU from the arena. `--vram-margin 0.5` leaves 0.5 GiB of card 0
free beside the expert cache. `--kv-q8` keeps the compressed KV and the indexer keys in q8_0. Prompts of more than 2560
tokens skip layers 21-39 for all but their last 2560 tokens (enc-skip, on by default): those layers see earlier
tokens only through layer 20's compressed KV and index keys, and the last row is byte-identical to the full pass.

Environment: `STRATA_DENSE_REQUANT=q6_K` (the 84 Q8_0 dense matrices to q6_K, below), `STRATA_DENSE_NOCONT=1` (no
`ggml_cont` of an already contiguous tensor), `STRATA_DENSE_ALIAS=1` (results written straight into the persistent
state tensors), `STRATA_SUM_ZEROCOPY=1` (the weighted sum of a layer's experts reads the CPU's and card 1's rows from
their pinned host buffers), `STRATA_PRESTAGE_GIB=1.5` / `STRATA_PRESTAGE_MIN=512` (a device region the prompt path
stages the next layer's experts into), `STRATA_VRAM_BG=1`, `STRATA_PF_NO_DUMMY=1`, `STRATA_NO_LARGEPAGES=1`, the
peer switches above, `STRATA_SERVE_STEP_MAX=96` (up to 96 new prompt tokens run as decode passes, more as a batch),
`STRATA_SERVE_PASS_BATCH=1`, `STRATA_SERVE_PASS_BIAS=0`.

## Speed (2026-10-26)

One-shot, the configuration's flags, power-tune on (`local/power-tune.sh`: NVMe APST and CPU C3 off, `performance`
governor), 3 runs each (`local/bench-v41/ab-rco-requant.sh`, `ab-rco-levers.sh`; `local/sweep-rco-rq`,
`local/sweep-rco-lev`):

| | tok/s |
| --- | ---: |
| decode, Promessi Sposi (19-token prompt, 300 tokens) | 23.2-23.7 |
| decode at 17K context (128 tokens; one-shot with the server's elastic state columns, `STRATA_ELASTIC_COLS=1`) | 18.4-19.2 |
| prompt from scratch, 8,192 tokens | 371-414 |
| prompt from scratch, 17,349 tokens | 433-459 |
| prompt from scratch, 25,500 tokens | 458-484 |

A from-scratch 123,416-token prompt read at 483-493 tok/s (`local/bench-v41/repro-mmq-crash.py`, 2026-10-27).
Through the server, in a coding-agent session at 120-190K context: 16-20 tok/s decode, a turn's new tokens (tens to
~1,500) resumed on the saved conversation in 0.6-10.5 s (`local/serve-engine.log`, 2026-10-27).

## Quality (2026-10-26)

Lucebox's held-out set (8 x 1024 tokens; `local/eval-lucebox/run-eval.sh`, one card, arena 75 GiB, ctx 4096,
`--kv-q8`, 12 threads; `--against` scores one run's logits against another's):

| run | ppl | own top-1 | against | KL | same top-1 |
| --- | ---: | ---: | --- | ---: | ---: |
| the pack's iq4_nl Engram table (`rco-iq2xxs-iq4eng`) | 10.334 | 50.6% | | | |
| the checkpoint's FP8 Engram rows (`rco-iq2xxs-fp8eng`) | 10.369 | 50.7% | iq4_nl | 0.026 | 93.6% |
| FP8 rows, repeated (`rco-iq2xxs-fp8eng-rep`) | 10.369 | 50.7% | FP8 | 0.000 | 100% |
| FP8 rows + `STRATA_DENSE_REQUANT=q6_K` (`rco-iq2xxs-fp8eng-q6k`) | 10.372 | 50.7% | FP8 | 0.018 | 94.7% |

The one-card eval is deterministic (the repeat is bit-identical), so the two small KLs are real; neither moves the
ppl. The FP8 1017 checkpoint itself reads ppl 8.56 on this set.

## The q6_K dense requant

On this pack it re-types the 84 Q8_0 matrices (`attn_q_a`, `attn_kv`; 0.39 -> 0.30 GB). Speed, arms alternated:

| | q6_K | the pack's own types |
| --- | --- | --- |
| Promessi Sposi | 20.99 / 22.97 / 23.20 | 23.61 / 23.26 / 23.30 |
| 17K context | 17.66 / 18.30 / 18.32 | 15.11 / 17.88 / 18.30 / 18.33 |
| prompt 8,192 | 379.1 / 392.3 / 371.2 | 413.7 / 384.7 / 378.6 |
| prompt 17,349 | 436.1 / 433.4 / 450.0 | 435.3 / 459.2 / 447.5 |
| prompt 25,500 | 482.2 / 483.5 / 470.5 | 460.3 / 458.3 / 477.3 |

The same within the spread, ppl +0.003: the configuration keeps it (0.09 GB of VRAM).

## Where a token goes (2026-10-26)

`STRATA_DEC_PROF=1` / `DS4_TIER_PROF=1` / `STRATA_POOL_PROF=1`, ms per token (`local/sweep-prof-rco/`):

| | begin | attention + router | tier | finish | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promessi Sposi | 1.0 | 24.8 | 14.7 | 1.4 | 23.6 |
| 17K context | 0.5 | 27.8 | 27.5 | 1.6 | 18.4 |

In the tier: the CPU pool 7.4 ms (35 experts a token) / 16.8 ms (74), reading at 47 GB/s - the RAM's limit with
power-tune on (`local/ram-bench.sh`: 61-62 GB/s streaming); waiting for card 1's rows and the sum 6.2 / 7.1 ms. Card
0's kernel trace (`rocprofv3`, 64 Promessi Sposi tokens): 3,378 kernels and 19.9 ms busy a token (`mul_mat_vec_q` 7.2
ms, the expert kernels 3.6 ms). Of its idle time, 8.6 ms a token is the gap between its own expert kernels and the
sum (waiting for the CPU's rows) and ~6 ms is spread over the 40 per-layer host round trips (router readback, the
tier's decision, the next graph). Neither is a kernel cost: fusing the dense half's small kernels cannot remove them.
Routing on the card would remove a round trip only where all six of a layer's experts are on the card - ~35% of
layers on short prompts, ~6% at 17K (from the measured hit rates) - so it was not built.

Card 1's link: a hand-off moves a few KB each way, so it is latency-bound, not bandwidth-bound; a wider link would
not change it.

## Levers measured and not taken (2026-10-26)

`local/bench-v41/ab-rco-levers.sh`, 3 rounds alternated, tok/s (VRAM hits):

| | 17K, 128 tokens | Promessi Sposi, 300 tokens |
| --- | --- | --- |
| the configuration | 17.67 / 16.69 / 18.50 (62.7%) | 23.69 / 23.31 / 23.68 (84.0%) |
| the configuration again (a third arm; see below) | 18.44 / 18.33 / 18.28 | 23.55 / 23.65 / 23.74 |
| `--rank local/rank-rco-calib.bin` (the pack's byte-correct order over the calibration corpus) | 18.92 / 18.45 / 18.39 (62.6-64.9%) | 23.88 / 22.97 / 23.79 (84.2%) |

The rank is inside the spread, with the configuration's argmax counts. The third arm was meant to put the pack's
IQ2_XXS / IQ2_XS / IQ1_M down rows on the AMD expert layouts, but those layouts are not compiled for gfx1100
(`STRATA_EXP_LAYOUTS` is 0 there), so it ran the default kernels: it is a repeat of the configuration, and the switch
was removed. A release of
ggml's device memory pool after the prompt gave back 0.22 GiB and was removed. A one-shot run keeps the prompt's 8192
state columns (~2.5 GiB) after the prompt where the server gives them back (`elastic_cols`); `STRATA_ELASTIC_COLS=1`
lets a one-shot run do as the server does (2,176 experts on card 0 after the 17K prompt instead of 1,594).

## The Engram tables, in either form

The Engram rows come as the pack's own iq4_nl GGUF (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, 110.6 GB, one tensor a
layer, 144 bytes a row read in one pread) or as the checkpoint's FP8 rows (two safetensors a layer, F8_E4M3 weights
and F8_E8M0 scales, 256 + 8 bytes a row in two preads; 203 GB in `models/engram-1017`). `--engram` takes either.
iq4_nl is not linear: `ds41_dequant_iq4_nl_row` is ggml's `block_iq4_nl` arithmetic, gated by
`tools/v41/engine/test_ds41_engram.cpp` (1,028,096 of 1,028,096 values bit-identical to ggml's `to_float`).

Rows per second of the batched reader (`build-hip-fused/bench_ds41_engram`, 4096-token sequence, 16 threads, internal
NVMe; cold = `drop_cache` between reps):

| pass | FP8 warm | iq4_nl warm | FP8 cold | iq4_nl cold |
|---|---:|---:|---:|---:|
| 1 token | 592,878 | 2,466,598 | 577,617 | 2,671,119 |
| 512 tokens | 3,014,128 | 7,001,799 | 1,962,231 | 7,464,885 |
| 4096 tokens | 5,630,238 | 5,951,369 | 5,062,294 | 7,150,841 |

Quality: the Lucebox table above. The configuration names the FP8 rows; where they are missing the engine uses the
iq4_nl table beside the pack and prints that it does. Two guards against a table cut short by a crash (a
`truncate()`-reserved file keeps its size and reads as zeros): the extractor writes `.part`, renames after `fsync` and
samples 256 rows, and the engine samples 512 rows at open (`STRATA_ENGRAM_NO_ZERO_CHECK=1` skips it).

## CPU expert rows: IQ1_M in AVX-512

`src/kernels/cpu/iq_avx512.cpp` computes IQ1_M rows (and IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S) for several
tokens at once: it decodes a block once and applies it to every token. Real rows of the pack, one core, ns per
row-token against ggml-cpu's dispatched dot (`local/bench-v41/fmt-cost/fmt_cost.cpp`):

| rows | ggml-cpu | kernel, 4 tokens | kernel, 8 tokens |
|---|---:|---:|---:|
| IQ1_M gate/up @5120 | 168.00 | 150.23 (1.12x) | 138.86 |
| IQ1_M down @2304 | 78.53 | 67.51 (1.16x) | 61.54 |
| IQ2_XXS gate/up @5120 | 83.25 | 38.28 (2.17x) | 38.32 |
| IQ2_XS gate/up @5120 | 198.60 | 89.97 (2.21x) | 93.06 |

IQ1_M rows are bit-exact against ggml's generic dot (680 / 680 at eight tokens). Below four tokens ggml-cpu is faster
(`native_gu_mt_min` = 4 for IQ1_M), and decode hands the CPU pool one token at a time, so the kernel serves only
multi-token groups; in decode it is not on the path (`STRATA_NO_IQ512=1` decodes the same).

## Conversations

Every conversation a request does not continue is kept in a 4 GiB pool of pinned RAM
(`--conversation-cache-mib 4096`) and on disk (`--conversation-disk local/conv-cache-rco --conversation-disk-gib 32`,
saved in the background after each answer); a request resumes the longest saved prefix it shares. Resuming is exact:
a resume a few tokens before a saved end (a client that re-renders the tail of the last turn) re-feeds from the even
position `ps = lcp & ~1` after `restore()`, only when `P - ps <= 16` (the window ring's reach), for the live and for a
parked conversation, and the Engram history of the first new positions is rewritten.
`tools/v41/engine/check_conv_reuse.py --scenario single` (a re-render 12 tokens back, resumed at 298 of 310) is
bit-exact against a fresh prefill on the CPU backend. A client that cuts the middle of a conversation is read again
from the cut: there is no state to resume from there. `STRATA_SERVE_LIVE_PARTIAL=0` turns the live-conversation
partial resume off.

## Fixed: the prompt MMQ read past its id buffer (2026-10-27)

The MMQ product reads up to J_best - 1 (<= 127) ids past a run's last row; the down product took them from a buffer
allocated to exactly n x top_k entries, so the last run of a batch read past it - a GPU page fault ("chunk sync: an
illegal memory access") on a 3,918-token batch over a 123K conversation. Reproduced at 8,000 + 3,918 and 123,416 +
3,918 tokens (`local/bench-v41/repro-mmq-crash.py`); with 128 ints of slack on those buffers both run through three
turns, at the same prompt speed.

## The expert rank

`local/update-rank.sh` adds the chat's own routing counts (`local/routes-rco`, written by `--dump-routes`) to the
calibration corpus's (`local/calib/stats-merged`), ranks the experts per byte with the pack's formats
(`local/rco-iq2xxs-map.json`), measures the candidate against the rank in use on a code review and a documentation
prompt (`local/make-dev-prompts.py`: `dev-code`, 16.7K tokens of C++, and `dev-docs`, 10.8K tokens of docs) and
applies it only if it is at least 1% faster with neither prompt more than 2% slower.

Run on 2026-10-27 (the chat's 41.4M routed selections against the corpus's 2.0M, 3 runs a side, power-tune on;
`local/sweep-rank`): rank in use 19.15 / 19.77 / 19.56 on `dev-code` and 16.79 / 17.99 / 16.42 on `dev-docs`, the
candidate 19.68 / 17.20 / 17.36 and 19.14 / 18.00 / 18.92 - code -7%, docs +9%: kept.

## Reproduce

```
./build.sh
./run.sh                                         # the server, local/ds41-local.json
bash local/bench-v41/ab-rco-requant.sh           # decode and prompt speed (server stopped)
bash local/eval-lucebox/run-eval.sh TAG          # quality; MODEL / RANK / ENGRAM as in the table above
```

## Gate

`test_ds41_geometry <shard 1>` with the overlay, `test_ds41_engram`, `test_ds41_dense <shard 1> --engram <tables>` on
the CPU, `bench_ds41_engram ... --check 200`, the server tests (`.venv/bin/python -m unittest discover -s serve -p
"test_*.py" -t .`) and `check_conv_reuse.py` for anything that touches conversation reuse.
