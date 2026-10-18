# V4.1 GSQ: exact routing at 10-14 tok/s (2026-10-17)

The route bias made the model fast and wrong (`HANDOVER.md` §3). This note records how the GSQ was made fast **with
the model's own routing** (`--route-bias 0 --route-bias-file 0`) on this box: 2x RX 7900 XTX (card 1 behind a PCIe
4.0 x2 bridge), Ryzen 9 9900X (12 cores / 24 threads), 122 GiB RAM, the GSQ on the Lexar NM1090 PRO (PCIe 5.0 x4).
Every number below was measured on this box; one-shot runs are greedy, 300 tokens, `--ctx 8192`, arena 90 GiB.

## Result

| configuration (exact routing throughout) | Divina Commedia | Promessi Sposi | Python (English) |
|---|---:|---:|---:|
| before: GSQ `rank.bin`, 24 threads | 2.83-3.36 | 3.37 | 3.93 |
| + `local/rank-gsq-mixed.bin` | 5.86 | 4.96 | 4.50 |
| + `--arena-adapt` | - | 5.88 | 5.21 |
| + `STRATA_PF_NO_DUMMY=1` | - | 6.72 | - |
| + `--peer-device 1` (miss mode, below) | - | 7.29 | 5.84 |
| + `--threads 12` | - | 10.39 | - |
| + `--vram-margin 0.5` (= `local/ds41-local-fast-exact.json`) | 10.05 / **11.16** free | 10.68 | **9.85** |
| + DSpark, peer in the verify path, `STRATA_TIER_BATCH_PCIE=0` | - | 11.06 | **14.03** |
| `--vram-margin 0.5` config + `--pf-b 0`, no DSpark (= `local/ds41-local.json` now) | **13.07** | **12.98** | 10.32 |

Through the server (`./run-v41-gsq.sh`, ctx 262144, arena 75 GiB, temperature 0), with `--pf-b 0`: Divina Commedia
12.10 cold / 14.51 repeated, the nine circles 13.70 (TTFT 1.2 s), Promessi Sposi 13.40, the Python prompt 10.94, and
with thinking on (temperature 0.8) 12.95 tok/s; before `--pf-b 0`: 11.84 / 14.05 / 13.64, thinking 11.27. The chat with exact routing ran at ~2.7 tok/s
before. Answers checked by hand: correct dates, structure, the nine circles.

Token agreement against the exact reference (the same stream forced with `STRATA_FORCE_IDS`, `STRATA_DEC_MARGIN=1`
counting the positions whose argmax differs): every lossless change above stays at **2-7 of 300**, all near-ties
(logit gaps <= 0.27), the same positions as an identical exact run repeated (2 of 300: the CPU and the cards round a
row differently, and which computes it depends on the cache state). The route bias at 0.5/0.3 differs at **30 of
300**, with gaps up to 18 logits.

## What each step is

1. **The rank was the main defect, not the routing.** The GSQ's `rank.bin` comes from an agentic/code corpus: on the
   Divina Commedia answer's own routing it covers **11.5%** of the routed tokens in the first 14 GiB (the VRAM seed)
   and 78.9% in 104 GiB; the Q2's counts (`local/calib-q2/stats-merged`, mixed Italian/English/code, same base
   router) cover 51.6% and 97.5%. `local/rank-gsq-mixed.bin` is those counts ranked with the GSQ's real per-expert
   sizes (`local/gsq-alloc-map.json`, from `V41-GSQ.gguf.experts.json`: type 11 = Q3_K, 42 = Q2_0, 143 = PTQ1_0):
   `python3 tools/v41/engine/v41_rank.py --stats local/calib-q2/stats-merged --map local/gsq-alloc-map.json --out
   local/rank-gsq-mixed.bin`. The Divina Commedia answer is partly in that corpus; the Promessi Sposi and the Python
   prompt are not, and gain +47% and +15%.
2. **`--arena-adapt`** (existing): a file-tier read takes the arena slot of the layer's least recently used expert.
3. **`STRATA_PF_NO_DUMMY=1`** (existing): the prefetch DMA'd a zero blob for every predicted expert outside the
   arena (6,070 of 17,102 prefetches in the exact run, 9.7 MB each) to keep an old simulator's timing.
4. **The second card in miss mode** (`--peer-device 1`, new): the saved peer patch (`local/peer-tier-attempt.patch`)
   re-applied, with one change of design. The old peer only moved the *kernel* of rows card 0 would compute anyway
   and kept their DMA, to stay bit-identical; under exact routing the misses are the cost (~2 ms each on the
   critical path), so now a miss whose expert the peer's store holds is computed on card 1 and **no blob moves** -
   only n_embd floats each way over the x2 link. The store is the 22 GiB (`STRATA_PEER_GIB=22`, ~2,390 experts) of
   ranked experts after card 0's VRAM head; with `--arena-adapt` its arena copies are not refreshed, so the arena
   swaps them out. Wired in the single-token decode (`gpu_run`), its prefetch, and the DSpark verify pass
   (`gpu_run_n_ex`, `prefetch_multi`); not in the prompt path. `STRATA_PEER_HOT=1` = the original hot split.
5. **`--threads 12`**: the 12 physical cores. With 24 SMT threads the pool's threads contend with the thread that
   drives the cards: 7.29 -> 10.39 tok/s, and the prompt 189.8 -> 246.4 tok/s (5,783 tokens).
6. **`--vram-margin 0.5`**: 0.5 GiB more expert slots on card 0.
7. **`--pf-b 0`** (no predicted prefetch): with the peer and 12 threads the prefetch's DMAs cost more than its hits
   save - Promessi Sposi 10.68 -> 12.98, Divina Commedia 10.05 -> 13.07, Python 9.85 -> 10.32. Not with DSpark: there
   the prefetch is what admits experts into VRAM in the verify pass (hits 74% -> 21-53% without it; Sposi 11.06 -> 9.60).

## What does not work

- **The near-tie bias** (`STRATA_RB_MARGIN=d`): d = 0.02 changes 5 of 300 positions and is no faster (45.7% VRAM
  hits, the same as exact); d = 0.05 changes 8 and is no faster. The bias is fast only because it rewrites many
  choices, so restricting it to near-ties leaves nothing to gain. `STRATA_RB_MARGIN=1` (the old chat config) is
  wider than the scores' spread: it is the flat bias.
- **File-tier-only penalty with the top 4 protected** (`STRATA_RB_PROTECT=4 --route-bias-file 0.3`): 6 of 300
  changed, no faster.
- **DSpark on Italian prose**: the drafter accepts ~1.0 draft per pass (code: 2.5-2.9), so the verify width is mostly
  wasted; 11.06 vs 10.68 tok/s single-step.

## The dense half is GPU-bound, not sync-bound (measured)

An opt-in asynchronous dense path (`STRATA_DENSE_ASYNC=1`: `graph_compute_async`, the readbacks queued behind the
graphs with one synchronize, and the tier handing over with an event that ggml's stream waits on instead of a
stream sync) changed nothing: Sposi 12.98 sync vs 12.87 async at `--pf-b 0`, and the all-VRAM ceiling 20.54 vs 19.69.
It was removed; the code is `local/dense-async-attempt.patch`. `rocprofv3 --kernel-trace` on the all-VRAM ceiling
(64 tokens) says why: **32.4 ms of kernels per token in 4,596 launches** - `mul_mat_vec_q` 12.1 ms (335 launches, the
~7.3 GB of Q8_0 dense weights at ~600 GB/s), the expert kernels 5.6 ms, `mul_mat_vec_f` 3.2 ms (197), copies 1.5 ms
(815), rms_norm 1.6 ms (306), argsort 1.2 ms, the rest ~7 ms. Against the 48.7 ms an unprofiled token takes, ~16 ms is
the gaps between ~4,600 launches (~3.5 us each on HIP), which the host cannot hide by running ahead: it launches one
kernel at a time either way. Removing them needs ggml's HIP graphs to replay instead of re-capturing every token
(`AMD_HIP_NOTES.md`, "ggml's HIP graph capture costs more than it saves here").

## Session 2 (2026-10-17 night): the miss path and the prompt passes

Three opt-in changes, now in the chat config (`local/ds41-local.json`); one-shot numbers are greedy 300-token runs on
the forced exact stream, server numbers are `./run-v41-gsq.sh` at temperature 0.

1. **The peer store follows the working set** (`STRATA_PEER_ADAPT=1`, `STRATA_PEER_ADAPT_INFLIGHT=8`,
   `STRATA_PEER_ADAPT_WIN=16`): a miss that recurs within 16 calls of its layer and is in the arena is copied to
   card 1 on a separate stream (after the peer's last launch), into the least recently used peer slot of the same
   format class; the victim is unmapped at once and the newcomer published when its copy's event fires (a layer's
   copies are waited for at its next call, because that call may hand the source arena slot to someone else).
   1,360-1,870 experts admitted per 300 tokens, never a wait. A/B, alternated: Promessi Sposi 13.16 -> 13.55 tok/s
   (mean of 2 vs 3 runs), Python 10.92 -> 11.39 (one adaptive run at 8.89 left out as an outlier).
2. **`--pcie 0`**: the arena misses on the CPU pool (12 threads) instead of DMA'd to card 0 - with the peer and
   12 threads the CPU is the cheaper path for one token: Sposi 13.55 -> 14.63, Python 11.39 -> 12.35 (`--pcie 0.25`:
   13.44 / 12.22). It made the server's short prompts slower (they ran token by token, each recomputing its misses on
   the CPU: a 40-token prompt 2.7 -> 6.1 s), which 3. fixes.
3. **`STRATA_SERVE_PASS_BATCH=1`**: the server's prompt passes (prompts up to 768 tokens) call the tier with
   `run_multi_to_dev` per 8 tokens, as the DSpark verify pass does - each distinct expert is served once for every token
   routing it - instead of `run_to_dev` per token.

| server, temperature 0 | before session 2 (graphs on) | after 1+2 | after 1+2+3 (= now) |
|---|---:|---:|---:|
| Divina Commedia cold / repeated | 11.71 / 15.18 | 13.32 / 15.46 | 12.77 / **16.07** |
| nine circles: TTFT / decode | 1.26 s / 14.04 | 2.40 s / 14.82 | **0.94 s / 15.11** |
| Promessi Sposi: TTFT / decode | 0.89 s / 13.69 | 1.39 s / 13.20 | **0.72 s / 15.02** |
| Python: TTFT / decode | 2.33 s / 10.80 | 4.86 s / 10.75 | **2.00 s / 11.81** |
| thinking (40-token prompt): TTFT / decode | 2.73 s / 13.50 | 6.12 s / 13.50 | 3.07 s / 12.86 |

Answers checked: the nine circles, Beatrice (dead in 1290, guide from the Earthly Paradise). Gate (default flags,
`code_review.i32`): tokens and tier counters identical to `local/gate-before.txt`.

4. **`STRATA_VRAM_BG=1`**: with `--pcie 0` card 0's cache stopped changing (its promotions rode on the PCIe misses'
   DMA): ~99 VRAM hits per token in the chat. Now one recurring miss per call is promoted into card 0 after the call,
   off the critical path: the copies go on the tier's stream behind the call's kernels (a later kernel reading the slot
   is ordered after them), the mapping is published at once, and a victim without an arena copy is demoted into the
   slot the newcomer leaves. One-shot: Sposi 14.63 -> **15.74**, Python 12.35 -> **13.53** (`=2`: 10.48 / 13.35).
   Server: ~167 VRAM hits per token; Divina Commedia 13.48 cold / **17.40** repeated, nine circles 15.82, Promessi Sposi
   15.76, Python 10.83, thinking 14.37, two new questions 16.01 / 16.03 tok/s.

5. **DSpark on this configuration** (one-shot, `STRATA_TIER_BATCH_PCIE=0`, the peer in the verify pass): Python at depth 4
   **18.58** tok/s (3.7 tokens/pass) against 13.53 single-step, but Promessi Sposi 10.55 (depth 4) / 12.78 (depth 2)
   against 15.74 - the drafter accepts ~1 draft per pass on Italian prose. Promotions inside the verify pass
   (`STRATA_VRAM_BG_PASS=1/2`) gave Python 16.60 / 12.10 (depth 4), 15.27 / 15.38 (depth 2) and prose 11.84 / 11.99,
   10.70 / 13.73: no clear gain inside the spread, so they are a separate opt-in, off. The chat stays single-step.
6. **Where a token goes at the ceiling** (`rocprofv3`, all-VRAM, graphs on): 28.5 ms of kernels, then **3,751
   inter-kernel gaps of ~3.6 us = 13.5 ms** (the dispatch cost of ~3,500 kernels a token, inside the graphs too) and
   ~120 host round trips of ~40 us = ~7 ms. The next large lever is the kernel COUNT of the dense layer (~88 per layer).

7. **Fewer kernels in the dense layer + a stable machine** (2026-10-18). With the CPU at `performance` and the cards'
   clocks raised (the user's LACT settings), the all-VRAM ceiling stopped swinging (17.97-22.58 -> 22.3-22.5 tok/s).
   `STRATA_DENSE_NOCONT=1` skips 21 `ggml_cont` calls whose input is already contiguous (a pure copy each; tokens
   identical): ceiling 22.3-24.2 -> 24.2-24.6. `STRATA_DENSE_ALIAS=1` computes five layer outputs (hap, comb_f,
   shexp, o_fn, o_wts) straight into the persistent state - the op becomes a view of its destination, as ggml's
   in-place ops are - instead of copying them at the end: prose steady state 16.61/17.18 -> 17.26/17.39, Python
   12.73/13.31 -> 13.30/13.53, the prose argmax-change positions identical. Measured along the way: ggml's own
   fusion off (`GGML_CUDA_DISABLE_FUSION=1`) is not faster (ceiling 20.31/22.29 vs 22.19/22.58 on), pinning the
   engine to cores is much slower (5.8-9.0 vs 11-16 tok/s), and the pool's spin time (`STRATA_POOL_SPIN_US`) changes
   nothing measurable. **The second card is worth it**: same configuration without `--peer-device 1`, prose
   14.08/14.16 vs 15.83/16.24 tok/s (+13%), Python 11.97/12.18 vs 12.73/13.31 (+8%); `local/ds41-local-1gpu.json`
   is the one-card config. A fresh process pays a warm-up (the first ~64 tokens at up to 116 ms of attn_router), so
   one-shot numbers understate a running server; `local/bench-v41/steady.py` prints the steady state.
   Server now: Divina Commedia 15.37 cold / **18.95** repeated, nine circles 16.02, Promessi Sposi 16.35, Python 12.27,
   thinking 14.75, two new questions 17.08 / 17.19 tok/s.

**The rank from the chat's own routing.** The chat config also runs `--dump-routes local/routes-chat`: in `--serve`
the engine loads the counts already there at start and writes them after every request, so they accumulate across
restarts (git-ignored: they are this user's routing). `local/rebuild-rank.sh [chat-weight]` adds them to the
calibration counts and writes `local/rank-gsq-chat.bin`; point `--rank` at it and restart to use it.

## ggml's HIP graphs pay now (`GGML_CUDA_DISABLE_GRAPHS` removed from the chat config)

`AMD_HIP_NOTES.md` measured the graphs as a loss (biased routing, an older state: 20 captures per token). In this
configuration they replay: `rocprofv3 --hip-runtime-trace` on the all-VRAM decode counts **69 `hipGraphLaunch` and
1.3 captures per token** (613 direct launches are the tier's). Measured with and without
`GGML_CUDA_DISABLE_GRAPHS=1`, same binary:

| | graphs off | graphs on |
|---|---:|---:|
| one-shot Promessi Sposi (`--pf-b 0`) | 12.15 | **12.98** |
| one-shot all-VRAM ceiling | 18.68 | **20.54** |
| server, 7,491-token prompt: TTFT / decode after it | 37.5 s / 10.41 | **30.5 s / 12.02** |
| server, Divina Commedia cold / repeated | 12.10 / 14.51 | 11.71 / **15.18** |
| server, nine circles / Promessi Sposi / Python / thinking | 13.70 / 13.40 / 10.94 / 12.95 | **14.04 / 13.69** / 10.80 / **13.50** |

So the graph replay the dense half needs already works; what is left between the kernels at the ceiling is the
per-layer hand-off to the expert tier (the router ids to the host, the tier's plan and launches, the sum), which a
graph cannot hold because the plan is made on the host from the ids (`V41_DECODE_OVERLAP_PLAN.md` §5: the miss service
is host work keyed on the ids, so the layers cannot run ahead).

## Where the time goes now

`STRATA_DEC_PROF=1`, 300-token Divina Commedia, the configuration above: begin 3.5, attn_router 48.7, tier 31.0,
finish 5.1 ms/token (11.2 tok/s). With every expert in VRAM (`--route-bias 20`, quality meaningless, speed only):
begin 4.0, attn_router 30.4, tier 7.9, finish 4.7 = **20.7 tok/s**, so the remaining misses cost ~42 ms/token and the
dense half alone caps single-token decode near 21 tok/s at 8K context. DSpark verify pass (5.5 tokens): dense 55-78
ms, tier 105-195 ms.

Prompt (5,783 tokens, 12 threads): 246.4 tok/s, split dense 7.2 s / experts 16.0 s (file wait 5.1 s).

## Reproduce

```
./run-v41-gsq.sh        # local/ds41-local.json = local/ds41-local-fast-exact.json (the old one: .pre-fast-exact)
```

One-shot (stop the chat first, `fuser -k 8150/tcp`):

```
bash tools/ds4/memguard.sh 110 82 -- env STRATA_PF_NO_DUMMY=1 STRATA_PEER_GIB=22 ./build-hip-fused/ds41_generate \
  -m models/V41-GSQ/V41-GSQ.gguf --ids-file local/prompts/divina-breve.i32 -n 300 --temp 0 --backend cuda \
  --experts gpu --arena-gib 90 --ctx 8192 --kv-q8 --threads 12 --rank local/rank-gsq-mixed.bin \
  --engram models/engram/layers14:models/engram/layers1 --arena-skip-vram --route-bias 0 --route-bias-file 0 \
  --arena-adapt --vram-margin 0.5 --peer-device 1
```

All runs and logs: `local/sweep-gsq/` (`summary.txt` has one line per run). The server needs `HIP_VISIBLE_DEVICES=0,1`
in the config's `env` (the server otherwise exposes only `"gpu"`).

## Gate

Default flags (no `--peer-device`), greedy 96 tokens on `local/prompts/code_review.i32`: tokens identical to
`local/gate-before.txt`, and the arena/file/VRAM counts (8229 / 7131 / 1622) and hit counters too.

## Next levers, by expected value

1. **The dense half's launch count** (4,596 kernels a token, ~16 ms of gaps at the ceiling): ggml graph replay
   instead of re-capture, or fewer, fused ops. Removing host syncs alone was measured and does not pay (above).
2. **An adaptive peer store**: card 1's store is static; admitting card 1's own misses over the x2 link (async, ~3 ms
   per blob) would let both cards follow the conversation's working set (one answer uses < 64 GiB of experts).
3. **The verify pass's tier** (105-195 ms for 5.5 tokens): union hit rate 60-74% against 88% single-step.
4. **A drafter that accepts Italian**: at ~1 draft per pass DSpark cannot pay on prose.
