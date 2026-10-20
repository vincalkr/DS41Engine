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

8. **The dense half in Q6_K at load** (`STRATA_DENSE_REQUANT=q6_K`, the existing switch; elsewhere measured at KL vs FP8 0.2281 -> 0.2306 and API top-1 85.87 -> 85.76 % with a 1,200-token greedy edit
   text identical to Q8_0). The file is untouched; 321 Q8_0 matrices are re-quantized in memory, 7.59 -> 5.86 GB, and
   card 0's expert slots grow 14.58 -> 16.19 GiB. This is a weight-precision change, not a routing one. One-shot:
   ceiling 22.46/22.53 -> **25.71/26.73** tok/s; on the forced exact stream 4 of 300 positions differ (Q8_0 itself: 6);
   the free Divina Commedia and nine-circles answers are correct. Server (now the chat default): Divina Commedia 16.52
   cold / 18.32 repeated, nine circles **18.63**, Promessi Sposi **18.45**, Python 12.63, thinking **16.35**, two new
   questions 18.19 / 18.45 tok/s; 7,491-token prompt TTFT 30.5 -> **26.75 s** (280 tok/s), decode after it 12.02 -> 14.14.

**Thinking budget: 192 -> 1024** (2026-10-18). The 192 came from the biased-routing state (`HANDOVER.md`: 384 leaked
the deliberation into the answer, 2048 gave no answer). With exact routing, `local/bench-v41/think-budget.py` ran four
prompts at budgets 192 / 512 / 1024 / 2048 (thinking on, temperature 0.8): all 16 `finish=stop`, no think tags or
loops in the answers. Checked by hand: the ordering puzzle right at every budget; Manzoni and the 1840-1842 edition
right at every budget, but at **192** (thinking cut by the budget) the first part is placed "mainly in Milan" - wrong;
from 512 up it is Lecco and the lake of Como, right; the four Divina Commedia answers right (34/33/33 cantos, the
guides, 9 circles / 7 terraces / 9 heavens + Empyrean); the interval-merge code passes 6/6 tests at every budget, but
the 2048 answer had one corrupted token (`return risultato talents`, a SyntaxError; 6/6 with it removed). Eight more
runs of the code prompt at budget 1024 (`code-corrupt.py`, repetition_penalty 1.1 and 1.0 alternated) passed 5/5
tests each, so that corruption is rare (1 in 12 code answers) and not tied to the penalty in this sample. Cost: the
facts question thought 2,167 tokens (133 s) at 2048, 1,156 (76 s) at 1024. The chat configs now set
`reasoning_budget_tokens` 1024; a request can still set its own.

**One card by default** (2026-10-18). On an agent-like session at 25-27K context the second card measured 14.31 /
15.16 tok/s against 15.51 / 13.47 on card 0 alone - no difference inside the spread - so `local/ds41-local.json` now
runs on card 0 only (`HIP_VISIBLE_DEVICES=0`, no `--peer-device`, no `STRATA_PEER_*`) and card 1 stays free. The
two-card chat is `./run-v41-gsq.sh --config local/ds41-local-2gpu.json`.

**The rank from the chat's own routing.** The chat config also runs `--dump-routes local/routes-chat`: in `--serve`
the engine loads the counts already there at start and writes them after every request, so they accumulate across
restarts (git-ignored: they are this user's routing). `local/rebuild-rank.sh [chat-weight]` adds them to the
calibration counts and writes `local/rank-gsq-chat.bin`; point `--rank` at it and restart to use it.

**The small-turn prefill and the conversation-adaptive arena** (2026-10-18). Four ds41 engine commits are taken in: 716cf901 (the span-index upload memo and the lazy
prestage below `STRATA_PRESTAGE_MIN` rows), 5429b5ad / cff4bb24 / 8d43fe1b (`STRATA_ARENA_REBAL`, the
conversation-adaptive arena, with its cancel fix). Measured with `local/bench-v41/ab-rebal.sh` on the agent replay
(`agent-replay.py`, 25,563-token first turn, then follow-ups of +19, +28, +304 and +16 tokens, 15 s between turns,
a fresh server per run, the chat config, one card); `rebal-summary.py` prints the table:

| build / setting | runs | 25K prompt | +304 turn | file-tier experts, +19 / +304 turn |
|---|---:|---:|---:|---:|
| before the commits | 4 | 68.3-70.0 s | 4.8-5.0 s | 261-278 / 1,814-1,904 |
| new, as committed (lazy below 4096 rows), arena off | 4 | 69.8-72.9 s | 4.0-4.3 s | 246-288 / 1,783-1,932 |
| new, arena on (`STRATA_ARENA_REBAL=1`) | 2 | 72.5-73.2 s | 4.0-4.9 s | 361-377 / 1,609-1,635 |
| new, lazy prestage off (`STRATA_PRESTAGE_MIN=0`) | 2 | 68.6-69.3 s | 4.9-5.1 s | 268-279 / 1,866-1,918 |
| **new, lazy below 512 rows** (`STRATA_PRESTAGE_MIN=512`, the chat now) | 2 | **69.5-69.6 s** | **4.1-4.4 s** | 270-271 / 1,850-1,857 |

Decode (400 tokens a turn, 9-16 tok/s) and the one-token "exact passes" follow-ups (1.3-3.4 s) do not differ
beyond the spread in any row. So:
- The lazy prestage is the small-turn gain (the 304-row turn's experts 3.8-4.0 -> 3.2-3.5 s, its dense 0.96 -> 0.78 s), but
  at the commits' 4,096-row threshold the 25K prompt's last 987-row batch also goes lazy and its 40 layers lose the DMA
  overlap: +2-3 s. At 512 rows the long prompt is back to the old time and the 304-row turn keeps the gain.
  `local/ds41-local*.json` set `STRATA_PRESTAGE_MIN=512`.
- The adaptive arena does not pay here and stays off (opt-in again, as it first was): with a 75 GiB arena the
  file tier is small, the swaps it plans from the 25K document push out experts the short chat follow-ups route
  (+19 turn: 361-377 file-tier experts against 246-288), and only the +304 turn reads less (-12%).
- Temperature 0 is not reproducible across runs of the same binary (base-r1 and base-r2 part after 113 characters):
  where an expert is served from (VRAM, prestage region, streamed) picks MMVQ or MMQ, and that placement depends on
  timing (`--arena-adapt`, the background promotions). A quality gate here has to be read by hand, not diffed.

Not taken: dee2af6c (prompt-lookup drafts up to 14 tokens, DSpark eviction during long prompts) needs DSpark, which
loses on Italian prose here; 60f2c020 (keep-cache) never engages with `--vram-lru`; f7c561b7 (`--tier-batch-pcie`)
is a flag for a setting the config already sets.

**DeepSeek-V4.1-Flash-1017 tried and dropped** (2026-10-18). DeepSeek's 2026-10-17 checkpoint has the same
config.json as V4.1-Flash (new weights, Engram tables included, same tokenizer: tokenizer.json and
tokenizer_config.json have the same sha256). The one 1017 quant in this engine's GGUF format was
[lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit) (148 GB, 2.03
bpw: 11,147 of 15,360 experts' gate/up are ternary PTQ1_0, against 7,152 in the GSQ; map from
`local/gguf-expert-map.py`). Fetched with `local/fetch-1017.sh` (pinned revisions, every file checked against its
Hugging Face sha256; the Engram straight from the official shards 47/48 with an index, no extraction). It loads and
runs unchanged: 15.4-19.1 tok/s in the chat (Divina Commedia 17.8, nine circles 18.9, Promessi Sposi 19.1, Python
15.4, thinking 16.3), with a rank from its own routing (`local/calib-1017/stats-corpus`; that routing correlates 0.99
per layer with the September model's on the same corpus). But its answers had facts wrong in 3 of 5 prompts
(`local/bench-v41/v1017-chat.txt`): the 6th circle given as the envious, Don Rodrigo "converting", Perpetua as Fra
Cristoforo's servant, a Python test whose expected values fail (`[(1, 11)]` for `[(1, 4), (4, 8), (8, 12)]`).

To rule out this engine's loading, Lucebox's held-out set (8 x 1024 tokens with the FP8 1017 teacher's top-256
log-probs, `local/eval-lucebox/`, `run-eval.sh` + `score.py`) was run through it:

| | ppl | top-1 vs teacher | KL top-256 |
|---|---:|---:|---:|
| FP8 1017 teacher (published) | 8.56 | 100 % | 0 |
| lxe's quant on his engine (published) | 11.66 | 71.51 % | 0.462 |
| lxe's quant on this engine, dense Q8_0 | **11.63** | **71.49 %** | 0.475 |
| lxe's quant on this engine, dense Q6_K (`STRATA_DENSE_REQUANT=q6_K`) | 11.73 | 70.87 % | 0.488 |
| Lucebox ROCMFP 2.5 bpw (published, another engine) | 9.78 | 78.7 % | 0.253 |

So the loading is faithful (ppl and top-1 match the author's within 0.3 %) and the Q6_K dense costs +0.9 % ppl; the
errors come from the 2.03 bpw quant itself. The quant was deleted and the GSQ restored from the USB backup
(`local/restore-gsq.sh`, sha256-checked); the 1017 Engram shards stay in `models/engram-1017/` for a better 1017 quant.

**An agent session, and two levers from it** (2026-10-18). A Pi session against the chat (23 turns, context
18K -> 41.5K, `local/serve-engine.log`): decode took 1,263 s of the 1,460 s and ran at 10-12 tok/s, 11.5 on average
(8.2 on the last turn); per decoded token 60-65% of the expert lookups hit VRAM (~1,785 experts there), ~85 were
computed on the CPU and ~10 read from NVMe - against 69-75%, 60-70 and 3-5 on short Italian chats. Prompts were 13% of
the time: the layer-major batch costs ~4-6 s even for 372-600 new tokens, and the one 171-token turn, which the exact
passes served (`STRATA_SERVE_STEP_MAX` 256), took 9.3 s. Two levers were measured with `local/bench-v41/ab-levers.sh`
(fresh server per run, two runs each):

- **`STRATA_SERVE_STEP_MAX` 256 -> 96**, `followup-sizes.py` (25K first turn, then +171, +256, +341, +425-token
  follow-ups): +171 turn 9.71-9.80 s -> 3.92-4.10 s, +256 turn 12.85-14.37 s -> 4.35-6.45 s; the batch then re-opens
  the VRAM cache (seed 1.7-2.9 s, before the first token), so time to first token 9.7-9.8 -> 6.3-6.8 s and 12.9-14.4
  -> 6.0-9.4 s. The +341 and +425 turns run the batch either way (4.8-6.3 s). The passes read ~17.5 new tokens a
  second, the batch has a ~4 s floor: the crossover is near 70-100 tokens. **Adopted** in `local/ds41-local*.json`.
- **A rank from the chat's own routes** (`local/rebuild-rank.sh 1`: the calibration counts plus 48K routed chat
  tokens, mostly this agent session; half of the VRAM head and a fifth of the arena change), agent replay: decode
  11.85 / 12.19 tok/s against 11.81 / 11.73 with `rank-gsq-mixed.bin`, VRAM hits 63.1-63.4% against 63.9%, CPU
  experts 83-84 against 82 a token. Inside the spread, **not adopted**.

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

## How long card 0 waits for card 1 (2026-10-20)

`STRATA_PEER_PROF=1` (opt-in diagnostic, `PeerGpu::prof_handoff`): at each layer's hand-off card 0's stream is synced
first, then the peer's event, so the host sees when card 0 (its own launches and the CPU rows) was ready and how
much longer the peer's rows took. One-shot runs, the chat configuration's flags and env (`--arena-gib 75`, Q6_K dense
half, `--pcie 0`, `--pf-b 0`, the peer adaptive), forced on the 1-card run's tokens, 2 repetitions
(`local/bench-v41/ab-peerwait.sh`, results in `local/sweep-peerwait/`):

| | 1 card | 2 cards | 2 cards + profiler |
| --- | ---: | ---: | ---: |
| Promessi Sposi, 265 tokens, decode tok/s | 14.74 / 14.13 | 15.34 / 14.94 | 15.27 / 14.92 |
| 17,349-token prompt, 128 tokens, decode tok/s | 8.78 / 8.49 | 9.65 / 9.77 | 9.21 / 9.63 |

| profiler, per hand-off (one layer that used the peer) | Sposi | 17K |
| --- | ---: | ---: |
| hand-offs per token | 22.2 | 30.3 |
| card 0 ready after the peer's issue | 0.62-0.63 ms | 1.27-1.51 ms |
| peer later than card 0 | 69% | 45% |
| card 0 idle waiting for it | 0.135 ms (3.0 ms/token, ~5% of a token) | 0.097 ms (2.9 ms/token, ~3%) |
| issue -> peer rows back on the host, when late | 0.40 ms | 0.47 ms |
| host time to issue the peer's work | 0.017 ms | 0.018 ms |

The round trip over the x2 chipset link is ~0.4-0.5 ms and mostly hides behind card 0's own work; removing the wait
entirely is worth at most ~3-5%. What the second card changes is the CPU work: CPU-computed misses 18,134 -> 9,506
(Sposi) and 14,759 -> ~9,130 (17K), `cpu` 3.9 -> 2.1 s and 3.0 -> 1.95 s. The NVMe reads do not go down (Sposi 1,528
-> 1,432, 17K 1,818 -> ~1,900; `file` 3.7-4.2 s either way, ~30 ms/token at 17K): the peer's store is filled from the
arena, so it holds copies of arena experts, not experts that would otherwise be read from disk. The prompt is the
same with one or two cards (337-346 tok/s at 17K).

**A store disjoint from the arena.** Two opt-in envs (`tools/v41/engine/ds41_moe.cpp`), both off by default:

- `STRATA_PEER_DISJOINT=1` (`peer_seed`): card 1's store takes the ranked experts the arena does *not* hold (card 0's
  resident ones still left out), read from the model file by 8 threads through pinned bounces: 2,353-2,364 experts
  in 7.2-8.5 s, as long as the copy from the arena takes (7.2-8.4 s). `STRATA_PEER_ADAPT` still admits recurring
  arena misses over the least recently used peer slots.
- `STRATA_ARENA_EVICT_PEER=1` (`arena_victim`): with `--arena-adapt`, a file-tier read takes the arena slot of an
  expert card 1 holds first (the least recently used of those), since that copy is redundant.

The first was the plan's option (a); the second is the eviction half of option (b), measured alone and with (a). (a)
was the simpler one to make disjoint: one change in the seed, and the store holds no arena expert from the first
token on, where (b) only frees one duplicate per file read. Same flags and env as above, 2 repetitions alternated
with a fresh 2-card run in the same session, forced on the same reference tokens
(`local/bench-v41/ab-peerdisjoint.sh`, results in `local/sweep-peerdisjoint/`):

| Promessi Sposi, 265 tokens | 2 cards | + disjoint | + evict peer | + both |
| --- | ---: | ---: | ---: | ---: |
| decode tok/s | 13.53 / 14.61 | 14.89 / 15.77 | 15.50 / 15.50 | 14.89 / 16.26 |
| NVMe reads (`file`) | 1,432 | 964 / 953 | 1,272 / 1,265 | 836 / 831 |
| `file` ms | 4,457 / 3,839 | 2,870 / 3,180 | 3,546 / 3,582 | 2,491 / 2,680 |
| CPU-computed misses | 9,506 | 11,298 / 11,297 | 9,430 / 9,420 | 11,112 / 11,136 |
| card 0 VRAM hits | 82.6% | 79.5% | 82.4-82.5% | 79.4-79.5% |
| margin lines (forced) | 6 / 6 | 4 / 2 | 5 / 6 | 4 / 3 |

| 17,349-token prompt, 128 tokens | 2 cards | + disjoint | + evict peer | + both |
| --- | ---: | ---: | ---: | ---: |
| decode tok/s | 9.67 / 10.14 | 10.93 / 10.20 | 10.78 / 10.63 | 11.60 / 10.03 |
| NVMe reads (`file`) | 1,894 / 1,916 | 1,235 / 1,247 | 1,676 / 1,665 | 1,119 / 1,132 |
| `file` ms | 4,211 / 3,948 | 3,204 / 3,043 | 3,552 / 3,579 | 2,852 / 3,023 |
| CPU-computed misses | 9,114 / 9,149 | 9,562 / 9,641 | 9,047 / 9,080 | 9,531 / 9,523 |
| card 0 VRAM hits | 61.6 / 61.3% | 58.3 / 58.1% | 60.6% | 55.8 / 55.9% |
| margin lines (forced) | 0 | 0 | 0 | 0 |

The reads go down as intended and the counts repeat run to run: both envs together cut them 42% on Sposi (1,432 ->
~834) and 41% at 17K (~1,905 -> ~1,126), `file` 4.1 -> 2.6-2.9 s (17K: ~32 -> ~23 ms/token); disjoint alone 33-35%,
evict alone 11-12%. Two costs eat part of it. Experts the peer serves from outside the arena have no arena copy, and
`STRATA_VRAM_BG` promotes a recurring miss into card 0 only from its arena copy, so card 0's VRAM hits fall (82.6 ->
79.5% on Sposi, 61.5 -> 55.9% at 17K with both). And the arena's hottest experts start off card 1, so more misses
go to the CPU (Sposi 9,506 -> ~11,100-11,300) until `STRATA_PEER_ADAPT` brings them over. The tokens stay as close to
the reference as the plain 2-card runs' tokens do (2-4 margin lines against 6 on Sposi, 0 at 17K).

Decode, mean of 2: Sposi 14.07 (2 cards) -> 15.33 / 15.50 / 15.58 tok/s (disjoint / evict / both), +9% / +10% / +11%;
17K 9.91 -> 10.57 / 10.71 / 10.82, +7% / +8% / +9%. Against all four plain 2-card runs of the day (both sessions,
Sposi mean 14.61, 17K mean 9.81) both envs give +7% and +10%. The three variants are not separated by 2 runs: their
means are within 0.25 tok/s of each other, and one 17K run with both (10.03) was slower than its own 2-card run (10.14),
while evict alone was the steadiest (15.50 / 15.50, 10.78 / 10.63).

Verdict: make `STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1` the 2-card config's default. It has the highest mean
on both prompts, no loss in token agreement, the fewest NVMe reads, and the same startup time. Whether evict alone is as
fast needs more than 2 runs; it does not cut reads nearly as much (11-12%).

## Where a two-card token goes (2026-10-20)

The default two-card configuration (`local/ds41-local.json`), one-shot, forced on reference tokens
(`local/bench-v41/prof-decode.sh`, results in `local/sweep-prof/`).

**Stages** (`STRATA_DEC_PROF=1`, ms/token):

| | begin | attn_router | tier | finish | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promessi Sposi (246 tokens) | 2.8 | 27.9 | 30.8 | 1.5 | 15.8 |
| 17,349-token prompt (128 tokens) | 0.6 | 37.3 | 59.6 | 2.0 | 10.7 |

Inside the tier, per token: Sposi 42.9 experts computed on the CPU (`cpu` 10.5 ms) and 3.2 file reads (`file` 9.2
ms); 17K 74 on the CPU (16.9 ms) and 8.8 file reads (25.9 ms). A file read costs 2.5-2.9 ms in the engine.

**Card 0's kernels** (`rocprofv3 --kernel-trace --memory-copy-trace`, a run of 80 tokens minus one of 16 = 64 decode
tokens, `local/bench-v41/trace-summary.py`): 3,582 kernels and 25.7 ms busy per token (card 1: 115 kernels, 2.8 ms).
`mul_mat_vec_q` (the Q6_K dense half) 321 launches, 10.7 ms: ~5.6 GB at ~530 GB/s, 55% of the card's 960 GB/s. The
expert kernels 3.4 ms (card 1: 2.1 ms), 377 `copyBuffer` 1.2 ms, and 164 host-to-device + 72 device-to-host copies a
token at 39 / 61 us each. Card 0's idle gaps over 20 us (last 3 s of the profiled run): **25.7 ms/token after its
own expert kernels, waiting for the CPU tier's rows** (once per layer), 2.8 ms between layers, ~4.4 ms of copies
queued behind each other. So the card waits for the CPU tier most of the token.

**The CPU tier's limits, measured apart from the engine:**
- RAM (`local/bench-v41/membw.c`): 12 threads streaming an 8 GiB buffer read 51.6-54.3 GB/s, the same with 4 KiB pages and transparent huge
  pages (6 / 24 threads: 52.0-54.3). The CPU experts run at ~47-50 GB/s (`cpu` ms against 9.7 MiB per expert), so
  they are at the RAM's limit: faster CPU kernels or huge pages cannot help them, only fewer of them or faster RAM.
- NVMe (Lexar NM1090 PRO, PCIe 5.0 x4, ext4; `local/bench-v41/nvme-blob-read.c`): O_DIRECT reads of 9.7 MiB at random offsets of the GSQ file take 3.01
  ms on 1 thread, 1.65 ms on 8, 1.24-1.27 ms on 16-32 (8.0-8.2 GB/s), and two to four blobs in flight reach 9.5-11
  GB/s. The engine pays 2.5-2.9 ms per read.

**ROCm runtime knobs** (Sposi, against 14.99 / 14.19 / 15.52 tok/s): `HIP_FORCE_DEV_KERNARG=1` 15.17 / 15.48 and
`HSA_ENABLE_INTERRUPT=0` 15.58 / 15.03, both inside the spread; `HSA_ENABLE_SDMA=0` (copies by blit kernels)
11.58 / 11.72, -22%; all three 12.52 / 11.37. Checked and fine as they are: both cards in identity (pass-through)
IOMMU domains, Resizable BAR on (32 GiB BAR), the memory clock at its top level in ~97% of decode samples. The core
clock's median during decode was 2.2 GHz on Sposi and **1.3 GHz at 17K** (level `auto`): the card idles between
layers and the clock drops meanwhile (`local/gpu-perf.sh` sets `high` / `compute`, with root; not measured yet).

**Noise:** single runs of the same configuration differ by 5-10% here (16.46 against 15.67, 16.33 against 14.87), so
every comparison below has at least two runs each.

**Fast runs and slow runs.** The same configuration lands in one of two modes, and the split of the token time
(`local/sweep-prof/`, 30+ runs per prompt across the variants below) shows what differs:

| ms/token | total | CPU experts | file reads (per read) | the rest |
| --- | ---: | ---: | ---: | ---: |
| 17K, fast runs | 71-74 | 16-18 | ~11 (~1.25 ms) | ~44 |
| 17K, slow runs | 89-103 | 16-18 | 22-26 (2.5-2.9 ms) | 49-59 |
| Sposi, fast runs | 53-55 | 9.4-10.4 | ~4 (~1.3 ms) | ~40 |
| Sposi, slow runs | 59-70 | 9.3-10.5 | 9-12 (2.9-3.7 ms) | 42-49 |

In the fast runs a file read costs what the drive delivers when busy; in the slow ones twice that, and the rest of the
token is slower too. The default configuration's 17K runs: 2 fast of 5 (mean 12.11 tok/s, 10.92-13.93).

**Idle power states** (`local/bench-v41/nvme-idle-read.c`: one 9.7 MiB O_DIRECT read on 8 threads every GAP,
60 reads; "CPU tick" = a thread waking every 0.5 ms and doing nothing; "NVMe keep-alive" = a 4 KiB read every 0.5 ms):

| a read every | alone | + CPU tick | + NVMe keep-alive |
| --- | ---: | ---: | ---: |
| 10 ms | 2.16-2.77 ms (p90 4.6-6.0) | 1.17 | 1.19-1.29 |
| 20 ms | 5.54 | 4.92 | 1.25 |
| 30 ms | 5.59 | - | 1.36 |

Two states add up: at 10 ms the CPU's deep idle state (C3, 350 us exit latency on this box) is what costs, from
~20 ms the drive's own power saving (APST, enabled: `pm_qos_latency_tolerance_us` 100000). Decode's reads come 10-20
ms apart. Twelve busy-waiting threads (the CPU pool parked) did not change the microbench (2.69 ms at 10 ms).

**`local/power-tune.sh on`** (root: APST off, C3 off, the `performance` governor) brings the microbench's spaced reads
to 1.15 / 1.51 / 1.17 ms at 10 / 20 / 30 ms, and the engine (default two-card configuration, 5 runs each):

| | before (tok/s) | mean | with `power-tune.sh on` (tok/s) | mean | file read |
| --- | --- | ---: | --- | ---: | ---: |
| 17K | 10.96 10.92 13.93 10.92 13.84 | 12.11 | 12.92 13.83 13.83 13.85 13.72 | **13.63 (+12.6%)** | 2.04 -> 1.40 ms |
| Sposi | 16.68 18.72 18.34 16.05 16.06 | 17.17 | 17.60 18.70 18.69 15.34 18.15 | 17.70 (+3%) | 2.18 -> 1.45 ms |

At 17K every run is now a fast one (ms/token 83.0 -> 72.8: file 17.8 -> 12.1, the rest 48.8 -> 44.5). On Sposi one
run of five is still slow, and not because of the reads: its file and CPU time are the others' (5.4 / 9.4 ms/token)
and the rest is 50.2 ms against 39.7-41.3 - a second source of spread, in the dense half or the host. The settings
cost idle power and do not survive a reboot; `local/power-tune.sh off` restores the boot values.

**The cards' clock level `high`** (`local/gpu-perf.sh high`, on top of `power-tune.sh on`, 5 runs each) is slower:
17K 13.23 13.14 12.99 12.86 13.22 = 13.09 tok/s against 13.63 at `auto`, Sposi 17.37 17.34 17.18 16.82 15.75 = 16.89
against 17.70 (the rest of the token 44.5 -> 47.5 and 42.4 -> 45.1 ms; the first 17K run overlapped another session's
GPU benchmark and was repeated as r6). Even at `high` card 0's core clock averaged
1.74 GHz while busy, 29% of the samples under 1.5 GHz. Keep `auto`.

**Host-path options** (all opt-in, the default path unchanged; tokens identical to the reference where checked):

| tok/s, one value per run | 17K | Sposi |
| --- | --- | --- |
| default | 10.96 10.92 13.93 10.92 13.84 | 16.68 18.72 18.34 16.05 16.06 |
| `STRATA_SUM_ZEROCOPY=1` (the sum reads the CPU's and the peer's rows from pinned host memory: ~105 fewer copies a token) | 13.49 10.55 | 15.92 18.83 |
| `STRATA_FILE_POOL=1 STRATA_FILE_THREADS=16` (file reads on persistent threads, 16 parts) | 11.11 10.41 | 18.61 16.27 |
| `STRATA_FILE_THREADS=16` / `32` without the pool (threads created per layer) | 9.77 / 9.12 | 14.87 / 14.33 |
| `STRATA_POOL_SPIN_US=300` (the CPU pool sleeps after 300 us instead of 20 ms) | 13.49 13.62 13.54 13.35 10.20 | 18.20 14.39 15.17 15.37 14.30 |
| `STRATA_POOL_SPIN_US=100` / `1000` | 10.19 11.28 10.13 / 9.61 9.77 13.34 | 15.64 16.76 18.64 / 16.80 14.86 15.17 |
| `STRATA_SPEC_TOP=4 STRATA_SPEC_ADAPT=1` + the pool (speculative reads two layers ahead under `--arena-adapt`) | 8.77 9.12 | 14.99 15.32 |
| `STRATA_SPEC_TOP=2 STRATA_SPEC_ADAPT=1` + the pool | 11.88 11.08 | 15.27 13.82 |

None separates from the two modes, except the speculative reads, which lose: at 17K 8,443 reads issued for 333 used
of ~1,140 file misses (TOP 4), 3,845 for ~200 (TOP 2) - 4-5% of what they read is used, and the rest costs RAM
bandwidth (`cpu` 2.1 -> 2.7-3.1 s). `STRATA_NVME_KEEPALIVE_US=500` (the keep-alive inside the engine) and
`STRATA_POOL_YIELD=1` (the parked pool yields its CPU) were only smoke-tested.

**Prompt-lookup drafting** (`local/bench-v41/pld-sim.py`, the model's tokenizer, no model run): the next tokens
copied from the conversation after the last 4/3/2-token match. On the Pi session of 2026-10-18 that used ds41e (10,240
output tokens) 1.42-1.51 tokens per step (drafts of 4-16), 2.5-3.1 per drafted pass; on two coding-agent sessions of
2026-10-19 (another model, 19,415 and 47,614 output tokens) 1.33-1.40. The reasoning text is mostly new, so a verify
pass (which costs several single steps here) would rarely pay.

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
