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

Through the server (`./run-v41-gsq.sh`, ctx 262144, arena 75 GiB, temperature 0): **11.8-14.1 tok/s** (Divina
Commedia 11.84 cold, 14.05 repeated; the nine circles 13.64, TTFT 1.3 s). The chat with exact routing ran at ~2.7 tok/s
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

## What does not work

- **The near-tie bias** (`STRATA_RB_MARGIN=d`): d = 0.02 changes 5 of 300 positions and is no faster (45.7% VRAM
  hits, the same as exact); d = 0.05 changes 8 and is no faster. The bias is fast only because it rewrites many
  choices, so restricting it to near-ties leaves nothing to gain. `STRATA_RB_MARGIN=1` (the old chat config) is
  wider than the scores' spread: it is the flat bias.
- **File-tier-only penalty with the top 4 protected** (`STRATA_RB_PROTECT=4 --route-bias-file 0.3`): 6 of 300
  changed, no faster.
- **DSpark on Italian prose**: the drafter accepts ~1.0 draft per pass (code: 2.5-2.9), so the verify width is mostly
  wasted; 11.06 vs 10.68 tok/s single-step.

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

1. **The dense half's host synchronizations** (`AMD_HIP_NOTES.md`: 215 stream syncs a token, 76% of the token's wall):
   it caps every path, single-token and verify.
2. **An adaptive peer store**: card 1's store is static; admitting card 1's own misses over the x2 link (async, ~3 ms
   per blob) would let both cards follow the conversation's working set (one answer uses < 64 GiB of experts).
3. **The verify pass's tier** (105-195 ms for 5.5 tokens): union hit rate 60-74% against 88% single-step.
4. **A drafter that accepts Italian**: at ~1 draft per pass DSpark cannot pay on prose.
