# Strata-DSV4.1 — full handover: what was done, what was measured, what is blocking

Written 2026-10-17, end of a long session, for a stronger model to continue from. Repository:
`/run/media/vincenzo/Dati/projects/other/Strata-DSV4.1`, branch `ds4-q2` (last commit `c583b2da`; everything after it is
**uncommitted**). Nothing here is speculation unless marked; every number was measured on this machine.

> **Update 2026-10-17 (evening):** exact routing now runs at 10-14 tok/s through the chat (was ~2.7) with the same
> answers - see [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md): the GSQ's `rank.bin` was the main defect
> (11.5% coverage of the VRAM head), plus `--arena-adapt`, `STRATA_PF_NO_DUMMY=1`, the second card as a miss tier
> (`--peer-device 1`), `--threads 12` and `--vram-margin 0.5`. The near-tie bias (P1) was measured and does not pay.
>
> **Update 2026-10-18:** the chat runs on one card with the dense half re-quantized to Q6_K (16-18.6 tok/s on short
> prompts, ~11-14 at 25K context), and four upstream ds41 commits are in (small-turn prefill: a +304-token agent turn
> 4.8-5.0 -> 4.1-4.4 s with `STRATA_PRESTAGE_MIN=512`; the conversation-adaptive arena measured and left off) - all in
> [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md).

## 0. TL;DR

The stack is a fork of **Strata** (a Qwen3.8-oriented server + C++/HIP engine) with a v41 (DeepSeek-V4.1) shim and a
second engine (`tools/v41/engine`, HIP/ggml) that runs V4.1 with a **three-tier expert cache** (VRAM, RAM arena, file
tier). On this machine it runs two models:

- **GSQ** (`models/V41-GSQ`, 2.25 bpw, `IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored`): 10.8 tok/s with the
  engine's route bias, **~2.7-3.2 tok/s with the routing made exact**, and in that exact configuration it answers
  **correctly** (verified: the nine circles of Hell, a full Divine Comedy summary with correct dates, structure,
  characters).
- **ds4's Q2** (`DeepSeek-V4.1-Flash-Q2.gguf`, IQ2_XXS/Q2_K, antirez's model): 9.0 tok/s with the bias, **4.53 tok/s
  with exact routing and its own rank.bin**, correct circles. It does **not** work through the chat server (its
  thinking never closes).

**The central discovery**: the engine's expert-selection **route bias** (a term added to the router score that depends
on where each expert's bytes live: VRAM, RAM arena, file tier) is what makes both models invent facts. With
`--route-bias 0 --route-bias-file 0` the facts are right and the speed collapses; the whole session was about that
trade-off.

## 1. Hardware and storage

- 2× AMD Radeon RX 7900 XTX (gfx1100, 24 GB each, ROCm 7.14, Clang 23) **plus an iGPU** (gfx1036) that ggml sees as
  device 2 — the engine must keep using device 0 only. Only GPU 0 is used today; **GPU 1 is idle**.
- 122 GiB RAM (~110 GiB available). 24 threads used.
- Model disk: `/run/media/vincenzo/Dati` (1.4 TB, ~265 GB free with the GSQ set). The Q2 is on a second NVMe:
  `/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf`.
- Models:
  - `models/V41-GSQ/V41-GSQ.gguf` 162,957,934,592 B (Q8_0 attention+output, Q3_K and two fork-private types on the
    experts) + `models/engram/layers1|14` (native FP8 safetensors, 101,377,628,584 + 101,380,404,328 B) +
    `models/V41-GSQ/rank.bin` + the DSpark drafter `V41-MTP-Vision.Q2_K.gguf` (5.66 GB).
  - Q2 GGUF 365,713,686,528 B, arch `deepseek41`, 1046 tensors: I8 202.8 GB (the **two Engram tables packed inside**,
    `blk.1.engram_embd` / `blk.14.engram_embd`), IQ2_XXS 93.4 GB, Q2_K 59.5 GB, Q8_0 7.6 GB, F16 2.2 GB, F32 0.3 GB.
    Its Engram rows are **the same native FP8 data** as the GSQ's safetensors.

## 2. The stack, in the order a request travels

1. **Client** (web chat at `http://127.0.0.1:8150`, or any OpenAI/Anthropic client) → `serve_ds41.py` (the v41 shim,
   `tools/v41/serve/serve_ds41.py`) → the platform's `serve/server.py` → the engine subprocess
   (`build-hip-fused/ds41_generate --serve`) over a line protocol.
2. **Prompt rendering**: `serve_ds41.py`'s `render()` decides the mode: `mode = "chat" if kw.get("enable_thinking") is
   False else "thinking"`. So the thinking is ON unless a client turns it off; `_level()`/`effort_kwargs()` only choose
   the *number* (V4.1 renders "Reasoning Effort: N", HIGH=75, MAX=100; the low/medium names are Qwen's and render as
   75). The **platform's sampling** = `{**config_sampling, **shared, **request_values}`, so the web panel's values win
   over the config; "Use for other apps" off means the panel only affects the chat.
3. **Engine**: `tools/v41/engine/ds41_generate.cpp` (driver, sampling, DSpark, file tier) + `ds41_dense.cpp` (attention,
   router, shared expert; the router is at lines ~655-700, `ggml_argsort_top_k(gc, sel, NU)` at ~697 gives the `ids`)
   + `ds41_moe.cpp` (the expert tier: VRAM cache, predicted prefetch, CPU pool, PCIe share, file tier; `ds41_moe.hpp`
   documents the rank format and the tier defaults `gu_type=16`, `d_type=10`) + `ds41_engram.cpp` + `ds41_dspark.cpp`.
   `ds41_geometry.hpp` reads the GGUF's `deepseek41.*` metadata at runtime and validates the full tensor map.
4. **The three tiers** of the routed experts: VRAM cache (slot per expert blob), RAM arena (`--arena-gib`), file tier
   (O_DIRECT preads of the expert blob). The VRAM/arena seed follows `--rank` (hottest experts first, by routed count
   per byte).
5. **DSpark** (speculative decoding) is ON in the chat (`STRATA_SERVE_DSPARK` = the GSQ's MTP drafter, depth 4, KEEP 64).

## 3. The route bias — the core defect (read this first)

`ds41_dense.cpp` ~655: `sel = s0 + rb` where `s0` is the model's own selection score (`sqrt(softplus(router)) +
exp_probs_b.bias`) and `rb` is filled in `ds41_generate.cpp` (~514): **+route_bias (+1.0 default, 0.5 in the shipped
serve config) if the expert is resident in VRAM now, -route_bias_file (0.6 / 0.3) if it is in the file tier**. The
weights of the chosen experts still come from the unbiased scores. The engine's own comments record the cost: "-14 pt
API top-1, repetition loops", perplexity 2.38 -> 2.67, agentic 15.1 -> 6.9 at 1.0/0.6, KL 0.10-0.15.

**Timing matters**: `bias_now` is false until the VRAM seed, so the **prompt** routes exactly (the model understands
the question) and **every generated token routes on the biased set** (it then invents). That is exactly the observed
symptom: fluent, coherent openings, wrong facts, loops.

**Measured** (same prompt, greedy, ~600 tokens): with the bias the model says "the Inferno is divided into nine
circles (stage)", "the Purgatorio into seven stage" (the circles never named). With `--route-bias 0 --route-bias-file
0` it names all nine. The **Q2** behaves the same way: bias -> "Cerchio della Terra / Cielo"; exact -> "Limbo,
Lussuria, Gola, Avidità, Ira, Eresia, Violenza, Frode, Tradimento".

**`STRATA_RB_MARGIN=d` and `STRATA_RB_PROTECT=P`** (opt-in paths that restrict the bias to near-ties / protect the
exact top-P) **do not reproduce the exact-routing answer** in this build: measured GSQ margin=1 -> 11.99 tok/s, 95.7%
hits, wrong list; Q2 margin=1 -> 6.9 tok/s, wrong list. **This is an open problem**: either the implementation or the
expectation is wrong, and the README claim written earlier in the session ("margin=1 = same correct answer") is
**not reproducible** and is flagged, not relied on.

## 4. What was built (uncommitted, in the working tree)

1. `ds41_geometry.hpp`: the tensor-map validation accepts `blk.N.engram_embd.*` (the engine reads the Engram from
   `--engram` safetensors, same rows). The mutation test still rejects every other unexpected tensor.
2. `ds41_dense.cpp`: the dense loader skips those tensors (the `skip` lambda next to `token_embd.weight`). Without it
   the dense init tried to allocate 197 GiB on the GPU.
3. `ds41_moe.cpp`: `uniform_slots()` caps the slot **count** by the automatic free-VRAM budget on the uniform path
   (an explicit `--slots N` is unchanged); used in the init open and the deferred seed.
4. `ds41_generate.cpp`: `mc.defer_cache = defer_vram` (ds4's `defer_seed`, never wired in the v41 fork) — without it the
   first serve request leaves a uniform geometry with zero VRAM slots (the server decoded at 6.2 tok/s).
5. `ds41_generate.cpp`: **`--dump-routes DIR`** — counts the selected (layer, expert) pairs at the host `ids` (5 call
   sites) and writes `DIR/LNN.json` in `v41_rank.py`'s format (`experts[e].n`). No arithmetic change. Plus
   **`--arena-skip-vram`** (the arena holds the ranking past the VRAM head instead of duplicating it).
6. `tools/ds4/cmake/ggml_cuda.cmake`: new `STRATA_GGML_HIP` option (ggml's HIP backend instead of CUDA);
   `ds4_engine.cmake` links `ggml-hip` when present. The working HIP configure is in §8.
7. `local/rank-q2.bin` (md5 `21d9e957e557e3f9e6579a496fb1dd89`) built from 8,296 tokens / 1,991,040 selections; the
   calibration corpus and the reproduction steps are in `local/calib-q2/`.
8. Chat/UI and configs: `serve/web/app.js` defaults (temperature 0.8, top_k 0), `serve/web/index.html` (the top-k
   slider minimum 0), `local/ds41-local.json` (budget 192, penalty 1.1/256, `STRATA_RB_MARGIN=1`, and now
   `--route-bias 0 --route-bias-file 0`), `local/ds41-local-q2*.json`, `local/ds41-local-gsq-nodspark.json`,
   `local/test-q2-rank.sh` (acceptance harness), `local/prompts/{circles,long-it}.i32`, `docs/ds4/DS4_Q2_STATUS.md`.
9. Docs: README "The ds4 Q2 weights" and "The Q2's own rank.bin", `docs/ds4/V41_SAMPLER_PENALTIES.md`, this file.

## 5. Every measurement that matters

| what | result |
|---|---|
| GSQ, chat, default bias | 10.8 tok/s, wrong facts |
| GSQ, exact routing (`--route-bias 0`), chat | ~2.7-3.2 tok/s, **correct facts** (nine circles; full Divine Comedy summary verified) |
| GSQ, exact routing, one-shot | 3.19 tok/s, nine circles |
| Q2, default bias | 9.01 tok/s, wrong facts |
| Q2, exact routing, GSQ rank | 0.97 tok/s, nine circles, 45.7% VRAM hits, ~86 file reads/token |
| Q2, exact routing, **Q2 rank** (75 GiB) | 3.13 tok/s, nine circles, 20 reads/token |
| Q2 + `--arena-skip-vram` (75 GiB) | 3.49 tok/s, 14.5 reads/token |
| **Q2 + `--arena-skip-vram` (90 GiB)** | **4.53 tok/s**, 7.4 reads/token |
| Q2 rank coverage | 8/16/32/48/64/80/100 GiB -> 38.0/52.4/70.3/81.7/89.4/94.5/98.2% of routed tokens |
| ceiling with exact routing | the file tier is 3.1% of lookups at 90 GiB (4.5 s of 13.0 s); the non-file cost is ~0.13 s/token -> **even a zero file tier is ~7.4-7.7 tok/s** |
| thinking budget | 2048 -> no content at all; 384 -> content but the deliberation leaks; **192 -> finish=stop, thinking ~114 words, clean content** |
| loop metric (12-gram coverage, last 600 words) | bias runs 48.2%; with penalties 35.2%; exact-routing runs 0% |
| route-bias 0 cost | Q2 9x slower (9.01 -> 0.97 before its rank), GSQ 3.4x (10.81 -> 3.19) |
| DSpark at exact routing | slower (Q2 0.82 vs 0.97) because the decode is I/O-bound; it is ON and drafts 2.0-2.5 tok/pass |
| the cycle that failed | a first calibration corpus that was 88% one repeated prompt: 99% self-coverage, 85.9% on the target, 2.10 tok/s |

## 6. Constraints and traps found (the useful part for whoever continues)

1. **The expert set is 142.4 GiB; this machine holds 104 GiB (90 arena + 14 VRAM).** Even perfect ranking leaves ~38
   GiB on the SSD, and the serial 9.5 MiB O_DIRECT preads run at ~0.93 GB/s. The ceiling with exact routing is ~7.4-7.7
   tok/s.
2. **`max_tokens` must exceed `reasoning_budget_tokens` plus the answer.** With budget 192 and `max_tokens: 200` the
   content comes back **empty** (the cap is consumed inside the wrap-up). Use >= 600, or the CLI with `-n >= 200`.
3. **The Q2's thinking does not close through the platform's budget mechanism**: with the Q2 as the chat model the
   answer is empty (`finish=length`, content 0) while the GSQ closes correctly. The Q2 is usable one-shot only.
4. **`tools/ds4/` is a different engine** for the older V4 Flash (`deepseek4.*` metadata keys); it cannot load this
   model ("missing deepseek4.embedding_length"). The model belongs to `tools/v41` (`deepseek41.*`).
5. **`v41_calib.py` needs the original FP8** (another machine) — not a path here. The rank was built by making the
   engine count its own routing (`--dump-routes`).
6. **The web panel's values override the config** per request; "Reset" is needed after changing the defaults because
   the panel stores them in localStorage.
7. **The chat server holds the 75-90 GiB arena**: a one-shot run must stop it first (`fuser -k 8150/tcp`).
8. **`memguard.sh` needs `mkdir -p ~/.quetza-data/conductor`** (a lock file path from the author's machine).
9. **The abliteration hypothesis is weaker than it looked**: dealignai's card says only 27/48 dense shards were edited,
   routed experts/Engram/router are byte-identical to base, and claims -1.1 pp MMLU outside the ethics cluster
   (self-reported). The measured cause of the wrong facts is the **route bias**, not the abliteration; the 2.06-2.25
   bpw quant is the second suspect.
10. **The routing dominates everything at 2 bits**: with the bias, both models degrade; exact routing restores the facts.

## 7. The open problems, ranked for a stronger model

**P1 — the route bias trade-off (facts vs speed).** The bias exists to keep hot experts in VRAM; it also rewrites the
model's expert choice on every generated token. Fixing it properly means a selection that keeps the model's exact top-k
and only *prefers* resident experts among near-ties. The code has two attempts (`STRATA_RB_PROTECT`,
`STRATA_RB_MARGIN`) that **do not work as documented** (measurements in §3) — explain and fix them, then the machine
gets correct facts at ~9-11 tok/s. This is the single highest-value item.

**P2 — speed under exact routing.** Done: the Q2's own rank (0.97 -> 4.53). Next: use **GPU 1** as an extra expert
tier (+21 GiB ≈ +2,300 experts); parallelise the file-tier preads (the serial 9.5 MiB reads at 0.93 GB/s are the
floor); consider a larger arena (90 GiB leaves ~20 GiB RAM — risky). Note a *previous* attempt at a peer GPU tier
measured 11.8 vs 12.6 tok/s with the **biased** routing (where I/O was not the bottleneck); it must be re-measured
under exact routing, where it should help.

**P3 — the Q2 in the chat.** Its thinking does not close with `reasoning_budget_tokens` (content empty). Either find
why the wrap-up path fails for those weights (the same mechanism works for the GSQ) or keep the Q2 one-shot only.

**P4 — knowledge quality.** Even with exact routing, a 2.06-2.25 bpw quant confabulates on encyclopedic detail. The
only real cure here is a higher-precision model; the candidates that would run on this hardware were researched and
none is deployable (see `docs/ds4/DS4_Q2_STATUS.md` and the earlier research: llama.cpp cannot load V4.1, SGLang/other
V4.1 runtimes are CUDA/2-device only, and the Metal engine is Apple-only).

## 8. How to run everything (reproducible)

The chat (GSQ, exact routing, correct; ~2.7 tok/s):
```
cd /run/media/vincenzo/Dati/projects/other/Strata-DSV4.1 && ./run-v41-gsq.sh    # http://127.0.0.1:8150
```
`local/ds41-local.json` today: model = GSQ, `--rank models/V41-GSQ/rank.bin`, `--route-bias 0 --route-bias-file 0`,
arena 75 GiB, ctx 262144, kv-q8, budget 192, sampling temp 0.8 / top_p 0.95 / repetition_penalty 1.1 / penalty_last_n
256, `STRATA_RB_MARGIN=1`.

The Q2 one-shot, fastest correct (4.53 tok/s):
```
cd /run/media/vincenzo/Dati/projects/other/Strata-DSV4.1
mkdir -p ~/.quetza-data/conductor
fuser -k 8150/tcp
bash tools/ds4/memguard.sh 110 82 -- ./build-hip-fused/ds41_generate \
  -m "/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf" \
  --ids-file local/prompts/circles.i32 -n 200 --temp 0 --backend cuda --experts gpu \
  --arena-gib 90 --ctx 8192 --kv-q8 --threads 24 --rank local/rank-q2.bin \
  --engram models/engram/layers14:models/engram/layers1 \
  --route-bias 0 --route-bias-file 0 --arena-skip-vram > /tmp/q2.out 2>&1
grep -E '^[0-9]+$' /tmp/q2.out | /run/media/vincenzo/Dati/projects/other/Strata/.venv/bin/python local/detok.py
```
The safe variant is `memguard.sh 80 82` + `--arena-gib 75` (3.49 tok/s). `local/test-q2-rank.sh` wraps all of this
(it uses `--ctx 262144`, so expect ~3.5 tok/s). A custom prompt: `local/encode_prompt.py --chat "..." out.i32`
(with the Strata venv, which has numpy).

Build (HIP, whole tree):
```
cmake -S . -B /tmp/build-ds4-cfg -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dhip_DIR=/opt/rocm/lib/cmake/hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_GGML_HIP=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
make -C /tmp/build-ds4-cfg -j18 ds4_generate_cpu     # the tools/ds4 CPU engine (V4 Flash keys)
make -C build-hip-fused -j18 ds41_generate test_ds41_geometry
./build-hip-fused/test_ds41_geometry "<any V4.1 gguf>"   # must print all ok
```

## 9. Files, state, git

- **Committed**: `c583b2da` "v41: run the ds4 Q2 weights, and fix the uniform expert-cache budget" (the two tier bugs,
  the two Engram patches, the two CMake HIP patches, the README section, `docs/ds4/DS4_Q2_STATUS.md`, the three new
  configs). Plus the earlier commits of the same session (`3c4f4c5f` penalties, `3a0a2b1f` local scripts,
  `2ad065b9` README, `5310be69` cleanup).
- **Uncommitted** (branch `ds4-q2`): `ds41_generate.cpp` (`--dump-routes`, `--arena-skip-vram`), the chat config
  (`--route-bias 0`), `local/rank-q2.bin`, `local/calib-q2/`, `local/test-q2-rank.sh`,
  `local/prompts/{long-it,circles}.i32`, the README rank section, the updated `DS4_Q2_STATUS.md`.
- **Untouched by design**: `models/` (the GSQ, its Engram, its rank) and the Q2 file on the SSD (read-only). Backups of
  every patched source are in `/tmp/*.bak` (`geom`, `dense`, `gc`, `de`, `root`, `shim`, `cfg-chat`, `cfg`).
- The chat is running on port 8150 with the GSQ + exact routing.

## 10. Methodology notes (so results stay comparable)

- Decode speed: `ds41_generate`'s own `decode: N tokens in T s = R tok/s`, greedy (`--temp 0`), same prompt.
- Loop detection: fraction of the last 600 words inside 12-word passages seen >= 5 times (the metric used throughout;
  >25% means a loop).
- Detokenising engine output: ids are one per line on stdout; `local/detok.py` with the **Strata** venv (needs numpy).
- The gate for any engine patch: with the defaults the greedy output must be **byte-identical** to the previous build
  (it is compared against `local/gate-before.txt` and the geometry test must print all ok).
- Never trust a model card, a comment or an earlier claim without re-measuring it: two "certain" diagnoses in this
  session (the 1-byte expert blob, the margin path) were wrong, and the README's margin claim had to be retracted.
