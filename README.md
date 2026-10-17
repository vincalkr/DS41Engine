<h1 align="center">Strata-DSV4.1 - AMD (gfx1100) branch</h1>

<p align="center"><b>DeepSeek-V4.1-Flash (40 layers x 384 experts) on ONE RX 7900 XTX - 152 GiB of experts across VRAM, pinned RAM and NVMe, plus 189 GiB of Engram on disk</b><br>
The AMD port and tuning of <a href="https://github.com/Indras-Mirror/Strata-DSV4.1">Strata-DSV4.1</a>, itself built on <a href="https://github.com/Niko1221/Strata">Strata</a> - Linux - ROCm/HIP - MIT</p>

<p align="center"><b>RX 7900 XTX (gfx1100, ROCm 7.14) + 128 GiB RAM:&nbsp; prefill 240-363 tok/s &nbsp;·&nbsp; DSpark decode 24.3 tok/s &nbsp;·&nbsp; 256K context, 8-bit KV</b></p>

## What this branch is

This is the **AMD (gfx1100)** branch of the fork `vincalkr/Strata-DSV4.1`: the same engine and serve as upstream, ported
and tuned to run DeepSeek-V4.1-Flash on **one** RX 7900 XTX.  Every number on this page was measured on that machine;
the engine runs on a single card (`"gpu": 0` in [local/ds41-local.json](local/ds41-local.json)).

The platform is not ours.  The engine, the serve and the earlier DeepSeek-V4-Flash (0731) path come from
[Indras-Mirror/Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1), which builds on
[Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT; `LICENSE` is kept as it is).  Their README - the NVIDIA
measurements, the 512K-context settings, the API and the V4-Flash engine - is kept verbatim in
**[docs/upstream/README.md](docs/upstream/README.md)**, with the localized copies next to it; read it for anything that
is not AMD-specific.  Our contribution is the AMD path: the port, the tuning, the measurements and these notes.

## The machine, and what is on disk

Two RX 7900 XTX are installed, but **one** is used.  128 GiB of RAM, one 2 TB NVMe, ROCm 7.14.  On disk:
`models/V41-GSQ/V41-GSQ.gguf` (152 GiB), the 5.3 GiB DSpark draft with its small files, and the two Engram tables in
`models/engram/` (189 GiB, layers 1 and 14).  The second card was tried as an expert tier and dropped: behind its
PCIe 4.0 **x2** bridge it is worth +10-20% as extra cache, not 2x (see the negatives below).

| measured here, one sequence | |
| --- | ---: |
| decode, 4K state, DSpark off | 13.7 tok/s |
| decode, 64K state, DSpark off | 14.0 tok/s |
| decode through the server, DSpark depth 4, ggml graphs off | 22.8 -> 24.3 tok/s |
| prompt, 5787 tokens | 240-245 tok/s |
| prompt, 63,617 tokens, fresh, through the server | 363 tok/s |

## The optimizations, each with what it was worth

| change | measured effect |
| --- | --- |
| ggml's fused dense ops (`local/build-ggml-fused.sh`: `dsv41-fake-quant.patch` + `dsv41-fuse.patch`) | +13% decode, +23% with DSpark; last-row logits top-1 identical, mean absolute difference 0.156 |
| `--arena-gib 75` (from 60) | +40% decode (16.4-17.4 -> 23.8-25.2 tok/s), prompt file wait 16.6 -> 11.0 s; 80 and above are worse or unstable |
| `GGML_CUDA_DISABLE_GRAPHS=1` | +6.6% decode, +3% prompt in the server |
| `STRATA_PRESTAGE_GIB=1.5` | +5.8% prompt (231.2 -> 244.5 tok/s) |
| `--kv-q8` (8-bit KV, not 4-bit) | lossless here: last-row cosine 0.998987, top-5 5/5, +0.37 GiB |
| DSpark depth 4 with `KEEP 64` | depth 5, depth 6, `KEEP 32` and `KEEP 128` all measured worse |
| `STRATA_SERVE_PASS_BIAS=0` | the upstream default (0.5/0.3) gives 2.2x the expert misses here: 12.1 against 25.1 tok/s |
| the v41 serve's thinking budget (`reasoning_budget_tokens 8192`, `reasoning_close_retry`) | a runaway reasoning loop is now closed at the budget and the model answers (tested with a 128-token budget) |
| the effort mapping aligned with the reference implementation | `low`/`medium`/`xhigh` resolve the way the reference server resolves them for V4.1 (75/75/75) |
| DSML tool-call parsing in the local path | tool calls reach the client, streaming and not |
| prompt batching | a fresh 63K prompt at `--prompt-batch 12288` can run out of memory and segfault the one-shot CLI; the server sizes the batch itself and does 363 tok/s |

## Where the time goes, and what is left

A runtime trace at 17K state: the GPU kernels are 32.2 ms of a 139 ms profiled token, and 215 `hipStreamSynchronize`
calls a token account for 105 ms - 95.3 of them after a device-to-host readback and 76.7 after the dense graph's
launch - with 4,238 kernel launches a token.  Decode is therefore **host/sync-bound, not GPU-bound**, and the
synchronizations are mostly real work being waited for (the experts' DMA and the launch latency), not idle overhead.
The lever that is left is the number of kernel launches.  The per-layer host handshake itself was investigated to the
end and closed: see [docs/ds4/V41_PIPELINE_OVERLAP.md](docs/ds4/V41_PIPELINE_OVERLAP.md).

## Tried and reverted

With the numbers, so they are not repeated: `--pf-b` below 1.0 (-19% decode; the default 1.43 sits just above that
cliff), DSpark depth 5 and 6, `KEEP 32` and `128`, `STRATA_DSPARK_ADAPT`, the adaptive arena, `STRATA_PROM=1`, the
peer expert tier on the second card (correct and bit-identical rows, 91.5 rows a token moved off card 0, but 11.8
against 12.6 tok/s), the transcode decode, `STRATA_HIP_WMMA` (the +5.4% was noise: v41 never calls that kernel), a
cache for `ggml_cuda_get_device` (120 calls a token, 0.06 ms) and the 685 per-token error queries (below 0.3 ms).

## Run it

```sh
./run-v41-gsq.sh                 # the server on port 8150, with the configuration in local/ds41-local.json
./run-v41-gsq.sh --port 8151     # extra server arguments are passed straight through
```

It first checks that the model and the two Engram tables are in `models/` and says what is missing if they are not.
It picks the interpreter itself: `STRATA_PY` if set, then a `.venv` in the repository, then `venv/`, then `python3` -
the first one that imports **pillow, psutil, regex and jinja2**, which are the only packages the server needs
(`python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2`; `.venv/` is already gitignored).
Loading takes a couple of minutes (152 GiB are mapped, the RAM arena is filled, the Engram tables are opened); once
it is up, `curl -s http://127.0.0.1:8150/v1/models` answers `"status": {"value": "loaded"}`, and the OpenAI- and
Anthropic-compatible API is on that port.  For a single generation without the server, use `local/run-one-shot.sh`
(with `EXE=$PWD/build-hip-fused/ds41_generate` for the fused build the server uses).

## Getting the model files

The model is not in git (355 GiB in total).  [local/fetch-model.sh](local/fetch-model.sh) builds exactly the layout
this branch uses:

1. the 16 parts of `V41-GSQ.gguf` plus `SHA256SUMS`, `rank.bin`, `V41-GSQ.gguf.experts.json`, the DSpark draft and the
   tokenizer, from [IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
   (resumable), joined and verified by the repo's own `reassemble.sh`, which checks the sha256 of every part before
   appending anything and verifies the joined file at the end.  The join keeps the parts until the file is verified
   (so an interruption can be re-run) and only then removes them; with less than twice the model free it joins in
   place instead, dropping each part as it appends it, and says so in the log.  What stays is the joined
   `V41-GSQ.gguf`, byte for byte the tree the numbers above were measured on;
2. the two Engram tables, which are **not** in that repo: they are shards 47 (layer 1) and 48 (layer 14) of
   [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8),
   fetched by `tools/v41/fetch_v41.sh` and extracted with `tools/v41/extract_engram.py` into `models/engram/layers1`
   and `models/engram/layers14`, with the merged index from `local/engram_index.py` (the engine needs it because the
   two tables live in different directories).

Space: 163 GB for the GGUF, then the two Engram tables (101 GB each) plus the two 101 GB shards they are extracted
from.  `extract_engram.py` works on the whole set - it validates every shard the index names and needs a `--to` for
every Engram layer - so both shards must be on disk at once and both destinations are given in one call: there is no
one-table-at-a-time order (measured: "shard not found: model-00048-of-00048.safetensors" with only 47 present, and
"no --to for layers.14" with a single `--to`).  Peak for that phase is therefore 163 + 203 + 202 GB, which is why
`WORK` and `DEST_LAYERS1`/`DEST_LAYERS14` exist: put the shards on one disk and the tables on another, and the peak
split across the two is what the machine has to hold.

```sh
local/fetch-model.sh /path/to/models     # DEST defaults to <repo>/models
```

It is resumable, it uses `aria2c` when it is installed (8 connections per file: measured 270 MiB/s against 5.7 MB/s for a single `curl`), and it checks: every expected size comes from the Hugging Face API, a file counts as complete only
when its size matches (so a partial download is resumed with `curl -C -`, never trusted), the parts are joined by the
repo's own `reassemble.sh` (which verifies their sha256 first and the joined file at the end), and an interrupted join
is reported instead of being joined over.  Every parameter:

| variable | default | what it does |
| --- | --- | --- |
| first argument | `<repo>/models` | where the trees go |
| `GSQ_REPO`, `FP8_REPO` | the two repositories linked above | where to fetch from |
| `JOBS` | 1 | how many files are fetched at once (one by default) |
| `CONN` | 4 | connections per file with `aria2c`.  The Hugging Face CDN throttles each connection: measured on this repo, 1 gives 5.7 MB/s, 4 gives ~27 MB/s and 8 gives **140 MiB/s** |
| `ARIA2` | 1 when `aria2c` is installed | `0` forces `curl` |
| `DRY_RUN=1` | 0 | print the whole plan (what is complete, what would be fetched, the join, the Engram steps) and change nothing |
| `DELETE_PARTS` | 1 | drop each GGUF part as it is appended; `0` keeps them and needs the space twice |
| `DEST_LAYERS1`, `DEST_LAYERS14` | `<models>/engram/layers1`, `layers14` | put the two Engram tables on different disks |
| `KEEP_FP8` | 0 | keep the FP8 shards after the extraction (101 GB each) |
| `FORCE_JOIN` | 0 | remove an interrupted `V41-GSQ.gguf` and join again (by default it stops and says so) |
| `VERIFY_ROWS` | 200 | how many random Engram rows are checked against the source |
| `WORK` | `DEST` | scratch for the FP8 shards and the size lists |
| `HF_TOKEN` | `~/.cache/huggingface/token` | needed only for gated repositories |

## Documents

- [docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md) - every measurement and every setting on this machine.
- [docs/ds4/V41_PIPELINE_OVERLAP.md](docs/ds4/V41_PIPELINE_OVERLAP.md) - the per-layer host handshake, measured and closed.
- [docs/ds4/V41_AMD_PEER_TIER.md](docs/ds4/V41_AMD_PEER_TIER.md) - the second-card design that was measured and dropped.
- [docs/upstream/README.md](docs/upstream/README.md) - the upstream README (NVIDIA, 512K, V4-Flash), verbatim.
- [local/ds41-local.json](local/ds41-local.json) - the configuration these numbers were measured with.

## The routing bias, and why the answers were wrong

The engine's expert selection adds up to +1.0 / -0.6 to each expert's score depending on where that expert's bytes
live (VRAM, RAM arena or the file tier), and it does so on every generated token - not during the prompt. It is not
part of the model: it trades fidelity for speed by keeping hot experts resident. Measured here on 2026-10-17, same
prompt, greedy, 600 tokens:

| route bias | what it answered | decode |
|---|---|---|
| default (+1.0 / -0.6) | "the Inferno is divided into nine circles (stage)", "the Purgatorio into seven stage" - the nine circles never named | 13.17 tok/s |
| 0 / 0 (the model's routing bit for bit) | the nine circles named, Limbo first | 2.97 tok/s |
| **`STRATA_RB_MARGIN=1`** (the bias only among near-ties) | the same correct answer | **11.12 tok/s** |

So `local/ds41-local.json` sets `STRATA_RB_MARGIN=1`: the bias can only reorder experts whose exact scores are within
one point of the cutoff, which cannot change the selection in practice, and it costs about 16% of the decode speed
instead of 77%. The engine's own comments describe the unrestricted bias as "-14 pt API top-1, repetition loops", KL
0.10-0.15, and this is exactly what the facts-and-loops symptom was: the model reads the question exactly, then every
answer token routes on a set that depends on the machine's memory layout. No other change was needed - not the
weights, not the quantisation, not the sampler.

Re-measured on 2026-10-17 while bringing up the ds4 Q2 weights (next section but one): `STRATA_RB_MARGIN=1` did not
reproduce that answer in either the one-shot greedy run or the server run - both returned invented circles - while
`--route-bias 0 --route-bias-file 0` named all nine. The margin run was fast (11.99 tok/s, 95.7% VRAM hits), so the
margin is not a routing no-op in this build; check an answer with `--route-bias 0`, not with the margin, until that
is explained.

## Making the thinking work (2026-10-17)

With the thinking on, this quant answered nothing: it reasoned for 94-270 words, never emitted the close token and
`content` came back empty (and in the web chat it span in loops like "Aspetta, penso... Conclusione finale
definitiva"). The cause was not a bug: `reasoning_budget_tokens` was **2048**, so the platform's wrap-up never fired
before the request hit `max_tokens`. The budget is the mechanism that closes the thinking, and it has to be **below
the model's typical reasoning length**:

| budget | result |
|---|---|
| 2048 (before) | thinking 94-270 words, **content empty**, `finish=length`/`cancel` |
| 384 | content arrives; sometimes the deliberation still leaks into the answer ("Wait, let's be precise") |
| **192** (now) | `finish=stop`, thinking 114 words, **content clean, no leakage**, and the nine circles named correctly: Limbo, Lussuriosi, Golosi, Avari/Prodighi, Irascibili, Eretici, Violenti, Malebolge, Nono |

`render()` decides the mode: `mode = "chat" if kw.get("enable_thinking") is False else "thinking"`, so the thinking is
on unless a client asks for it off; the budget is what makes it finish. The sampler was never the problem: the
penalties are in the config (`repetition_penalty 1.1`, `penalty_last_n 256`) and the repeated-target loops are 0% in
both acceptance tests (the `local/prompts/divina.i32` prompt, `max_tokens` 700, no `reasoning_effort` sent).

Two notes: the "raccontami la Divina Commedia" answer needs more than 700 tokens (`finish=length`), so raise
`max_tokens` for long prose; and the dates it gives are approximations (1305-1321, 1300-1310) - that is the 2.25 bpw
quant, not the serving path.

## The ds4 Q2 weights

This branch also runs antirez/ds4's `DeepSeek-V4.1-Flash-Q2.gguf` (365,713,686,528 B, architecture `deepseek41`,
1046 tensors) on the same `tools/v41` engine. Its experts are IQ2_XXS (gate and up) and Q2_K (down); the two 101.4 GB
Engram tables travel inside the gguf as `blk.1.engram_embd` and `blk.14.engram_embd` and are skipped by the dense
loader - the engine reads the same `F8_E4M3` weight / `F8_E8M0` scale rows from `models/engram/`, exactly as the GSQ
run does. Same command as the GSQ, only `-m` changes:

```
./build-hip-fused/ds41_generate -m "/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf" \
  --ids-file local/prompts/divina.i32 -n 200 --backend cuda --experts gpu --arena-gib 75 --ctx 262144 --kv-q8 \
  --temp 0 --threads 24 --rank models/V41-GSQ/rank.bin --engram models/engram/layers14:models/engram/layers1
```

### What had to be fixed

The expert tier sized the blob correctly from the start - 9,953,280 B = gate 1320 B/row x 2304 rows (IQ2_XXS at
5120 = 20 blocks of 66 B) x 2 + down 756 B/row x 5120 rows (Q2_K at 2304 = 9 blocks of 84 B) - and still refused to
start:

```
tier init: ExpertCache: 152882380800 slots x 1 B = 142.38 GiB, but only 14.94 GiB of VRAM is free
```

The `1 B` is `ExpertCache::open_sized`'s uniform-path parameter, not the blob. 152,882,380,800 B is 15,360 slots
(40 layers x 384 experts) x 9.49 MiB: with an automatic `--slots`, the cache had been asked for the WHOLE expert set
because the free-VRAM budget (`slot_gib`) was only honoured for a per-layer/per-class geometry. Two fixes in
`tools/v41/engine/`:

1. **`ds41_moe.cpp`, `uniform_slots`**: an automatic slot budget now caps the slot COUNT of a uniform geometry too.
   The Q2 opens 1,481-1,518 slots out of the 14.08 GiB free after the dense half instead of asking for 142 GiB. An
   explicit `--slots N` is unchanged (`slot_gib` is 0 then).
2. **`ds41_generate.cpp`, `mc.defer_cache = defer_vram`** (the ds4 engine's `defer_seed`, which the v41 fork never
   set): the uniform cache opens at the seed, after the prompt. Opened at init it holds ~14 GiB, so the first serve
   request measures 0.87 GiB free, declares "no room", releases it - and a uniform geometry has no other seed path
   that opens it again, so the decode ran with no VRAM slots (the server decode ran at 6.2 tok/s; the one-shot
   before the fix was 7.89).

### Measured

Same machine as the rest of this README (RX 7900 XTX 24 GiB, 122 GiB RAM), `--kv-q8`, `--ctx 262144`, arena 75 GiB,
24 threads, the GSQ's `rank.bin` (the only profile that existed when this was measured; the Q2's own rank is below),
the GSQ Engram tables, `local/prompts/divina.i32`, 200 new tokens, greedy:

| | Q2 | GSQ (same binary, same settings) |
|---|---|---|
| decode | **9.01 tok/s** (22.07 s) | **10.81 tok/s** (18.41 s) |
| prompt (29 tokens) | 12.67 s | 5.17 s |
| VRAM slots / hits | 1,481 / 97.0% | 1,640 / 95.2% |
| expert blob | 9,953,280 B | 6.7-9.5 MB, 9 classes |

So the Q2 is about 17% slower than the GSQ at the same settings. Its first run, before the deferred seed, was 7.89
tok/s. The arena seed took 122-142 s (75 GiB of expert blobs read from the external SSD, page cache cold).

**The nine circles.** The acceptance prompt is the chat-encoded `elenca i nove cerchi dell'Inferno, solo i nomi`
(`local/prompts/circles.i32`). With route bias 0 (the model's own routing) the Q2 names all nine - Limbo, Lussuria,
Gola, Avidità, Ira, Eresia, Violenza, Frode, Tradimento - in 47 tokens at **0.97 tok/s**; through the server (thinking
budget 192, `temp 0.8` / `top_p 0.95` / `rep 1.1`) it is `finish=stop`, 263 tokens, the same nine names, at 1.1 tok/s.
With the default route bias (+1.0/-0.6, the speed lever) it invents circles ("Cerchio della Terra", "Cerchio del
Cielo") and the GSQ does the same in the same harness ("Vestibolo" x8) - not a Q2 problem. `STRATA_RB_MARGIN=1`, which
the section above records as restoring the GSQ's answer, did not do so in these runs: with it the GSQ was fast
(11.99 tok/s, 95.7% VRAM hits) and wrong ("Inferno, Antinferno, Centro, Dis..."), while only route bias 0 named all
nine (3.19 tok/s, 45.5% hits). The Q2 one-shot was not run with the margin; its server run has it, and there too
the nine were not named.

Exact routing costs the Q2 9x its biased speed (9.01 -> 0.97 tok/s) and the GSQ 3.4x (10.81 -> 3.19), because with
the GSQ's rank the Q2's own routing hits the VRAM cache 45.7% of the time and reads ~86 experts per token from the
file tier (9.49 MiB each, about 1 GB/s of SSD). A `rank.bin` measured on the Q2 itself is the missing piece, and the
next subsection is that measurement; there is no Q2-matched DSpark drafter either. The GSQ's MTP sidecar does load
and draft for the Q2 (1.47 accepted drafts/pass, 2.47 tokens/pass) but exact-routing decode got slower, 0.82 against
0.97 tok/s: the verify pass reads every token's experts, so a drafter cannot help an I/O-bound decode.

### The Q2's own `rank.bin` (2026-10-17)

`local/rank-q2.bin` is ranked from the Q2's own routing counts. `ds41_generate` grew a `--dump-routes DIR` flag: the
router's `ids` were already a graph output (`o_ids`) and already copied host-ward for the expert tier, so counting
them changes no arithmetic; the run writes `DIR/LNN.json` with `experts[e].n` = tokens routed to `e`, which is exactly
what `tools/v41/engine/v41_rank.py` reads. On the Q2 no `--map` is needed: the tool's default blob,
2 x IQ2_XXS (1320 B/row at 5120) + Q2_K (756 B/row at 2304) = 9,953,280 B, is the Q2's expert to the byte, so every
expert is ranked by routed count alone.

The corpus is 8,154 prompt tokens of mixed text (`local/calib-q2/corpus.i32` = 2,500 tokens of
`docs/DETAILS.md` + 2,500 of `ds41_generate.cpp` + 800 of `README.md` + the 656-token `long-it` + `divina` prompts +
the 1,669-token Italian answer in `local/v41-text.log`) and the acceptance prompt with its answer (60 tokens, replayed
with `STRATA_FORCE_IDS` so the counts are exactly the routing that answer takes) = **8,296 tokens, 1,991,040
(layer, expert) selections, 129.6 a pair**. The counts are in `local/calib-q2/stats-{corpus,circles}`, merged by
`local/calib-q2/merge_stats.py`. A first attempt on 6,581 tokens built from the prompts already in `local/prompts/`
was 88% one repeated code-review prompt: it covered 99.0% of *that* corpus at 80 GiB but only 85.9% of the
nine-circles routing and decoded at 2.10 tok/s - a repeat-heavy corpus overfits, so the final rank is the diverse one.

Coverage printed by `v41_rank.py`: the first 8 GiB of experts cover 38.0% of the calibration routing, 16 GiB 52.4%,
32 GiB 70.3%, 48 GiB 81.7%, 64 GiB 89.4%, 80 GiB 94.5%, 100 GiB 98.2%. On the nine-circles routing (the 79 tokens of
the prompt and its answer, so in-sample) the top 1,526 experts - what the VRAM cache seeds - cover 47.0%, the top
8,090 - the 75 GiB arena - 88.8%, the top 11,229 - 90 GiB of arena plus the VRAM slots - 96.6%.

Measured on the usual machine, one-shot greedy, exact routing (`--route-bias 0 --route-bias-file 0`), `--ctx 8192`,
24 threads, the nine-circles prompt:

| resident set | decode | file-tier reads |
|---|---|---|
| GSQ `rank.bin`, 75 GiB arena | 0.97 tok/s | ~86/token (36%) |
| Q2 rank, 75 GiB arena (arena duplicates the VRAM head) | 3.13 tok/s | 1,203 / 60 tokens = 20/token |
| Q2 rank, `--arena-skip-vram`, 75 GiB arena | 3.49 tok/s | 683 / 47 tokens = 14.5/token |
| Q2 rank, `--arena-skip-vram`, 90 GiB arena (`memguard.sh 110 82`) | **4.53 tok/s** | 443 / 60 tokens = 7.4/token |

`--arena-skip-vram` is new and opt-in: the arena holds the ranking *past* the VRAM cache's head instead of holding
the same experts again, so the two tiers cover distinct experts (the tier already demotes an evicted VRAM expert into
the slot its replacement vacates; that mode just had no flag). Every configuration named all nine circles; the
wording is the 47-token "Lussuria, Gola, Avidità..." list in the 75 GiB `--arena-skip-vram` run and the 60-token
"Lussuriosi, Golosi, Avari e Prodighi..." list in the other two, a near-tie difference, both correct. The command
behind 4.53 tok/s:

```
bash tools/ds4/memguard.sh 110 82 -- ./build-hip-fused/ds41_generate \
  -m "/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf" \
  --ids-file local/prompts/circles.i32 -n 64 --temp 0 --backend cuda --experts gpu \
  --arena-gib 90 --ctx 8192 --kv-q8 --threads 24 --rank local/rank-q2.bin \
  --engram models/engram/layers14:models/engram/layers1 --route-bias 0 --route-bias-file 0 --arena-skip-vram
```

The 90 GiB arena leaves ~20 GiB of RAM free on the 122 GiB machine (75 GiB with `memguard.sh 80 82` is the safe
default, 3.49 tok/s). What is still on the table: the file tier is 3.1% of the decode's lookups and 4.5 s of the
13.0 s at 90 GiB; the measured non-file cost is ~0.13 s/token, so even a zero file tier would be ~7.4-7.7 tok/s, not
9.01. Reaching 9 needs more of the 142.4 GiB expert set resident than the 104 GiB this machine can hold (90 GiB arena
+ 14 GiB VRAM) or a faster file tier (the reads are serial 9.5 MiB preads at ~0.93 GB/s). Weighting the acceptance
prompt's own counts 16-64x does reach 97-100% coverage on it, but that is a rank for that prompt, not for the model,
so it was not kept.

**Prose.** At exact routing both models are coherent and close: 300 greedy tokens of `raccontami la divina
commedia` give the three cantiche, Virgilio and Beatrice, Lucifero at the centre, the seven cornici, the nine heavens
and the Empireo; the Q2 says 1307-1321, the GSQ 1304-1321, and both are cut mid-sentence at 300 tokens. With the
default route bias both degrade - the Q2 dates the poem 1300-1307, writes "immaginato non morto" and repeats "È un
viaggio che parte da un Inferno" - so in these samples the prose quality follows the routing much more than the
2.06 bpw.

The new pieces are `local/ds41-local-q2.json` (the chat config with the Q2 and no drafter),
`local/ds41-local-q2-exact.json` (the same with `--route-bias 0 --route-bias-file 0`) and
`local/ds41-local-gsq-nodspark.json` (the GSQ reference without DSpark).
