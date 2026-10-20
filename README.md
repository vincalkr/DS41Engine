<h1 align="center">ds41e - DeepSeek-V4.1-Flash on one RX 7900 XTX</h1>

<p align="center"><b>DeepSeek-V4.1-Flash (40 layers x 384 experts) on ONE RX 7900 XTX - 152 GiB of experts across VRAM, pinned RAM and NVMe, plus 189 GiB of Engram on disk</b><br>
Linux - ROCm/HIP - MIT</p>

## What this repository is

The DeepSeek-V4.1-Flash engine (`tools/v41/engine`), its OpenAI- and Anthropic-compatible server and web app
(`serve/`, `tools/v41/serve/serve_ds41.py`), and the configurations, tools and measurements for running it on **one**
RX 7900 XTX with the model's own routing (`--route-bias 0`: every expert the model picks, wherever it lives).
Everything that belonged to other models (Strata's Qwen engine, the SYCL port, the Qwen installer) was removed on
2026-10-18; of Strata's C++ only the 61 files the engine links are left (expert cache, pinned memory, CPU and HIP
expert kernels, the prompt path's MMQ).

Keep the folder name `ds41e`: another Strata install's setup moves the `models/` folder of every sibling folder whose
name starts with "Strata" into its shared `Strata-data`.

## The machine, and what it does

RX 7900 XTX (24 GB), Ryzen 9 9900X (12 cores), 122 GiB RAM, a PCIe 5.0 NVMe, ROCm 7.14. The numbers below are one
card; the default configuration also uses a second 24 GB card as an extra expert tier, and one card works too (see
"Quick start", step 3). On disk in `models/`: the GSQ quant
(152 GiB), its DSpark draft and tokenizer, and the two Engram tables (189 GiB).

| through the server, exact routing | |
| --- | ---: |
| decode, short chat prompts (Divina Commedia, nine circles, Promessi Sposi) | 16.5-18.6 tok/s |
| decode, thinking on | 16.4 tok/s |
| decode, agent sessions at 18-41K context | 10-12 tok/s |
| prompt, 25K tokens fresh | ~370 tok/s |
| prompt, a +300-token agent turn on 27K | 4.1-4.4 s |

Two cards (the default) with `local/power-tune.sh on`, one-shot, 5 runs each: 17.70 tok/s mean on Promessi Sposi,
**13.63 tok/s at 17K context** (one card, another series without power-tune: ~14.4 / ~8.6).

## What made it fast, each with what it was worth

All from [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md), where every row is measured.

| change | measured effect |
| --- | --- |
| a `rank.bin` from the model's own routing (`local/rank-gsq-mixed.bin`) | Divina Commedia 2.83-3.36 -> 5.86 tok/s (the GSQ's own rank covered 11.5% of the VRAM head) |
| `--arena-adapt`, then `STRATA_PF_NO_DUMMY=1` | Promessi Sposi 4.96 -> 5.88 -> 6.72 tok/s |
| `--threads 12` (the physical cores, not 24), then `--vram-margin 0.5` | Promessi Sposi 7.29 -> 10.39 -> 10.68 tok/s |
| `--pf-b 0` | Divina Commedia 11.16 -> 13.07 tok/s |
| ggml's HIP graphs on (they replay now) | 7,491-token prompt 37.5 -> 30.5 s, decode after it 10.41 -> 12.02 tok/s |
| dense half re-quantized Q8_0 -> Q6_K at load (`STRATA_DENSE_REQUANT=q6_K`) | all-VRAM ceiling 22.5 -> 25.7-26.7 tok/s, chat 16.5-18.6 tok/s; costs +0.9% ppl (measured on the 1017 quant) |
| `STRATA_PRESTAGE_MIN=512` (the lazy prestage) | +300-token turn 4.8-5.0 -> 4.1-4.4 s, long prompts unchanged |
| `STRATA_SERVE_STEP_MAX=96` (from 256): agent turns of 97-256 new tokens go to the layer-major batch | time to first token on a +171 / +256-token turn 9.7-9.8 -> 6.3-6.8 s / 12.9-14.4 -> 6.0-9.4 s |
| `--kv-q8`, `STRATA_PRESTAGE_GIB=1.5` | 8-bit KV: last-row top-5 identical (cosine 0.999); prestage +5.8% prompt |
| second card, its store disjoint from the RAM arena (`STRATA_PEER_DISJOINT`, `STRATA_ARENA_EVICT_PEER`) | NVMe reads -42%; short prose 14.07 -> 15.58, 17K 9.91 -> 10.82 tok/s |
| `local/power-tune.sh on` (root: NVMe APST off, CPU C3 off, `performance` governor) | a file read 2.0 -> 1.4 ms; 17K 12.11 -> 13.63 tok/s, every run fast (before: 2 of 5) |

## Where the time goes

Measured 2026-10-17 at 8K context (one-shot, before the Q6_K dense half): a token cost ~48.7 ms in the dense half
(attention, router) and ~31 ms in the expert tier; with every expert in VRAM the same decode ran at 20.7 tok/s, so the
misses cost ~42 ms a token. In the agent session of 2026-10-18 the VRAM
held ~1,785 of the 15,360 experts and 60-65% of lookups hit it; the rest were computed on the CPU (~85 a token) or read
from NVMe (~10 a token).

Two cards, 2026-10-20 (`STRATA_DEC_PROF`, rocprofv3): short prose ~28 ms dense half + ~31 ms expert tier a token, 17K
~37 + ~60. Card 0 is busy ~26 ms of a ~60 ms token and waits for the CPU tier once per layer. The CPU experts read
~47-50 of the RAM's 52-54 GB/s, so only faster RAM or fewer CPU misses speed them up. Decode's NVMe reads come 10-20 ms
apart, which let the drive and the CPU doze off between them: that is what `power-tune.sh` fixes.

## Tried and reverted

Route bias (fast and wrong: invented circles), DSpark on Italian prose (~1 draft accepted a pass), the
conversation-adaptive arena (more file reads on short turns), CPU pinning,
`GGML_CUDA_DISABLE_FUSION`, an async dense path, and DeepSeek-V4.1-Flash-1017 as lxe's 2-bit quant (faithful load,
worse answers: ppl 11.63 vs the FP8's 8.56). Measured on 2026-10-20 without a gain (kept opt-in or as scripts): ROCm
runtime knobs (`HSA_ENABLE_SDMA=0` -22%), the cards' clock level `high` (it holds the memory clock at 772 MHz: -4%),
zero-copy expert sums, persistent file readers, speculative file reads (4-5% of them used: -5 to -26%), a faster
Q6_K matvec kernel (`GGML_CUDA_Q6K_MMVQ=1`: faster alone, decode unchanged), prompt-lookup drafting (1.3-1.5 tokens a
step on agent sessions, simulated), and a smaller context (128K frees ~0.2 GiB of VRAM). Details and numbers in
[docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md).

## Quick start

Three steps: fetch the model, build the engine, start the server.

What it needs (measured on the machine above; other setups are untested):

| | |
| --- | --- |
| OS, GPU stack | Linux, ROCm (7.14 here) with `hipcc`, hipBLAS and `/opt/rocm/lib/llvm/bin/clang++` |
| GPU | an RDNA3 card the build targets (`gfx1100`: RX 7900 XT / XTX, 20-24 GB); others are not tested. The default config uses a second 24 GB card; one card needs `local/ds41-local-1gpu.json` |
| RAM | 128 GB class: the engine holds ~89 GB with the default 75 GiB arena (see "Adapting it" below) |
| CPU | x86-64 with AVX2 (AVX-512 kernels are used when present) |
| disk | 355 GB for `models/`, ~570 GB free at the download's peak (the Engram tables are extracted from two 101 GB shards); NVMe: experts that miss RAM are read from it during decode |
| tools | `cmake` >= 3.24, gcc, Python 3 with pillow, psutil, regex and jinja2; `aria2c` for fast downloads |

### 1. Get the model files

`local/fetch-model.sh` builds the layout in `models/` (355 GB): the GSQ from
[IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
(16 parts, joined and sha256-checked by the repo's `reassemble.sh`) and the two Engram tables, extracted from shards 47
and 48 of [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8).
It is resumable and uses `aria2c` (4 connections per file by default, `CONN`); `DRY_RUN=1` prints the plan. Every
option is documented at the top of the script.

### 2. Build

ggml once, from the copy in `third_party/llama.cpp/ggml` (with this repository's ggml changes already in it; see
`third_party/llama.cpp/VENDORED.md`), then the engine (HIP only). The ggml build compiles many HIP kernels: expect
tens of minutes.

```sh
cmake -S third_party/ggml-build -B third_party/llama.cpp/build-hip -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=OFF
cmake --build third_party/llama.cpp/build-hip -j

cmake -S . -B build-hip-fused -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_GGML_PREBUILT=$PWD/third_party/llama.cpp/build-hip \
  -DSTRATA_DSV41_HC_MIX=ON -DSTRATA_DSV41_FQ_OP=ON
cmake --build build-hip-fused -j      # ds41_generate, the engine's tests (test_ds41_*)
```

Server tests (no GPU or model needed): `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .`

### 3. Run it

```sh
./run-v41-gsq.sh                 # the server on 127.0.0.1:8150 (web chat at /, API at /v1), config local/ds41-local.json
./run-v41-gsq.sh --port 8151     # extra server arguments are passed straight through
./run-v41-gsq.sh --config local/ds41-local-1gpu.json    # one card
```

`local/ds41-local.json` (the default, the same as `local/ds41-local-2gpu.json`) uses two cards: `--peer-device 1`
gives card 1 its own 22 GiB expert store, and the experts that miss card 0's VRAM but sit there are computed on card 1
instead of the CPU. `STRATA_PEER_DISJOINT=1` fills that store with experts the RAM arena does not hold, and
`STRATA_ARENA_EVICT_PEER=1` makes the arena give up the experts card 1 holds first, so card 1 adds capacity instead of
copies. Measured 2026-10-20 (one-shot, 2 runs each): against two cards without the two envs, NVMe reads -42% and
decode 14.07 -> 15.58 tok/s on short prose, 9.91 -> 10.82 tok/s at 17K context; one card measured 14.4 and 8.6 tok/s on
the same prompts in another series. With a single card the default stops at startup ("peer device 1 does not
exist"): use `local/ds41-local-1gpu.json`. `--peer-device` takes one extra card.

Optional, with root: `sudo bash local/power-tune.sh on` keeps the NVMe and the CPUs out of their deep idle states
(APST, C3) and sets the `performance` governor. Decode reads an expert from the drive every 10-20 ms, and a drive or
core that dozed off in between doubled each read. Measured on this box (two cards, 5 runs each): 12.11 -> 13.63 tok/s
at 17K context, 17.17 -> 17.70 on short prose. It costs idle power and does not survive a reboot;
`sudo bash local/power-tune.sh off` restores the defaults.

`run-v41-gsq.sh` checks that the model and the Engram tables are in `models/`. Python needs pillow, psutil, regex and jinja2
(`python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2`). Never expose the server beyond
`127.0.0.1` without `--api-key`. Loading takes about a minute; then `curl -s http://127.0.0.1:8150/v1/models` answers.

### Adapting it to another machine

`local/ds41-local.json` is this machine's configuration; `local/cfg.py` edits it without breaking the argument list
(`python3 local/cfg.py show` prints it). The settings that depend on the hardware:

```sh
python3 local/cfg.py arg --threads 8         # the CPU's PHYSICAL cores (12 here: 12 measured 10.4 tok/s, 24 measured 7.3)
python3 local/cfg.py arg --arena-gib 50      # pinned RAM for experts: 75 GiB on 122 GiB here; keep ~45 GiB beside it
python3 local/cfg.py env HIP_VISIBLE_DEVICES=1,0  # the cards, the one on the fastest PCIe link first (it streams the prompt's experts)
```

The arena is the biggest lever after the card: every expert that is not in VRAM or the arena is read from NVMe during
decode. 75 GiB is what fits beside the engine's other ~14 GB, the server and the OS on this machine (80 and above
were slower or unstable); the 50 above is an example for a 96 GB machine, not a measured setting. `--vram-margin`
(0.5 GiB) is the VRAM left free beside the expert cache; raise it if another program uses the card.

## Documents

- [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md) - the current configuration and every measurement behind it.
- [docs/ds4/AMD_BRANCH_NOTES.md](docs/ds4/AMD_BRANCH_NOTES.md) - the earlier, longer README: the port, the route-bias
  investigation, the thinking budget, the ds4 Q2 weights.
- [docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md) - the HIP port's measurements and settings.
- [HANDOVER.md](HANDOVER.md) - the state of the work and how it got here.
- [local/ds41-local.json](local/ds41-local.json) - the configuration these numbers were measured with.
