<h1 align="center">ds41e - DeepSeek-V4.1-Flash on one RX 7900 XTX</h1>

<p align="center"><b>DeepSeek-V4.1-Flash (40 layers x 384 experts) on ONE RX 7900 XTX - 152 GiB of experts across VRAM, pinned RAM and NVMe, plus 189 GiB of Engram on disk</b><br>
The AMD port and tuning of <a href="https://github.com/Indras-Mirror/Strata-DSV4.1">Strata-DSV4.1</a>, itself built on <a href="https://github.com/Niko1221/Strata">Strata</a> - Linux - ROCm/HIP - MIT</p>

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

One RX 7900 XTX (a second one is installed but not used: behind its PCIe 4.0 x2 bridge it measured neither faster
nor slower), Ryzen 9 9900X (12 cores), 122 GiB RAM, a PCIe 5.0 NVMe, ROCm 7.14. On disk in `models/`: the GSQ quant
(152 GiB), its DSpark draft and tokenizer, and the two Engram tables (189 GiB).

| through the server, exact routing | |
| --- | ---: |
| decode, short chat prompts (Divina Commedia, nine circles, Promessi Sposi) | 16.5-18.6 tok/s |
| decode, thinking on | 16.4 tok/s |
| decode, agent sessions at 18-41K context | 10-12 tok/s |
| prompt, 25K tokens fresh | ~370 tok/s |
| prompt, a +300-token agent turn on 27K | 4.1-4.4 s |

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
| `STRATA_PRESTAGE_MIN=512` (upstream's lazy prestage) | +300-token turn 4.8-5.0 -> 4.1-4.4 s, long prompts unchanged |
| `STRATA_SERVE_STEP_MAX=96` (from 256): agent turns of 97-256 new tokens go to the layer-major batch | time to first token on a +171 / +256-token turn 9.7-9.8 -> 6.3-6.8 s / 12.9-14.4 -> 6.0-9.4 s |
| `--kv-q8`, `STRATA_PRESTAGE_GIB=1.5` | 8-bit KV: last-row top-5 identical (cosine 0.999); prestage +5.8% prompt |

## Where the time goes

Measured 2026-10-17 at 8K context (one-shot, before the Q6_K dense half): a token cost ~48.7 ms in the dense half
(attention, router) and ~31 ms in the expert tier; with every expert in VRAM the same decode ran at 20.7 tok/s, so the
misses cost ~42 ms a token. In the agent session of 2026-10-18 the VRAM
held ~1,785 of the 15,360 experts and 60-65% of lookups hit it; the rest were computed on the CPU (~85 a token) or read
from NVMe (~10 a token).

## Tried and reverted

Route bias (fast and wrong: invented circles), DSpark on Italian prose (~1 draft accepted a pass), the second card as
an expert tier (neutral at 25K context), the conversation-adaptive arena (more file reads on short turns), CPU pinning,
`GGML_CUDA_DISABLE_FUSION`, an async dense path, and DeepSeek-V4.1-Flash-1017 as lxe's 2-bit quant (faithful load,
worse answers: ppl 11.63 vs the FP8's 8.56). Details and numbers in [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md).

## Run it

```sh
./run-v41-gsq.sh                 # the server on 127.0.0.1:8150 (web chat at /, API at /v1), config local/ds41-local.json
./run-v41-gsq.sh --port 8151     # extra server arguments are passed straight through
```

It checks that the model and the Engram tables are in `models/`. Python needs pillow, psutil, regex and jinja2
(`python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2`). Never expose the server beyond
`127.0.0.1` without `--api-key`. Loading takes about a minute; then `curl -s http://127.0.0.1:8150/v1/models` answers.

## Getting the model files

`local/fetch-model.sh` builds the layout in `models/` (355 GB): the GSQ from
[IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
(16 parts, joined and sha256-checked by the repo's `reassemble.sh`) and the two Engram tables, extracted from shards 47
and 48 of [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8).
It is resumable and uses `aria2c` (4 connections per file by default, `CONN`); `DRY_RUN=1` prints the plan. Every
option is documented at the top of the script.

## Build

ggml once, from `third_party/llama.cpp` (a llama.cpp checkout, not tracked here, patched with `tools/v41/ggml/*.patch`),
then the engine (HIP only):

```sh
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-hip -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=OFF -DGPU_TARGETS=gfx1100
cmake --build third_party/llama.cpp/build-hip --target ggml -j

cmake -S . -B build-hip-fused -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_GGML_PREBUILT=$PWD/third_party/llama.cpp/build-hip \
  -DSTRATA_DSV41_HC_MIX=ON -DSTRATA_DSV41_FQ_OP=ON
cmake --build build-hip-fused -j      # ds41_generate, the engine's tests (test_ds41_*)
```

Server tests (no GPU or model needed): `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .`

## Documents

- [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md) - the current configuration and every measurement behind it.
- [docs/ds4/AMD_BRANCH_NOTES.md](docs/ds4/AMD_BRANCH_NOTES.md) - the earlier, longer README: the port, the route-bias
  investigation, the thinking budget, the ds4 Q2 weights.
- [docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md) - the HIP port's measurements and settings.
- [HANDOVER.md](HANDOVER.md) - the state of the work and how it got here.
- [docs/upstream/README.md](docs/upstream/README.md) - the upstream README (NVIDIA, 512K, V4-Flash), verbatim.
- [local/ds41-local.json](local/ds41-local.json) - the configuration these numbers were measured with.
