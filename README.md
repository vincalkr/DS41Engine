<h1 align="center">ds41e - DeepSeek-V4.1-Flash-1017 on RX 7900 XTX</h1>

<p align="center"><b>DeepSeek-V4.1-Flash-1017 (antirez/ds4's Q2) with exact routing on one or two RX 7900 XTX, its experts across VRAM, pinned RAM and NVMe</b><br>
Linux - ROCm/HIP - MIT</p>

A C++/HIP inference engine for DeepSeek-V4.1-Flash and an OpenAI- and Anthropic-compatible server with a web chat
and monitor. The model is DeepSeek-V4.1-Flash-1017 as [antirez/ds4](https://github.com/antirez/ds4)'s Q2
(`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`); every expert the model routes to is computed (no route bias),
wherever it is: card 0's VRAM, the second card, an 80 GiB pinned RAM arena, or read from the NVMe.

## Performance

Two RX 7900 XTX, Ryzen 9 9900X, 122 GiB RAM, PCIe 5.0 NVMe, ROCm 7.14, the default configuration with power-tune
on (2026-10-20/21):

| | |
| --- | ---: |
| decode, short answers (through the server) | 18.5-20.7 tok/s |
| decode at ~25K context (through the server) | 19.6 tok/s |
| decode in an agent session at ~120K context | 16.1-17.4 tok/s |
| prompt, 25.5K tokens from scratch | ~350-370 tok/s |
| prompt, a +497-token agent turn on 121K context | 7.5 s |

Quality against the 1017 FP8 on Lucebox's held-out set (8 x 1024 tokens): ppl 9.78 (FP8 8.56), top-1 agreement 78.8%,
KL 0.269. Every number and how it was measured: [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md).

## Requirements

| | |
| --- | --- |
| OS, GPU stack | Linux, ROCm (7.14 here) with `hipcc`, hipBLAS and `/opt/rocm/lib/llvm/bin/clang++` |
| GPU | RX 7900 XT / XTX (`gfx1100`); a second 24 GB card is used as an expert tier when present |
| RAM | 128 GB class (the engine holds ~94 GB with the 80 GiB arena) |
| CPU | x86-64 with AVX2 (AVX-512 used when present) |
| disk | ~570 GB on an NVMe for `models/` (366 GB GGUF + 203 GB Engram tables + 0.9 GiB vision encoder) |
| tools | `cmake` >= 3.24, gcc, Python 3, curl, `aria2c` (faster downloads) |

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # the server's Python packages
local/fetch-model.sh                                                  # the model into models/ (~570 GB, resumable)
./build.sh                                                            # ggml and the engine (tens of minutes the first time)
./run.sh                                                              # the server on 127.0.0.1:8150
```

- `local/fetch-model.sh` downloads the GGUF with `aria2c` (4 connections, `CONN`), checks its sha256, extracts the two
  Engram tables from it, fetches the tokenizer tables and builds the vision encoder; `DRY_RUN=1` prints the plan.
- `./build.sh` builds the vendored ggml (`third_party/llama.cpp/ggml`) into `third_party/llama.cpp/build-hip`, then the
  engine into `build-hip-fused`; `ROCM`, `GFX`, `JOBS`, `REBUILD_GGML=1` from the environment.
- `./run.sh` uses cards 0 and 1; `./run.sh --gpu 0` (or `--gpu 1`) one card, `--gpu 0,1` both. The number of cards
  picks the configuration: `local/ds41-local-vision.json` / `local/ds41-local-1gpu-vision.json` when the vision
  encoder is in `models/vision-1017`, otherwise `local/ds41-local.json` / `local/ds41-local-1gpu.json`;
  `--config FILE` uses that file. Other arguments go to the server (`./run.sh --port 8151`). Loading takes about a minute; the web chat and monitor
  are at `http://127.0.0.1:8150/`, the API at `/v1`. Never expose the server beyond `127.0.0.1` without `--api-key`.

Reasoning effort follows DeepSeek's encoder: `low` (50), `high` (75, default), `max` (100), a number 1-100, or `none`;
as `reasoning_effort`, `reasoning.effort` (OpenAI) or `output_config.effort` / `thinking` (Anthropic).

## Images (optional)

The model reads pictures (OpenAI `image_url`, Anthropic `image` blocks). `local/fetch-model.sh` builds the 1017 vision
encoder (0.9 GiB) into `models/vision-1017` and `./run.sh` starts the vision configuration; text runs as fast as
without it. `VISION=0 local/fetch-model.sh` leaves it out, and `./run.sh` then runs text only.

The encoder runs in the engine on the second card (0.2-5 s an image, exact against DeepSeek's code); a picture
costs 189-1014 prompt tokens. The September encoder (antirez's `DeepSeek-V4.1-Flash-Vision.gguf`) does not match the
1017 model. Details and measurements: [docs/ds4/V41_VISION.md](docs/ds4/V41_VISION.md).

## Conversations

The engine keeps every conversation a request does not continue in a 4 GiB pool of pinned RAM
(`--conversation-cache-mib`, ~380 MB for a 200K-token session) and resumes the one a request continues: two
conversations taking turns, 17K and 25.5K tokens, went from 36.9 to 6.9 s a turn
([docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md), "The conversation pool").

## Power-tune (optional, root)

Decode reads an expert from the NVMe every 10-20 ms; between reads the drive (APST) and the CPU cores (C3) fall
asleep and each read costs twice as much. `local/power-tune.sh on` keeps them awake and sets the `performance`
governor: through the server +11% on short answers and +21% at ~25K context. Install it once and `./run.sh` switches
it on at start and off when the server ends:

```sh
sudo local/install-power-tune.sh             # a root-owned copy and a sudo rule for it alone
```

`DS41_POWER_TUNE=0 ./run.sh` leaves the settings alone; `sudo local/install-power-tune.sh uninstall` removes it;
`sudo local/power-tune.sh on|off|show` by hand.

## Expert rank

Which experts live in VRAM and in the RAM arena follows a rank of the most used ones (`local/rank-q2.bin`). The server
counts what the router picks (`local/routes-1017`); `local/update-rank.sh` builds a new rank from those counts, measures
it against the one in use (~25 minutes, the server stopped meanwhile) and applies it only if it is faster.

## Another machine

`local/cfg.py` edits `local/ds41-local.json` (`python3 local/cfg.py show` prints it):

```sh
python3 local/cfg.py arg --threads 8         # the CPU's physical cores (12 here)
python3 local/cfg.py arg --arena-gib 50      # pinned RAM for experts: 80 GiB on 122 GiB here, keep ~40 GiB free beside it
```

The arena is the biggest lever after the cards: an expert that is in neither VRAM nor the arena is read from the
NVMe. `--vram-margin` (0.5 GiB) is the VRAM left free beside the expert cache; raise it if another program uses the
card. With one card use `./run.sh --gpu 0` (the two-card default stops at startup: "peer device 1 does not exist").

## Development

- Server tests (no GPU or model): `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .`
- Measurement scripts: `local/bench-v41/` (one-shot sweeps, profiles, the server benchmark `serve-bench.py`);
  quality: `local/eval-lucebox/`; `local/ram-bench.sh TAG [--decode]` compares RAM bandwidth and decode across BIOS
  settings (no arguments: the results so far).
- [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md) - the configuration and every measurement behind it;
  [docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md) - the HIP port; [HANDOVER.md](HANDOVER.md) - the history.
