<h1 align="center">ds41e - DeepSeek-V4.1-Flash-1017 on RX 7900 XTX</h1>

<p align="center"><b>DeepSeek-V4.1-Flash-1017 (the published GSQ-RCO IQ2_XXS pack) with exact routing on one or two RX 7900 XTX, its experts across VRAM, pinned RAM and NVMe</b><br>
Linux - ROCm/HIP - MIT</p>

A C++/HIP inference engine for DeepSeek-V4.1-Flash and an OpenAI- and Anthropic-compatible server with a web chat
and monitor. The model is DeepSeek-V4.1-Flash-1017 as the [ISTA-DASLab GSQ-RCO pack](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
(IQ2_XXS/IQ2_XS/IQ1_M experts, four shards, 138.2 GB), read through the geometry overlay `DS41_META` because its GGUF
carries llama.cpp metadata instead of this engine's `deepseek41.*` keys; every expert the model routes to is computed (no route bias),
wherever it is: card 0's VRAM, the second card, an 80 GiB pinned RAM arena, or read from the NVMe.

## Performance

Two RX 7900 XTX, Ryzen 9 9900X, 122 GiB RAM, PCIe 5.0 NVMe, ROCm 7.14, the default configuration's flags with
power-tune on, the GSQ-RCO pack (2026-10-26, one-shot runs, 3 each):

| | |
| --- | ---: |
| decode, Promessi Sposi (300 tokens) | 23.2-23.7 tok/s |
| decode at 17K context (128 tokens) | 18.4-19.2 tok/s |
| prompt from scratch, 8K / 17K / 25.5K tokens | 371-414 / 433-459 / 458-484 tok/s |

On antirez/ds4's Q2, the pack this box ran until 2026-10-25, the same prompts read 314-332 / 360-366 / 385-396 tok/s,
and through the server: 18.5-20.7 tok/s on short answers, 19.6 at ~25K context, 16.1-17.4 in an agent session at
~120K, a +497-token agent turn on 121K context in 7.5 s (2026-10-20/21; not yet measured through the server on the
GSQ-RCO pack).

Quality on Lucebox's held-out set (8 x 1024 tokens, exact routing): ppl 10.37 with the checkpoint's FP8 Engram rows,
10.33 with the pack's own iq4_nl table (KL 0.026 between the two). The FP8 1017 itself reads 8.56 and antirez/ds4's
Q2 9.78 (measured 2026-10-20 with a teacher file that is no longer on this box, so the RCO pack's teacher agreement is
not measured). Every number and how it was measured: [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md).

## Requirements

| | |
| --- | --- |
| OS, GPU stack | Linux, ROCm (7.14 here) with `hipcc`, hipBLAS and `/opt/rocm/lib/llvm/bin/clang++` |
| GPU | RX 7900 XT / XTX (`gfx1100`); a second 24 GB card is used as an expert tier when present |
| RAM | 128 GB class (the engine holds ~94 GB with the 80 GiB arena) |
| CPU | x86-64 with AVX2 (AVX-512 used when present) |
| disk | ~350 GB on an NVMe for `models/` (138 GB pack in four shards + 111 GB Engram GGUF + 0.9 GiB vision encoder); the checkpoint's own FP8 Engram rows instead of that GGUF are 203 GB |
| tools | `cmake` >= 3.24, gcc, Python 3, curl, `aria2c` (faster downloads) |

## Quick start

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # the server's Python packages
local/fetch-rco-1017.sh                                               # everything the engine needs (~249 GB, resumable)
./build.sh                                                            # ggml and the engine (tens of minutes the first time)
./run.sh                                                              # the server on 127.0.0.1:8150
```

- `local/fetch-rco-1017.sh` downloads the four shards and the pack's Engram IQ4_NL GGUF with `aria2c` (4 connections,
  `CONN`), checks each sha256, and writes the geometry overlay `local/rco-iq2xxs-meta/meta.gguf` with
  `tools/v41/ds41_pack_meta.py` (its reference geometry is `local/rco-iq2xxs-meta/geometry-keys.json`, 8 KB, in the
  repo) plus the rank byte map.
- The engine reads the Engram rows in either form, and `local/fetch-rco-1017.sh` brings the one that comes with the pack
  (iq4_nl, 144 bytes a row). The checkpoint's native FP8 rows (264 bytes a row) are the other: `local/fetch-model.sh`
  downloads [antirez/ds4](https://github.com/antirez/ds4)'s Q2 pack and `local/ds4-engram-extract.py` cuts the two tables
  out of it into `models/engram-1017` (101 GB a layer) - that pack is only their source, it can go once they are out
  (re-extractable with `LAYERS=14 local/ds4-engram-extract.py <pack> models/engram-1017`). A configuration naming tables
  this machine does not have still runs: the engine falls back to the pack's own table beside the pack and prints it.
  Both forms measured: docs/ds4/V41_EXACT_FAST.md.
- `./build.sh` builds the vendored ggml (`third_party/llama.cpp/ggml`) into `third_party/llama.cpp/build-hip`, then the
  engine into `build-hip-fused`; `ROCM`, `GFX`, `JOBS`, `REBUILD_GGML=1` from the environment.
- `./run.sh` uses cards 0 and 1; `./run.sh --gpu 0` (or `--gpu 1`) one card, `--gpu 0,1` both. The number of cards
  picks the configuration: `local/ds41-local.json` (two cards) or `local/ds41-local-1gpu.json` (one);
  `--config FILE` uses that file. Other arguments go to the server (`./run.sh --port 8151`). `./stop.sh` stops a
  server started in the background (as Ctrl+C would; `--force` kills it if it does not end in 60 s). Loading takes about a minute; the web chat and monitor
  are at `http://127.0.0.1:8150/`, the API at `/v1`. Never expose the server beyond `127.0.0.1` without `--api-key`.

Reasoning effort follows DeepSeek's encoder: `low` (50), `high` (75, default), `max` (100), a number 1-100, or `none`;
as `reasoning_effort`, `reasoning.effort` (OpenAI) or `output_config.effort` / `thinking` (Anthropic).
The level goes into the model's prompt (`Reasoning Effort: N`); the server also caps the thinking by level
(`reasoning_budget_by_effort` in the configuration: low 1024 tokens, high 4096, max none - at the cap it closes the
thinking and the model answers). A request's own `reasoning_budget_tokens` wins.

## Images (not with this pack)

Images need the model's image routing bias, `blk.N.exp_probs_b_vl.bias`: the GSQ-RCO pack carries none, so the vision
configurations were removed on 2026-10-25 and `--vision` says so when it is asked for. The encoder itself
(`models/vision-1017`, 0.9 GiB, built by `local/fetch-model.sh`) stays; a pack that holds the bias gets a vision
configuration back. Measurements of the path, on the pack that had it: [docs/ds4/V41_VISION.md](docs/ds4/V41_VISION.md).

The encoder runs in the engine on the second card (0.2-5 s an image, exact against DeepSeek's code); a picture
costs 189-1014 prompt tokens. The September encoder (antirez's `DeepSeek-V4.1-Flash-Vision.gguf`) does not match the
1017 model. Details and measurements: [docs/ds4/V41_VISION.md](docs/ds4/V41_VISION.md).

## Conversations

The engine keeps every conversation a request does not continue in a 4 GiB pool of pinned RAM
(`--conversation-cache-mib`, ~380 MB for a 200K-token session) and resumes the one a request continues: two
conversations taking turns, 17K and 25.5K tokens, went from 36.9 to 6.9 s a turn. They are also saved in
`local/conv-cache-rco` (`--conversation-disk`, up to 32 GiB), so a restarted server resumes a 25.7K-token session in 6.5 s
instead of reading it again (~65 s). The Monitor shows the folder and its size, and a button clears them all
([docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md), "The conversation pool", "Conversations on disk").

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
counts what the router picks (`local/routes-rco`); `local/update-rank.sh` builds a new rank from those counts, measures
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
