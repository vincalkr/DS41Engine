# AGENTS.md

ds41e runs DeepSeek-V4.1-Flash-1017 (antirez/ds4's Q2) on an AMD RX 7900 XTX (ROCm/HIP), a second 24 GB card as an expert tier, system RAM
and NVMe: a C++/HIP engine (`tools/v41/engine`, with the few Strata libraries it links in `src/` and `include/`), an
OpenAI- and Anthropic-compatible server with a web app (`serve/`, started through `tools/v41/serve/serve_ds41.py` by
`./run.sh`), and the configurations, tools and measurements in `local/`, `tools/v41/` and `docs/ds4/`.
It holds DeepSeek only: the Qwen engine, its installer and the other models' code were removed on 2026-10-18.

## Running it

- `./run.sh` starts the server on `127.0.0.1:8150` on cards 0 and 1 with `local/ds41-local-vision.json`
  (`local/ds41-local.json` without the vision encoder); `--gpu 0` / `--gpu 1` one card
  (`local/ds41-local-1gpu-vision.json`, or `-1gpu.json`), `--config FILE` any other. Never expose it beyond
  `127.0.0.1` without `--api-key`.
- The model files live in `models/` inside this folder (`local/fetch-model.sh` puts them there); keep it that way.
  The default model (2026-10-20) is DeepSeek-V4.1-Flash-1017 as antirez/ds4's Q2 (`models/ds4-1017`, Engram in
  `models/engram-1017`). The September GSQ it replaced is backed up on the USB SSD
  (`/run/media/vincenzo/SSD NVME SAMSUNG/models`); its config and fetcher are in the git history (before 2026-10-21).
- Keep the folder name `ds41e`: a sibling Strata install's setup moves the `models/` of every sibling folder named
  `Strata*` into its `Strata-data`.

## Working on the code

- From zero: `.venv` from `requirements.txt`, `local/fetch-model.sh`, `./build.sh`, `./run.sh` (README,
  "Quick start"). The build is HIP only: ggml from the vendored `third_party/llama.cpp/ggml` (its changes also kept
  as patches in `tools/v41/ggml/`) into `third_party/llama.cpp/build-hip`, then the engine into `build-hip-fused`.
- Server tests: `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .` (no GPU or model needed).
- How the engine works and every measured number: `docs/ds4/V41_EXACT_FAST.md`, `docs/ds4/AMD_HIP_NOTES.md`,
  `HANDOVER.md`.
- The engine is not bit-reproducible run to run (an expert's kernel depends on where it is served from, which
  depends on timing): compare quality with repeated runs or the Lucebox set in `local/eval-lucebox/`, not with a diff.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
- A new feature is opt-in and the default path stays as it was; change a default only where it measured better, and
  write down the numbers.
- Commit messages in English.
