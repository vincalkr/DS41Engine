# AGENTS.md

ds41e runs DeepSeek-V4.1-Flash on an AMD RX 7900 XTX (ROCm/HIP), a second 24 GB card as an expert tier, system RAM
and NVMe: a C++/HIP engine (`tools/v41/engine`, with the few Strata libraries it links in `src/` and `include/`), an
OpenAI- and Anthropic-compatible server with a web app (`serve/`, started through `tools/v41/serve/serve_ds41.py` by
`./run-v41-gsq.sh`), and the configurations, tools and measurements in `local/`, `tools/v41/` and `docs/ds4/`.
It holds DeepSeek only: the Qwen engine, its installer and the other models' code were removed on 2026-10-18.

## Running it

- `./run-v41-gsq.sh` starts the server on `127.0.0.1:8150` with `local/ds41-local.json` (two cards, the same as
  `local/ds41-local-2gpu.json`; `./run-v41-gsq-1gpu.sh` for one card). Never expose it beyond `127.0.0.1`
  without `--api-key`.
- The model files live in `models/` inside this folder (`local/fetch-model.sh` puts them there); keep it that way.
  The default model (2026-10-20) is DeepSeek-V4.1-Flash-1017 as antirez/ds4's Q2 (`models/ds4-1017`, Engram in
  `models/engram-1017`); the September GSQ is `local/ds41-local-gsq.json` (`local/fetch-model-gsq.sh`), backed up on
  the USB SSD (`/run/media/vincenzo/SSD NVME SAMSUNG/models`).
- Keep the folder name `ds41e`: a sibling Strata install's setup moves the `models/` of every sibling folder named
  `Strata*` into its `Strata-data`.

## Working on the code

- From zero: `.venv` from `requirements.txt`, `local/fetch-model.sh`, `./build.sh`, `./run-v41-gsq.sh` (README,
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
