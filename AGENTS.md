# AGENTS.md

ds41e runs DeepSeek-V4.1-Flash-1017 (the published GSQ-RCO IQ2_XXS pack) on an AMD RX 7900 XTX (ROCm/HIP), a second
24 GB card as an expert tier, system RAM and NVMe: a C++/HIP engine (`tools/v41/engine`, with the few Strata libraries
it links in `src/` and `include/`), an OpenAI- and Anthropic-compatible server with a web app (`serve/`, started
through `tools/v41/serve/serve_ds41.py` by `./run.sh`), and the configurations, tools and measurements in `local/`,
`tools/v41/` and `docs/ds4/`.
It holds DeepSeek only: the Qwen engine, its installer and the other models' code were removed on 2026-10-18.

## Running it

- `./run.sh` starts the server on `127.0.0.1:8150` on cards 0 and 1 with `local/ds41-local.json`; `--gpu 0` /
  `--gpu 1` one card (`local/ds41-local-1gpu.json`); `--config FILE` any other. Never expose it beyond
  `127.0.0.1` without `--api-key`. `./stop.sh` stops this folder's server (Ctrl+C's way, `--force` after 60 s).
  Images are off: the pack this box runs carries no `exp_probs_b_vl.bias` (the image routing bias), so the vision
  configurations were removed on 2026-10-25; a pack that has one gets its own configuration back.
- The model files live in `models/` inside this folder (`local/fetch-rco-1017.sh` puts them there); keep it that way.
  The default model (2026-10-25) is DeepSeek-V4.1-Flash-1017 as the ISTA-DASLab GSQ-RCO IQ2_XXS pack
  (`models/rco-iq2xxs`, four shards, 138.2 GB) read through the geometry overlay `local/rco-iq2xxs-meta/meta.gguf`
  (`DS41_META`, which `tools/v41/ds41_pack_meta.py` builds from the pack plus the 8 KB
  `local/rco-iq2xxs-meta/geometry-keys.json` - no reference pack needed). Its Engram rows come in two forms, both
  measured (docs/ds4/V41_EXACT_FAST.md): this box reads the checkpoint's own FP8 tables (`models/engram-1017`, 203 GB,
  264 bytes a row), cut out of antirez/ds4's Q2 pack by `local/ds4-engram-extract.py` - that 366 GB pack is only their
  source and is not kept here - and the pack's own iq4_nl table (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, 110.6 GB,
  144 bytes a row), which `local/fetch-rco-1017.sh` downloads and which the engine falls back to when a configuration
  names tables the machine does not have (it prints that it does).
- Keep the folder name `ds41e`: a sibling Strata install's setup moves the `models/` of every sibling folder named
  `Strata*` into its `Strata-data`.

## Working on the code

- From zero: `.venv` from `requirements.txt`, `local/fetch-rco-1017.sh` (or `local/fetch-model.sh` for the
  checkpoint's FP8 Engram rows), `./build.sh`, `./run.sh` (README,
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
