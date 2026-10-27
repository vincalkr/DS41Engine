# Handover

Where ds41e stands and what is open. How it runs and every measured number: [README](README.md),
[docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md), [docs/ds4/AMD_HIP_NOTES.md](docs/ds4/AMD_HIP_NOTES.md).

## State (2026-10-27)

- **Model:** DeepSeek-V4.1-Flash-1017 as the ISTA-DASLab GSQ-RCO IQ2_XXS pack (`models/rco-iq2xxs`, four shards,
  138.2 GB), read through the geometry overlay `local/rco-iq2xxs-meta/meta.gguf` (`DS41_META`). Exact routing.
- **Engram:** the checkpoint's FP8 rows in `models/engram-1017` (203 GB); the pack's own iq4_nl table is the
  measured-equivalent alternative and the fallback (`local/fetch-rco-1017.sh` brings it).
- **Machine:** two RX 7900 XTX (card 1 on a PCIe 4.0 x2 uplink), Ryzen 9 9900X, 122 GiB RAM, PCIe 5.0 NVMe, ROCm 7.14.
  The cards' 305 W cap, -200 mV and stock memory clock are deliberate; power-tune (NVMe APST and CPU C3 off, the
  `performance` governor) runs with the server.
- **Server:** `./run.sh` / `./stop.sh`, `127.0.0.1:8150`, the conversation pool (4 GiB) and disk store
  (`local/conv-cache-rco`, 32 GiB), reasoning effort none / low / high / max with a thinking cap per level.
- **Speed:** decode ~23 tok/s on short prompts, ~18-19 at 17K, 16-20 in agent sessions at 120-190K context; prompts
  from scratch 370-490 tok/s. **Quality:** Lucebox ppl 10.37 (the FP8 checkpoint: 8.56).

## Fixed recently

- 2026-10-27: the prompt MMQ read up to 127 ids past its buffer (a GPU page fault on a 3,918-token batch);
  `local/bench-v41/repro-mmq-crash.py` reproduces the shape.
- 2026-10-27: a prompt that re-rendered the last few tokens of the live conversation was read again from zero; it now
  resumes exactly (`STRATA_SERVE_LIVE_PARTIAL`).

## Open

- **Where the time goes:** the CPU tier is at the RAM's bandwidth (47 GB/s of 61), and card 0 waits ~6 ms a token in
  40 per-layer host round trips. Faster RAM (`local/ram-bench.sh` compares BIOS settings) is the hardware lever;
  routing on the card was estimated at ~5% on short prompts and ~0 at long ones and not built.
- **Speculative decoding:** a verify pass costs about as much as decoding its tokens here; lookup (prompt) drafting
  for agent sessions that copy code is the one variant not tried.
- **The rank:** `local/update-rank.sh` measures a rank built from the chat's routing on code and documentation prompts
  and applies it only if faster.
- **Images:** the pack has no VL router bias; a pack that has one would bring them back.

## Constraints

- Never expose the server beyond `127.0.0.1` without `--api-key`.
- A new feature is opt-in; a default changes only where it measured better, with the numbers written down.
- The engine is not bit-reproducible run to run (an expert's kernel depends on the tier that serves it): compare
  quality with Lucebox or repeated runs, not with a diff.
- Commit messages in English.
