# local/calib/ - the calibration corpus's routing counts

`stats-merged/LNN.json` holds, per layer, how often the router picked each of the 384 experts over a calibration corpus
of the 1017 checkpoint at exact routing: `corpus.i32` (8,154 tokens of mixed text: docs, C++ source, README, Italian
prose, chat log) plus a 64-token continuation, and `local/prompts/circles.i32` with the 60-token answer in
`circles-answer.ids` replayed through `STRATA_FORCE_IDS`. `local/rebuild-rank.sh` adds the chat's own counts
(`local/routes-rco`) to these and ranks the experts per byte (`tools/v41/engine/v41_rank.py`).

To collect counts again on the pack in use:

```
./build-hip-fused/ds41_generate -m models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf \
  --ids-file local/calib/corpus.i32 -n 64 --temp 0 --backend cuda --experts gpu --arena-gib 75 --ctx 16384 --kv-q8 \
  --threads 12 --rank local/rank.bin --engram models/engram-1017/layers14:models/engram-1017/layers1 --no-enc-skip \
  --route-bias 0 --route-bias-file 0 --dump-routes local/calib/stats-corpus      # with DS41_META set as in the config
python3 local/calib/merge_stats.py --out local/calib/stats-merged local/calib/stats-corpus local/calib/stats-circles
```
