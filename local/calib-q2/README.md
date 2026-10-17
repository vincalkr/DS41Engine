# local/calib-q2/ — the routing counts behind `local/rank-q2.bin`

`rank-q2.bin` is `tools/v41/engine/v41_rank.py`'s order over the Q2's own routing counts, collected with
`ds41_generate --dump-routes` (see the README's "The Q2's own rank.bin" and `docs/ds4/DS4_Q2_STATUS.md`).

| file | what |
|---|---|
| `corpus.i32` | 8,154 prompt tokens of mixed text: `docs/DETAILS.md` (2,500) + `ds41_generate.cpp` (2,500) + `README.md` (800) + `local/prompts/long-it.i32` (656) + `divina.i32` (29) + `local/v41-text.log` (1,669) |
| `circles-answer.ids` | the 60 tokens the Q2 answered to `local/prompts/circles.i32` at exact routing (one id per line) |
| `stats-corpus/` | `LNN.json` from the corpus run (8,154 prompt tokens + a 64-token continuation) |
| `stats-circles/` | `LNN.json` from the circles prompt + `circles-answer.ids` replayed with `STRATA_FORCE_IDS` (79 tokens) |
| `stats-merged/` | the sum, the input `v41_rank.py` was run on (the only stats dir in git; the three others stay local) |
| `merge_stats.py` | sums any number of `stats-*` dirs into a new one |
| `run.log`, `cal-*.log`, `circles-*.log/out` | the runs' logs and token outputs |

The two runs (both exact routing, `--route-bias 0 --route-bias-file 0`, arena 75 GiB):

```
./build-hip-fused/ds41_generate -m "<Q2.gguf>" --ids-file local/calib-q2/corpus.i32 -n 64 --temp 0 \
  --backend cuda --experts gpu --arena-gib 75 --ctx 16384 --kv-q8 --threads 24 --rank local/rank-q2.bin \
  --engram models/engram/layers14:models/engram/layers1 --no-enc-skip --dump-routes local/calib-q2/stats-corpus

STRATA_FORCE_IDS=local/calib-q2/circles-answer.ids ./build-hip-fused/ds41_generate -m "<Q2.gguf>" \
  --ids-file local/prompts/circles.i32 -n 64 --temp 0 --backend cuda --experts gpu --arena-gib 75 --ctx 8192 \
  --kv-q8 --threads 24 --rank local/rank-q2.bin --engram models/engram/layers14:models/engram/layers1 \
  --route-bias 0 --route-bias-file 0 --dump-routes local/calib-q2/stats-circles

python3 local/calib-q2/merge_stats.py --out local/calib-q2/stats-merged \
  local/calib-q2/stats-corpus local/calib-q2/stats-circles
python3 tools/v41/engine/v41_rank.py --stats local/calib-q2/stats-merged --out local/rank-q2.bin
```

`stats/` is the first attempt (6,581 tokens assembled from `local/prompts/`, 88% one repeated code-review prompt).
It is kept only as the record of what did not work: repeat-heavy text overfits the rank.
