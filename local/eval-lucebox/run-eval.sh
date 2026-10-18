#!/usr/bin/env bash
# run-eval.sh TAG [ENV=VAL ...] - Lucebox's 1017 held-out set (8 x 1024 tokens, eval/heldout_8x1024.npy) through
# this engine on the 1017 quant: one run per sequence with --ppl --dump-logits, then score.py compares every row with
# the FP8 1017 teacher's top-256 log-probs (eval/teacher_8x1024.npz) - the set lxe's README scores (ppl 11.66,
# top-1 71.5 %, KL 0.462 on his engine).  Exact routing; the extra ENV=VAL pairs go to the engine (e.g.
# STRATA_DENSE_REQUANT=q6_K).  The chat server must be stopped.
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; shift
E=local/eval-lucebox; O=$E/$TAG; mkdir -p "$O"
python3 -c "
import numpy as np
t = np.load('$E/heldout_8x1024.npy')
for i, s in enumerate(t): s.astype(np.int32).tofile('$O/seq%d.i32' % i)"
for i in 0 1 2 3 4 5 6 7; do
  env HIP_VISIBLE_DEVICES=0 STRATA_PF_NO_DUMMY=1 "$@" ./build-hip-fused/ds41_generate \
    -m models/V41-1017-2bit/DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf --ids-file "$O/seq$i.i32" -n 1 --temp 0 \
    --backend cuda --experts gpu --arena-gib 75 --ctx 4096 --kv-q8 --threads 12 --rank local/rank-1017.bin \
    --engram models/engram-1017 --route-bias 0 --route-bias-file 0 --ppl --dump-logits "$O/seq$i.f32" \
    > "$O/seq$i.out" 2> "$O/seq$i.log"
  echo "seq $i: $(grep '^PPL' "$O/seq$i.out")"
done
python3 $E/score.py "$O"
rm -f "$O"/seq*.f32
