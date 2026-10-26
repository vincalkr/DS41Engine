#!/usr/bin/env bash
# run-eval.sh TAG [ENV=VAL ...] - Lucebox's 1017 held-out set (8 x 1024 tokens, eval/heldout_8x1024.npy) through
# this engine on the 1017 quant: one run per sequence with --ppl --dump-logits, then score.py compares every row with
# the FP8 1017 teacher's top-256 log-probs (teacher_8x1024.npz, off this box since 2026-10-26: without it score.py
# prints ppl and the run's own top-1 only) - the set lxe's README scores (ppl 11.66,
# top-1 71.5 %, KL 0.462 on his engine).  Exact routing; the extra ENV=VAL pairs go to the engine (e.g.
# STRATA_DENSE_REQUANT=q6_K).  The chat server must be stopped.  MODEL, RANK and ENGRAM pick the files (defaults: lxe's
# 2-bit quant as measured on 2026-10-18; the pack this box runs today is
# MODEL=models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf (its geometry comes from the
# DS41_META overlay, see local/rco-iq2xxs-meta/); antirez's Q2 that the recorded numbers were measured on is
# MODEL=models/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf, re-fetched by local/fetch-model.sh
# RANK=local/rank-q2.bin ENGRAM=models/rco-iq2xxs/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf). KEEP_LOGITS=1 keeps the
# logits and AGAINST=<run dir> scores this run against another run's logits (see score.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; shift
MODEL=${MODEL:-models/V41-1017-2bit/DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf}
RANK=${RANK:-local/rank-1017.bin}
export DS41_META=${DS41_META-local/rco-iq2xxs-meta/meta.gguf}   # the geometry overlay of the pack this box runs
# ENGRAM= (set but empty) runs without the Engram at all - that is how the table form is measured against none
ENGRAM=${ENGRAM-models/engram-1017/layers14:models/engram-1017/layers1}
E=local/eval-lucebox; O=$E/$TAG; mkdir -p "$O"
python3 -c "
import numpy as np
t = np.load('$E/heldout_8x1024.npy')
for i, s in enumerate(t): s.astype(np.int32).tofile('$O/seq%d.i32' % i)"
for i in 0 1 2 3 4 5 6 7; do
  env HIP_VISIBLE_DEVICES=0 STRATA_PF_NO_DUMMY=1 "$@" ./build-hip-fused/ds41_generate \
    -m "$MODEL" --ids-file "$O/seq$i.i32" -n 1 --temp 0 \
    --backend cuda --experts gpu --arena-gib 75 --ctx 4096 --kv-q8 --threads 12 --rank "$RANK" \
    --engram "$ENGRAM" --route-bias 0 --route-bias-file 0 --ppl --dump-logits "$O/seq$i.f32" \
    > "$O/seq$i.out" 2> "$O/seq$i.log"
  echo "seq $i: $(grep '^PPL' "$O/seq$i.out")"
done
python3 $E/score.py "$O" ${AGAINST:+--against "$AGAINST"}
# KEEP_LOGITS=1 leaves the 8 x 528 MB logits for a later --against comparison (two runs of one change)
[ "${KEEP_LOGITS:-0}" = 1 ] || rm -f "$O"/seq*.f32
