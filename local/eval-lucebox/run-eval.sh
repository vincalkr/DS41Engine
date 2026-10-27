#!/usr/bin/env bash
# run-eval.sh TAG [ENV=VAL ...] - Lucebox's 1017 held-out set (8 x 1024 tokens, eval/heldout_8x1024.npy) through
# this engine: one run per sequence with --ppl --dump-logits, then score.py prints the ppl and the run's own top-1 (and
# KL / top-1 agreement against the FP8 1017 teacher's top-256 log-probs when teacher_8x1024.npz is present).  Exact
# routing; the extra ENV=VAL pairs go to the engine (e.g. STRATA_DENSE_REQUANT=q6_K).  The chat server must be
# stopped.  MODEL, RANK and ENGRAM pick the files (defaults: the GSQ-RCO pack, the rank and the FP8 Engram rows of
# local/ds41-local.json; ENGRAM=models/rco-iq2xxs/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf for the pack's own table).
# KEEP_LOGITS=1 keeps the logits and AGAINST=<run dir> scores this run against another run's logits (see score.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; shift
MODEL=${MODEL:-models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf}
RANK=${RANK:-$(python3 -c 'import json; a = json.load(open("local/ds41-local.json"))["args"]; print(a[a.index("--rank") + 1])')}
export DS41_META=${DS41_META-local/rco-iq2xxs-meta/meta.gguf}   # the pack's geometry overlay
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
