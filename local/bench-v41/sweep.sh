#!/usr/bin/env bash
# local/bench-v41/sweep.sh OUTDIR PROMPT N TAG "ENV=.." [args...] - one greedy one-shot run of the engine (the ds4 1017 Q2 by default).
# If OUTDIR/ref-PROMPTNAME.ids exists the run is FORCED on it (STRATA_FORCE_IDS + STRATA_DEC_MARGIN, unless NOFORCE=1):
# same text, so the speed is comparable, and the "margin:" lines count where this variant's argmax differs.
# Env: Q2 (model, default the ds4 1017 Q2), RANK, ENGRAM, ARENA (GiB, 90), THREADS (12), PIN (taskset cpu list).
set -u
cd "$(dirname "$0")/../.."
OUT=$1 PROMPT=$2 N=$3 TAG=$4 ENVS=$5; shift 5
Q2="${Q2:-models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf}"
# the GSQ-RCO pack has no deepseek41.* keys of its own: the geometry overlay (tools/v41/ds41_pack_meta.py)
# travels with it.  A ds4 pack ignores a DS41_META that is set by mistake (the tensor names follow the pack).
export DS41_META="${DS41_META-local/rco-iq2xxs-meta/meta.gguf}"
RANK="${RANK:-local/rank-q2.bin}"
ENGRAM="${ENGRAM:-models/engram-1017/layers14:models/engram-1017/layers1}"
ARENA="${ARENA:-90}"
PY=/run/media/vincenzo/Dati/projects/other/Strata/.venv/bin/python
mkdir -p "$OUT" ~/.quetza-data/conductor
pn=$(basename "$PROMPT" .i32)
ref="$OUT/ref-$pn.ids"
force=""
[ -s "$ref" ] && [ -z "${NOFORCE:-}" ] && force="STRATA_FORCE_IDS=$ref STRATA_DEC_MARGIN=1"
o="$OUT/$pn-$TAG"
env $force $ENVS bash tools/ds4/memguard.sh 110 82 -- ${PIN:+taskset -c $PIN} ./build-hip-fused/ds41_generate -m "$Q2" \
  --ids-file "$PROMPT" -n "$N" --temp 0 --backend cuda --experts gpu --arena-gib "$ARENA" --ctx 8192 --kv-q8 \
  --threads "${THREADS:-12}" --rank "$RANK" --engram "$ENGRAM" --arena-skip-vram "$@" > "$o.out" 2>&1
grep -E '^[0-9]+$' "$o.out" > "$o.ids"
[ -s "$ref" ] || cp "$o.ids" "$ref"
"$PY" local/detok.py < "$o.ids" > "$o.txt" 2>/dev/null
nd=$(grep -c '^margin:' "$o.out")
printf '%-28s %-14s diff=%-4s %s\n' "$TAG" "$pn" "$nd" "$(grep '^decode:' "$o.out" | tail -1 | cut -c1-170)" | tee -a "$OUT/summary.txt"
