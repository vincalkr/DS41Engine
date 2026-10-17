#!/bin/sh
# local/test-q2-rank.sh - the Q2 "correct AND fast" acceptance test.
#
#   local/test-q2-rank.sh [rank.bin]      (default: local/rank-q2.bin)
#
# Runs the nine-circles prompt on the ds4 Q2 with EXACT routing (the only setting that gives the right answer) and a
# given rank, stops the chat server first because the run needs the arena, prints the decode speed, the answer, and
# which of the nine circles were named. The criterion is: all nine named AND >= 2-3 tok/s.
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf"
RANK="${1:-local/rank-q2.bin}"
PROMPT="local/prompts/circles.i32"
PY="/run/media/vincenzo/Dati/projects/other/Strata/.venv/bin/python"

[ -f "$MODEL" ]  || { echo "missing model: $MODEL" >&2; exit 1; }
[ -f "$RANK" ]   || { echo "missing rank: $RANK (build it with tools/v41/engine/v41_rank.py)" >&2; exit 1; }
[ -f "$PROMPT" ] || { echo "missing prompt: $PROMPT" >&2; exit 1; }

mkdir -p "$HOME/.quetza-data/conductor" 2>/dev/null || true
fuser -k 8150/tcp 2>/dev/null || true
sleep 3
pkill -x ds41_generate 2>/dev/null || true
sleep 3

OUT=/tmp/q2-rank.out
bash tools/ds4/memguard.sh 80 82 -- ./build-hip-fused/ds41_generate \
  -m "$MODEL" --ids-file "$PROMPT" -n 200 \
  --backend cuda --experts gpu --arena-gib 75 --ctx 262144 --kv-q8 --temp 0 --threads 24 \
  --rank "$RANK" --engram models/engram/layers14:models/engram/layers1 \
  --route-bias 0 --route-bias-file 0 > "$OUT" 2>&1
rc=$?
echo "run exit: $rc"
grep -E "^decode:" "$OUT" | tail -1
grep -iE "error|fail|cannot|missing" "$OUT" | head -3

ids=$(grep -E '^[0-9]+$' "$OUT")
if [ -n "$ids" ]; then
  printf '%s\n' "$ids" | "$PY" local/detok.py > /tmp/q2-rank.txt 2>/dev/null
  echo "--- answer ---"
  cat /tmp/q2-rank.txt
  echo "--- circles named ---"
  grep -ioE "limbo|lussur|golosi|gola|avari|avid|iracond|ira|eresiarc|eresia|eret|violent|violenza|fraud|frode|tradit" /tmp/q2-rank.txt | tr 'A-Z' 'a-z' | sort -u
fi
