#!/usr/bin/env bash
# fetch-1017-progress.sh [SECONDS] - overall progress of local/fetch-1017.sh: bytes on disk of the 1017 quant and
# Engram against the 351 GB total, the rate over the last SECONDS (default 10) and the time left at that rate.
# A part already appended to the joined .gguf counts once (the joined file holds it).
cd "$(dirname "$0")/.." || exit 1
TOTAL=$((40000000000 * 3 + 28090175488 + 101535150936 + 101537926640))
have() { du -s --block-size=1 models/V41-1017-2bit models/engram-1017 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
a=$(have); sleep "${1:-10}"; b=$(have)
export LC_ALL=C
awk -v a="$a" -v b="$b" -v t="$TOTAL" -v s="${1:-10}" -v now="$(date +%H:%M:%S)" 'BEGIN {
    r = (b - a) / s; e = r > 0 ? (t - b) / r : 0
    printf "%s  %.1f / %.1f GB (%.1f%%)  %.0f MB/s  about %d h %02d min left\n", now, b / 1e9, t / 1e9, 100 * b / t, r / 1e6, int(e / 3600), int((e % 3600) / 60)
}'
grep -E "DL:" local/fetch-1017.log | tail -1
grep -E "^20.*(get|have|check|appended|ERROR|done)" local/fetch-1017.log | tail -1
