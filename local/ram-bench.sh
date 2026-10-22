#!/usr/bin/env bash
# ram-bench.sh [TAG [--decode]] - RAM read bandwidth, and optionally the engine's decode, before and after a BIOS
# change (FCLK, memory speed or timings).  TAG names the setting (fclk2000, fclk2100, ...).
#
#   local/ram-bench.sh                      no arguments: the results so far, one row per TAG
#   local/ram-bench.sh fclk2000             streaming read of an 8 GiB buffer, 6 / 12 / 24 threads, 4 KiB pages and
#                                           huge pages, best of 5 each (local/bench-v41/membw.c; ~1 minute)
#   local/ram-bench.sh fclk2000 --decode    + two one-shot decodes on the default two-card flags: Promessi Sposi
#                                           (300 tokens) and the 17K prompt (128 tokens), REPS (default 3) each
#                                           (~10 minutes; the server must be stopped)
#
# Results are appended to local/ram-bench/summary.txt, one line per measure, so runs with different TAGs line up.
# Decode depends on the RAM through the experts the CPU computes (12 threads, ~47-50 GB/s of 52-54 at FCLK 2000,
# 2026-10-22); the 17K prompt reads more of them than the short one.  Turn power-tune on first (sudo local/power-tune.sh
# on, or run it after run.sh installed it) so both sides of a comparison have it.
set -u
cd "$(dirname "$0")/.."
O=local/ram-bench
if [ $# -eq 0 ]; then
    [ -s "$O/summary.txt" ] || { echo "no results yet: local/ram-bench.sh TAG [--decode]"; exit 0; }
    # per TAG: the best bandwidth at 12 threads (4 KiB pages / THP), the mean decode of each prompt (and its runs)
    awk '
        $1 !~ /^[0-9][0-9][0-9][0-9]-/ { next }   # only the lines this script writes (they start with the date)
        / threads: / { tag = $2; split($0, a, "pages="); pg = substr(a[2], 1, index(a[2], " ") - 1)
                       match($0, /([0-9]+) threads: ([0-9.]+) GB/, m)
                       if (m[1] == 12) { k = tag SUBSEP pg; if (!(k in bw) || m[2] > bw[k]) bw[k] = m[2] }
                       seen[tag] = 1; if (!(tag in order)) order[tag] = ++n; next }
        /decode:/    { tag = $2; p = $3; match($0, /= ([0-9.]+) tok\/s/, m)
                       s[tag SUBSEP p] += m[1]; c[tag SUBSEP p]++; seen[tag] = 1; if (!(tag in order)) order[tag] = ++n }
        END {
            printf "%-16s %14s %14s %22s %22s\n", "TAG", "GB/s 12t 4K", "GB/s 12t THP", "decode sposi tok/s", "decode 17K tok/s"
            for (i = 1; i <= n; i++) for (t in order) if (order[t] == i) {
                b4 = (t SUBSEP "4K") in bw ? bw[t SUBSEP "4K"] : "-"; bt = (t SUBSEP "THP") in bw ? bw[t SUBSEP "THP"] : "-"
                ds = c[t SUBSEP "sposi"] ? sprintf("%.2f (%d runs)", s[t SUBSEP "sposi"] / c[t SUBSEP "sposi"], c[t SUBSEP "sposi"]) : "-"
                dh = c[t SUBSEP "huge17k"] ? sprintf("%.2f (%d runs)", s[t SUBSEP "huge17k"] / c[t SUBSEP "huge17k"], c[t SUBSEP "huge17k"]) : "-"
                printf "%-16s %14s %14s %22s %22s\n", t, b4, bt, ds, dh
            }
        }' "$O/summary.txt"
    echo "(every line: $O/summary.txt)"
    exit 0
fi
TAG=$1
DECODE=0; [ "${2:-}" = --decode ] && DECODE=1
mkdir -p "$O"
BIN=build-hip-fused/membw
if [ ! -x "$BIN" ] || [ local/bench-v41/membw.c -nt "$BIN" ]; then
    gcc -O3 -march=native -fopenmp -o "$BIN" local/bench-v41/membw.c || { echo "ram-bench: gcc failed" >&2; exit 1; }
fi
stamp=$(date +%F_%H:%M)
echo "=== $TAG: RAM read bandwidth"
for huge in 0 1; do
    "$BIN" 8 "$huge" | tee "$O/membw-$TAG-$huge.txt"
    grep -E "threads" "$O/membw-$TAG-$huge.txt" | sed "s/^/$stamp $TAG pages=$([ $huge = 1 ] && echo THP || echo 4K) /" >> "$O/summary.txt"
done
[ "$DECODE" = 1 ] || exit 0
if pgrep -f "[s]erve_ds41.py" >/dev/null || pgrep -x ds41_generate >/dev/null; then
    echo "ram-bench: the server or an engine is running: stop it for --decode" >&2; exit 1
fi
echo "=== $TAG: decode (the default two-card flags)"
export ARENA=75 NOFORCE=1   # 75 GiB, as the fclk-current baseline was measured (the chat config has 80)
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
for r in $(seq 1 "${REPS:-3}"); do
    for p in sposi huge17k; do
        n=300; c=""; [ $p = huge17k ] && { n=128; c="--ctx 24576"; }
        bash local/bench-v41/sweep.sh "$O/runs" local/prompts/$p.i32 $n "$TAG-r$r" "$E" $R $c > /dev/null
        line=$(grep -h '^decode:' "$O/runs/$p-$TAG-r$r.out" | sed -E 's/; spec.*//')
        echo "$p r$r: $line"
        echo "$stamp $TAG $p r$r $line" >> "$O/summary.txt"
    done
done
