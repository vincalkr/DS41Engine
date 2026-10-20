#!/usr/bin/env bash
# ab-rank1017.sh - the ds4 1017 Q2 with local/rank-q2.bin (the September Q2's corpus rank) against local/rank-1017.bin
# (rebuild-rank.sh: that corpus plus the 1017's own routing from the chat, local/routes-1017), interleaved, REPS runs
# each (default 3), the ab-1017.sh prompts and flags.  Results in local/sweep-1017/summary.txt (tags rq2-*, r1017-*).
cd "$(dirname "$0")/../.."
for r in $(seq 1 ${REPS:-3}); do
    REPS=1 TAG=rq2-$r RANK=local/rank-q2.bin bash local/bench-v41/ab-1017.sh
    REPS=1 TAG=r1017-$r RANK=local/rank-1017.bin bash local/bench-v41/ab-1017.sh
done
