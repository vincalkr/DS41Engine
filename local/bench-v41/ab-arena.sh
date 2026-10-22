#!/usr/bin/env bash
# ab-arena.sh - decode speed against the RAM arena size (--arena-gib) on the default two-card flags: ARENAS (default
# "70 75 80 85") in a rotating order, REPS (default 2) rounds; Promessi Sposi (300 tokens) and the 17K prompt (128
# tokens) per size.  The RAM outside the arena is the page cache the Engram tables and the file tier read through, so a
# bigger arena is not simply faster (measured on one card, 2026-10-15: 75 GiB best).  Results in
# local/sweep-arena/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-arena
export NOFORCE=1
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
set -- ${ARENAS:-70 75 80 85}
for r in $(seq 1 ${REPS:-2}); do
    for a in "$@"; do
        export ARENA=$a
        bash $S $O local/prompts/sposi.i32 300 a$a-r$r "$E" $R
        bash $S $O local/prompts/huge17k.i32 128 a$a-r$r "$E" $R --ctx 24576
    done
    set -- "${@:2}" "$1"   # rotate the order each round
done
echo ABARENADONE >> $O/summary.txt
