#!/usr/bin/env bash
# ab-fileread.sh - the decode file tier: STRATA_FILE_THREADS (threads one layer's file reads are split over, default 8)
# and the opt-in speculative reads (STRATA_SPEC_TOP=4), on the default two-card configuration, forced on the
# reference tokens in local/sweep-prof (prof-decode.sh).  Results appended to local/sweep-prof/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-prof
export ARENA=75
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
for r in ${REPS:-1}; do
    for p in huge17k sposi; do
        if [ $p = huge17k ]; then N=128; X="--ctx 24576"; else N=300; X=""; fi
        bash $S $O local/prompts/$p.i32 $N fr-base-r$r "$E" $R $X
        bash $S $O local/prompts/$p.i32 $N fr-ft16-r$r "$E STRATA_FILE_THREADS=16" $R $X
        bash $S $O local/prompts/$p.i32 $N fr-ft32-r$r "$E STRATA_FILE_THREADS=32" $R $X
        bash $S $O local/prompts/$p.i32 $N fr-spec4-r$r "$E STRATA_SPEC_TOP=4" $R $X
        bash $S $O local/prompts/$p.i32 $N fr-ft16spec4-r$r "$E STRATA_FILE_THREADS=16 STRATA_SPEC_TOP=4" $R $X
    done
done
echo FILEREADDONE >> $O/summary.txt
