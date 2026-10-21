#!/usr/bin/env bash
# ab-peerwait.sh - how long card 0 waits for card 1 at each layer's hand-off (STRATA_PEER_PROF=1), next to the same
# run without the profiler and without the second card, on short prose (sposi, 300 tokens) and at 17K context
# (huge17k, 128 tokens).  One-shot runs through sweep.sh, forced on the first run's tokens; results in
# local/sweep-peerwait/summary.txt and the "peer prof:" lines of the .out files.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-peerwait
export ARENA=75
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5"
P="HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16"
for r in 1 2; do
    bash $S $O local/prompts/sposi.i32 300 r$r-1gpu "$E HIP_VISIBLE_DEVICES=0" $R
    bash $S $O local/prompts/sposi.i32 300 r$r-2gpu "$E $P" $R --peer-device 1
    bash $S $O local/prompts/sposi.i32 300 r$r-2gpu-prof "$E $P STRATA_PEER_PROF=1" $R --peer-device 1
done
for r in 1 2; do
    bash $S $O local/prompts/huge17k.i32 128 r$r-1gpu "$E HIP_VISIBLE_DEVICES=0" $R --ctx 24576
    bash $S $O local/prompts/huge17k.i32 128 r$r-2gpu "$E $P" $R --peer-device 1 --ctx 24576
    bash $S $O local/prompts/huge17k.i32 128 r$r-2gpu-prof "$E $P STRATA_PEER_PROF=1" $R --peer-device 1 --ctx 24576
done
echo PEERWAITDONE >> $O/summary.txt
