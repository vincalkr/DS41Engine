#!/usr/bin/env bash
# ab-peerdisjoint.sh - card 1's store disjoint from the RAM arena (STRATA_PEER_DISJOINT=1: the ranked experts the arena
# does not hold, read from the model file), the arena evicting peer-held experts first (STRATA_ARENA_EVICT_PEER=1),
# and both, next to the 2-card run of ab-peerwait.sh, alternated, on short prose (sposi, 300 tokens) and at 17K
# context (huge17k, 128 tokens).  Forced on ab-peerwait.sh's reference tokens (the 1-card run's, copied into the
# output folder); results in local/sweep-peerdisjoint/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-peerdisjoint
export ARENA=75
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5"
P="HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16"
[ -s $O/ref-sposi.ids ] && [ -s $O/ref-huge17k.ids ] || cp local/sweep-peerwait/ref-*.ids $O/
for r in 1 2; do
    bash $S $O local/prompts/sposi.i32 300 r$r-2gpu "$E $P" $R --peer-device 1
    bash $S $O local/prompts/sposi.i32 300 r$r-2gpu-disj "$E $P STRATA_PEER_DISJOINT=1" $R --peer-device 1
    bash $S $O local/prompts/sposi.i32 300 r$r-2gpu-evict "$E $P STRATA_ARENA_EVICT_PEER=1" $R --peer-device 1
    bash $S $O local/prompts/sposi.i32 300 r$r-2gpu-disj-evict "$E $P STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1" $R --peer-device 1
done
for r in 1 2; do
    bash $S $O local/prompts/huge17k.i32 128 r$r-2gpu "$E $P" $R --peer-device 1 --ctx 24576
    bash $S $O local/prompts/huge17k.i32 128 r$r-2gpu-disj "$E $P STRATA_PEER_DISJOINT=1" $R --peer-device 1 --ctx 24576
    bash $S $O local/prompts/huge17k.i32 128 r$r-2gpu-evict "$E $P STRATA_ARENA_EVICT_PEER=1" $R --peer-device 1 --ctx 24576
    bash $S $O local/prompts/huge17k.i32 128 r$r-2gpu-disj-evict "$E $P STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1" $R --peer-device 1 --ctx 24576
done
echo PEERDISJOINTDONE >> $O/summary.txt
