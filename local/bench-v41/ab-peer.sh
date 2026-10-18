#!/usr/bin/env bash
# A/B: the second card (peer miss tier) on/off, and STRATA_DENSE_ALIAS, on the current one-shot configuration.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-gsq
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0"
E="STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DEC_PROF=1"
P="STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16"
for r in 3 4; do for p in sposi code-en; do
 bash $S $O local/prompts/$p.i32 300 pr$r-peer "$E $P" $R --peer-device 1
 bash $S $O local/prompts/$p.i32 300 pr$r-nopeer "$E" $R
 bash $S $O local/prompts/$p.i32 300 pr$r-peer-alias "$E $P STRATA_DENSE_ALIAS=1" $R --peer-device 1
done; done
echo ABDONE >> $O/summary.txt
