#!/usr/bin/env bash
# A/B: STRATA_DENSE_REQUANT=q6_K (the dense half re-quantized at load, upstream Xolotl's served default) on the current one-shot configuration.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-gsq
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DEC_PROF=1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16"
for r in 1 2; do
 bash $S $O local/prompts/sposi.i32 300 cq$r-q8 "STRATA_DEC_PROF=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1" --route-bias 20 --route-bias-file 20 --pf-b 0 --vram-margin 0.5
 bash $S $O local/prompts/sposi.i32 300 cq$r-q6k "STRATA_DEC_PROF=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K" --route-bias 20 --route-bias-file 20 --pf-b 0 --vram-margin 0.5
 for p in sposi code-en; do
  bash $S $O local/prompts/$p.i32 300 rq$r-q8 "$E" $R
  bash $S $O local/prompts/$p.i32 300 rq$r-q6k "$E STRATA_DENSE_REQUANT=q6_K" $R
 done
done
for p in divina-breve circles; do NOFORCE=1 bash $S $O local/prompts/$p.i32 300 free-q6k "$E STRATA_DENSE_REQUANT=q6_K" $R; done
echo Q6DONE >> $O/summary.txt
