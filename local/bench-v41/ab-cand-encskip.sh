#!/usr/bin/env bash
# ab-cand-encskip.sh - two A/Bs on the default two-card flags, one-shot runs alternated, REPS (default 3) each.
#  1. the candidate fast-path (every block kept, below ~32K positions): STRATA_CAND_FAST=1 against the default graph,
#     decode on Promessi Sposi (300 tokens) and the 17K prompt (128 tokens);
#  2. enc-skip: prompt speed of 8K / 17K / 25.5K-token prompts, default against --no-enc-skip (8 tokens decoded).
# The first run of each prompt is the reference the later ones are forced on ("diff=" counts argmax changes).
# Results in local/sweep-cand/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-cand
# Recorded runs were on antirez/ds4's Q2 pack; since 2026-10-25 the default is the GSQ-RCO pack (see sweep.sh).
export ARENA=75 Q2=models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf RANK=${RANK:-local/rank-q2.bin}
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
for r in $(seq 1 ${REPS:-3}); do
    bash $S $O local/prompts/sposi.i32 300 cand-r$r "$E STRATA_CAND_FAST=1" $R
    bash $S $O local/prompts/sposi.i32 300 always-r$r "$E" $R
    bash $S $O local/prompts/huge17k.i32 128 cand-r$r "$E STRATA_CAND_FAST=1" $R --ctx 24576
    bash $S $O local/prompts/huge17k.i32 128 always-r$r "$E" $R --ctx 24576
done
for r in $(seq 1 ${REPS:-3}); do
    for p in prompt8k huge17k prompt25k; do
        bash $S $O local/prompts/$p.i32 8 skip-r$r "$E" $R --ctx 32768
        bash $S $O local/prompts/$p.i32 8 noskip-r$r "$E" $R --ctx 32768 --no-enc-skip
    done
done
echo ABCANDDONE >> $O/summary.txt
