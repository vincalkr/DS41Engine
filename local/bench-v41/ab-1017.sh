#!/usr/bin/env bash
# ab-1017.sh - decode speed of the ds4 1017 Q2 (models/ds4-1017, antirez/deepseek-v4.1-flash-gguf) on the default
# two-card configuration's flags and env, the same prompts as the GSQ's 2026-10-20 runs (local/sweep-prof, hp-tuned:
# sposi 300 tokens, huge17k 128), REPS (default 5) runs each, unforced (another model: the GSQ's reference tokens do
# not apply; the first run of each prompt becomes this model's reference).  Results in local/sweep-1017/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-1017
export ARENA=75 Q2=models/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf RANK=${RANK:-local/rank-q2.bin}
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
TAG=${TAG:-q2}
for r in $(seq 1 ${REPS:-5}); do
    bash $S $O local/prompts/huge17k.i32 128 $TAG-r$r "$E" $R --ctx 24576
    bash $S $O local/prompts/sposi.i32 300 $TAG-r$r "$E" $R
done
echo AB1017DONE >> $O/summary.txt
