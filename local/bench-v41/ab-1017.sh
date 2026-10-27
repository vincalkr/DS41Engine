#!/usr/bin/env bash
# ab-1017.sh - one-shot decode speed of the default model and two-card configuration's flags and env: PROMPTS (default
# Promessi Sposi, 300 tokens, and the 17K-token prompt, 128 tokens), REPS (default 5) runs each; the first run of each prompt
# becomes the reference the later ones are forced on.  TAG and RANK select the variant (update-rank.sh uses them).
# Results in local/sweep-1017/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-1017
# The numbers this harness recorded (local/sweep-1017) were measured on antirez/ds4's Q2 pack; since
# 2026-10-25 its default is the GSQ-RCO pack this box runs, whose geometry sweep.sh reads from the overlay.
export ARENA=80 Q2=models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf RANK=${RANK:-local/rank-q2.bin}
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1 STRATA_SUM_ZEROCOPY=1"
TAG=${TAG:-q2}
# PROMPTS: name:tokens:ctx of local/prompts/<name>.i32 (ctx empty = sweep.sh's 8192); update-rank.sh passes the
# software-development pair (dev-code, dev-docs)
PROMPTS=${PROMPTS:-huge17k:128:24576 sposi:300:}
for r in $(seq 1 ${REPS:-5}); do
    for pr in $PROMPTS; do
        IFS=: read -r pn pt pc <<< "$pr"
        bash $S $O local/prompts/$pn.i32 $pt $TAG-r$r "$E" $R ${pc:+--ctx $pc}
    done
done
echo AB1017DONE >> $O/summary.txt
