#!/usr/bin/env bash
# ab-rco-requant.sh - the GSQ-RCO pack on the chat configuration's flags, STRATA_DENSE_REQUANT=q6_K (the chat
# config) against the pack's own dense types.  On this pack the switch only touches attn_q_a and attn_kv (the 80 Q8_0
# matrices; the rest of the dense half is Q5_K already).  REPS (default 3) rounds, arms alternated:
#  1. decode: Promessi Sposi (300 tokens) and the 17K prompt (128 tokens, ctx 24576), forced on the first run's stream;
#  2. prompt from zero on the internal NVMe: 8K / 17K / 25.5K-token prompts, 8 tokens decoded, ctx 32768.
# The chat server must be stopped.  Results in local/sweep-rco-rq/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=${OUT:-local/sweep-rco-rq}
if ss -ltn 2>/dev/null | grep -q ':8150 ' || pgrep -f '[d]s41_generate.*--serve' >/dev/null; then
    echo "ab-rco-requant: the chat server is running: stop it first (./stop.sh)" >&2; exit 3
fi
export ARENA=80
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1 STRATA_SUM_ZEROCOPY=1"
echo "ABRQ_START $(date '+%F %T')" >> $O/summary.txt 2>/dev/null || { mkdir -p $O; echo "ABRQ_START $(date '+%F %T')" >> $O/summary.txt; }
for r in $(seq 1 ${REPS:-3}); do
    bash $S $O local/prompts/sposi.i32 300 q6k-r$r "$E STRATA_DENSE_REQUANT=q6_K" $R
    bash $S $O local/prompts/sposi.i32 300 own-r$r "$E" $R
    bash $S $O local/prompts/huge17k.i32 128 q6k-r$r "$E STRATA_DENSE_REQUANT=q6_K" $R --ctx 24576
    bash $S $O local/prompts/huge17k.i32 128 own-r$r "$E" $R --ctx 24576
done
for r in $(seq 1 ${REPS:-3}); do
    for p in prompt8k huge17k prompt25k; do
        NOFORCE=1 bash $S $O local/prompts/$p.i32 8 pq6k-r$r "$E STRATA_DENSE_REQUANT=q6_K" $R --ctx 32768
        NOFORCE=1 bash $S $O local/prompts/$p.i32 8 pown-r$r "$E" $R --ctx 32768
    done
done
echo "ABRQ_DONE $(date '+%F %T')" >> $O/summary.txt
