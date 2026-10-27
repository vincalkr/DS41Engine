#!/usr/bin/env bash
# ab-rco-pcie.sh - the share of the CPU tier's misses computed on card 0 instead (--pcie F: the expert's blob is copied
# from the RAM arena over PCIe x16 and run on the card while the CPU computes the rest), against the configuration's
# --pcie 0.  The chat configuration's flags, one-shot runs with the server's elastic state columns, arms alternated,
# REPS (default 2) rounds; the 17K prompt (128 tokens) and Promessi Sposi (300 tokens), forced on local/sweep-rco-rq's
# reference streams.  PCIES (default "0 0.15 0.3").  The chat server must be stopped.  Results in local/sweep-rco-pcie.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=${OUT:-local/sweep-rco-pcie}
if ss -ltn 2>/dev/null | grep -q ':8150 ' || pgrep -f '[d]s41_generate.*--serve' >/dev/null; then
    echo "ab-rco-pcie: the chat server is running: stop it first (./stop.sh)" >&2; exit 3
fi
mkdir -p $O; cp -n local/sweep-rco-rq/ref-sposi.ids local/sweep-rco-rq/ref-huge17k.ids $O/
export ARENA=80
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1 STRATA_SUM_ZEROCOPY=1 STRATA_ELASTIC_COLS=1"
for r in $(seq 1 ${REPS:-2}); do
    for p in ${PCIES:-0 0.15 0.3}; do
        R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie $p --peer-device 1"
        bash $S $O local/prompts/huge17k.i32 128 pcie$p-r$r "$E" $R --ctx 24576 --prompt-batch 8192
        bash $S $O local/prompts/sposi.i32 300 pcie$p-r$r "$E" $R
    done
done
echo ABPCIEDONE >> $O/summary.txt
