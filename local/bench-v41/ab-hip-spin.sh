#!/usr/bin/env bash
# ab-hip-spin.sh - STRATA_HIP_SPIN=1 (host synchronizations spin instead of yielding) against the chat configuration's
# flags (two cards, arena 80 GiB, zero-copy sum, q6_K requant, elastic columns), arms alternated, REPS (default 3)
# rounds; Promessi Sposi (300 tokens) and the 17K prompt (128 tokens, ctx 24576), forced on local/sweep-rco-rq's
# reference streams.  The chat server must be stopped.  Results in local/sweep-hip-spin/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=${OUT:-local/sweep-hip-spin}
if ss -ltn 2>/dev/null | grep -q ':8150 ' || pgrep -f '[d]s41_generate.*--serve' >/dev/null; then
    echo "ab-hip-spin: the chat server is running: stop it first (./stop.sh)" >&2; exit 3
fi
mkdir -p $O; cp -n local/sweep-rco-rq/ref-sposi.ids local/sweep-rco-rq/ref-huge17k.ids $O/
export ARENA=80 RANK=local/rank.bin
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1 STRATA_SUM_ZEROCOPY=1 STRATA_ELASTIC_COLS=1"
run() {   # arm extra-env
    bash $S $O local/prompts/huge17k.i32 128 $1-r$r "$E $2" $R --ctx 24576
    bash $S $O local/prompts/sposi.i32 300 $1-r$r "$E $2" $R
}
for r in $(seq 1 ${REPS:-3}); do
    run base ""
    run spin "STRATA_HIP_SPIN=1"
done
echo ABSPINDONE >> $O/summary.txt
