#!/usr/bin/env bash
# ab-rco-levers.sh - an opt-in lever on the GSQ-RCO pack, each against the chat configuration's flags (two cards,
# arena 80 GiB, zero-copy sum, q6_K requant), arms alternated, REPS (default 3) rounds; Promessi Sposi (300 tokens)
# and the 17K prompt (128 tokens, ctx 24576), forced on local/sweep-rco-rq's reference streams:
#   rank  RANK=local/rank-rco-calib.bin   the pack's own byte-correct expert order instead of local/rank.bin
# (Two more arms were removed: a release of ggml's device pool after the prompt gave back 0.22 GiB, and
# STRATA_DOWN_AMD_IQ selected AMD layouts that this gfx1100 build does not compile, so it ran the default kernels.)
# The chat server must be stopped.  Results in local/sweep-rco-lev/summary.txt.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=${OUT:-local/sweep-rco-lev}
if ss -ltn 2>/dev/null | grep -q ':8150 ' || pgrep -f '[d]s41_generate.*--serve' >/dev/null; then
    echo "ab-rco-levers: the chat server is running: stop it first (./stop.sh)" >&2; exit 3
fi
mkdir -p $O; cp -n local/sweep-rco-rq/ref-sposi.ids local/sweep-rco-rq/ref-huge17k.ids $O/
export ARENA=80
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1 STRATA_SUM_ZEROCOPY=1"
run() {   # arm extra-env rank
    RANK=$3 bash $S $O local/prompts/huge17k.i32 128 $1-r$r "$E $2" $R --ctx 24576
    RANK=$3 bash $S $O local/prompts/sposi.i32 300 $1-r$r "$E $2" $R
}
for r in $(seq 1 ${REPS:-3}); do
    run base "" local/rank.bin
    run rank "" local/rank-rco-calib.bin
done
echo ABLEVDONE >> $O/summary.txt
