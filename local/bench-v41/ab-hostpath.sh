#!/usr/bin/env bash
# ab-hostpath.sh - the host side of a decode layer, on the default two-card configuration, forced on the reference
# tokens in local/sweep-prof (prof-decode.sh):
#   zc        STRATA_SUM_ZEROCOPY=1: the sum reads the CPU's and the peer's rows from pinned host memory (no per-row
#             copies, no weights upload)
#   fpoolN    STRATA_FILE_POOL=1 STRATA_FILE_THREADS=N: decode file reads on persistent reader threads, N parts
#   spinX     STRATA_POOL_SPIN_US=X: the CPU expert pool spins X us after a batch before sleeping (default 20 ms)
#   specpoolT STRATA_SPEC_TOP=T STRATA_SPEC_ADAPT=1 + fpool16: speculative file reads two layers ahead under --arena-adapt
# REPS (default 1) repetitions; results appended to local/sweep-prof/summary.txt (tags hp-*).
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-prof
export ARENA=75
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
VARIANTS=${VARIANTS:-"base zc fpool8 fpool16 fpool32 spin0 spin300"}
env_of() {
    case $1 in
        base | tuned | gpuhigh) echo "" ;;   # tuned: the default with local/power-tune.sh on; gpuhigh: + local/gpu-perf.sh high
        zc) echo "STRATA_SUM_ZEROCOPY=1" ;;
        fpool*) echo "STRATA_FILE_POOL=1 STRATA_FILE_THREADS=${1#fpool}" ;;
        spin*) echo "STRATA_POOL_SPIN_US=${1#spin}" ;;
        specpool*) echo "STRATA_SPEC_TOP=${1#specpool} STRATA_SPEC_ADAPT=1 STRATA_FILE_POOL=1 STRATA_FILE_THREADS=16" ;;
        all) echo "STRATA_SUM_ZEROCOPY=1 STRATA_FILE_POOL=1 STRATA_FILE_THREADS=${ALL_FT:-16}" ;;
    esac
}
for r in ${REPS:-1}; do
    for p in ${PROMPTS:-huge17k sposi}; do
        if [ $p = huge17k ]; then N=128; X="--ctx 24576"; else N=300; X=""; fi
        for v in $VARIANTS; do
            bash $S $O local/prompts/$p.i32 $N hp-$v-r$r "$E $(env_of $v)" $R $X
        done
    done
done
echo HOSTPATHDONE >> $O/summary.txt
