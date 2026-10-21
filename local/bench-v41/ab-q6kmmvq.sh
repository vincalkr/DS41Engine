#!/usr/bin/env bash
# ab-q6kmmvq.sh - GGML_CUDA_Q6K_MMVQ=1 (the dense half's decode Q6_K matvecs through q6k_mmvq_f,
# ggml-cuda/mmvq-q6k.cuh: no quantize_q8_1 launch, the f32 column used as it is) against the default, on the default
# two-card configuration, forced on the reference tokens in local/sweep-q6k (copied from local/sweep-prof,
# prof-decode.sh), so both variants decode the same text:
#   base   the default (ggml's quantize_q8_1 + mul_mat_vec_q)
#   q6k    GGML_CUDA_Q6K_MMVQ=1
# REPS (default "1 2 3 4 5") repetitions, the variants interleaved; results appended to $OUT/summary.txt (default
# local/sweep-q6k, tags q6k-*); `local/bench-v41/ab-summary.py q6k- local/sweep-q6k/summary.txt` prints the means.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=${OUT:-local/sweep-q6k}
export ARENA=75
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
VARIANTS=${VARIANTS:-"base q6k"}
env_of() {
    case $1 in
        base) echo "" ;;
        q6k) echo "GGML_CUDA_Q6K_MMVQ=1" ;;
        q6kprof) echo "GGML_CUDA_Q6K_MMVQ=1 STRATA_DEC_PROF=1" ;;
        baseprof) echo "STRATA_DEC_PROF=1" ;;
    esac
}
for r in ${REPS:-1 2 3 4 5}; do
    for p in ${PROMPTS:-huge17k sposi}; do
        if [ $p = huge17k ]; then N=128; X="--ctx 24576"; else N=300; X=""; fi
        for v in $VARIANTS; do
            bash $S $O local/prompts/$p.i32 $N q6k-$v-r$r "$E $(env_of $v)" $R $X
        done
    done
done
echo Q6KMMVQDONE >> $O/summary.txt
