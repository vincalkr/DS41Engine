#!/usr/bin/env bash
# prof-decode.sh - where a decode token goes on the default two-card configuration, and four ROCm runtime knobs.
#  1. STRATA_DEC_PROF=1 stage split (begin / predict / attn_router / tier / finish) on sposi and at 17K context;
#  2. a rocprofv3 kernel + copy trace of 16 and 80 decode tokens (the difference is 64 tokens of decode);
#  3. HIP_FORCE_DEV_KERNARG=1, HSA_ENABLE_INTERRUPT=0, HSA_ENABLE_SDMA=0 and all three, 2 runs each, on sposi.
# Results: local/sweep-prof/summary.txt, the .out files, and local/sweep-prof/trace-{16,80}/.
cd "$(dirname "$0")/../.."
S=local/bench-v41/sweep.sh; O=local/sweep-prof
export ARENA=75
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
mkdir -p $O
# the reference tokens (first run of each prompt) come from a plain run
bash $S $O local/prompts/sposi.i32 300 base-r1 "$E" $R
bash $S $O local/prompts/sposi.i32 300 decprof "$E STRATA_DEC_PROF=1" $R
bash $S $O local/prompts/huge17k.i32 128 base-r1 "$E" $R --ctx 24576
bash $S $O local/prompts/huge17k.i32 128 decprof "$E STRATA_DEC_PROF=1" $R --ctx 24576
for n in 16 80; do
    rm -rf $O/trace-$n
    env $E STRATA_FORCE_IDS=$O/ref-sposi.ids bash tools/ds4/memguard.sh 110 82 -- \
        rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d $O/trace-$n -o run -- \
        ./build-hip-fused/ds41_generate -m models/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --ids-file local/prompts/sposi.i32 -n $n --temp 0 \
        --backend cuda --experts gpu --arena-gib 75 --ctx 8192 --kv-q8 --threads 12 --rank local/rank-q2.bin \
        --engram models/engram-1017/layers14:models/engram-1017/layers1 --arena-skip-vram $R > $O/trace-$n.out 2>&1
done
for r in 2 3; do
    bash $S $O local/prompts/sposi.i32 300 base-r$r "$E" $R
    bash $S $O local/prompts/sposi.i32 300 kernarg-r$r "$E HIP_FORCE_DEV_KERNARG=1" $R
    bash $S $O local/prompts/sposi.i32 300 poll-r$r "$E HSA_ENABLE_INTERRUPT=0" $R
    bash $S $O local/prompts/sposi.i32 300 blit-r$r "$E HSA_ENABLE_SDMA=0" $R
    bash $S $O local/prompts/sposi.i32 300 all3-r$r "$E HIP_FORCE_DEV_KERNARG=1 HSA_ENABLE_INTERRUPT=0 HSA_ENABLE_SDMA=0" $R
done
echo PROFDONE >> $O/summary.txt
