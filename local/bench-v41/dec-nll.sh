#!/usr/bin/env bash
# dec-nll.sh TAG "ENV=VAL ..." [SEQS] - the decode path's perplexity (STRATA_DEC_NLL) of Lucebox's held-out sequences
# (local/eval-lucebox/heldout_8x1024.npy, the model's tokenizer): each sequence's first 32 tokens are the prompt and
# the other 992 are forced (STRATA_FORCE_IDS), so every scored token goes through the decode kernels (--ppl scores
# the prompt pass, which does not). The default two-card configuration; SEQS default "0 1 2 3". One line per sequence
# in local/sweep-q6k/nll/nll.txt; the runs' logs next to it.
set -u
cd "$(dirname "$0")/../.."
TAG=$1; X=$2; SEQS=${3:-"0 1 2 3"}
O=local/sweep-q6k/nll; mkdir -p "$O"
python3 -c "
import numpy as np
t = np.load('local/eval-lucebox/heldout_8x1024.npy')
for i, s in enumerate(t):
    s[:32].astype(np.int32).tofile('$O/seq%d.i32' % i)
    open('$O/ref-seq%d.ids' % i, 'w').write(''.join('%d\n' % v for v in s[32:]))"
export ARENA=75
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1"
for i in $SEQS; do
    bash local/bench-v41/sweep.sh "$O" "$O/seq$i.i32" 992 "$TAG" "$E STRATA_DEC_NLL=1 $X" $R
    echo "$TAG seq$i $(grep '^decode nll:' "$O/seq$i-$TAG.out") diff=$(grep -c '^margin:' "$O/seq$i-$TAG.out")" >> "$O/nll.txt"
done
