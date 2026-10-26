#!/usr/bin/env bash
# ab-rco-iq2xxs.sh - A/B decode speed of the model this box runs today against the published GSQ-RCO IQ2_XXS pack,
# on the same prompts and flags as ab-1017.sh (Promessi Sposi 300 tokens, the 17K prompt 128 tokens with ctx 24576).
#
# Run it ONLY while the chat server is closed: this script never stops or starts that server itself, and refuses to
# start if :8150 still answers or a `ds41_generate --serve` process exists. Close the chat yourself first:
#
#   ./stop.sh                 # or Ctrl+C in the chat terminal
#   REPS=3 bash local/bench-v41/ab-rco-iq2xxs.sh
#   ./run.sh                  # afterwards, back on 127.0.0.1:8150
#
# Modes: q2 = the antirez Q2 pack (removed from this box on 2026-10-25; local/fetch-model.sh brings it
# back), rco = the downloaded pack shard one (the engine
# resolves the other three shards beside it, plus the metadata overlay from tools/v41/ds41_pack_meta.py),
# rcogen = rco with STRATA_NO_IQ512=1, i.e. ggml-cpu's fallback for every IQ1_M row instead of the AVX-512 kernel -
# the within-pack measure of what the new kernel buys.  The antirez Q2 pack has no IQ1_M expert (gate/up IQ2_XXS,
# down Q2_K), so that measure only exists on the RCO pack.
# Env: MODELS ("rco rcogen"), REPS (2), ARENA (80 GiB), THREADS (12), OUT (local/sweep-rco2). Results there; each run keeps
# its .out/.ids/.txt like sweep.sh does, and ab-summary.txt holds the means per mode and prompt at the end.
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=${OUT:-local/sweep-rco2}
S=local/bench-v41/sweep.sh
Q2M=${Q2MODEL:-models/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf}
RCOM=${RCOMODEL:-models/rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf}
REPS=${REPS:-2}
RCOMETA=${RCOMETA:-$PWD/local/rco-iq2xxs-meta/meta.gguf}

export ARENA=${ARENA:-80} RANK=${RANK:-local/rank-q2.bin} \
       ENGRAM=${ENGRAM:-models/engram-1017/layers14:models/engram-1017/layers1} THREADS=${THREADS:-12}
E="STRATA_NO_LARGEPAGES=1 STRATA_PF_NO_DUMMY=1 STRATA_VRAM_BG=1 STRATA_DENSE_NOCONT=1 STRATA_DENSE_ALIAS=1 STRATA_DENSE_REQUANT=q6_K STRATA_PRESTAGE_GIB=1.5 HIP_VISIBLE_DEVICES=0,1 STRATA_PEER_GIB=22 STRATA_PEER_ADAPT=1 STRATA_PEER_ADAPT_INFLIGHT=8 STRATA_PEER_ADAPT_WIN=16 STRATA_PEER_DISJOINT=1 STRATA_ARENA_EVICT_PEER=1 STRATA_SUM_ZEROCOPY=1"
R="--route-bias 0 --route-bias-file 0 --arena-adapt --vram-margin 0.5 --pf-b 0 --pcie 0 --peer-device 1"

if ss -ltn 2>/dev/null | grep -q ':8150 '; then
    echo "ab-rco-iq2xxs: the chat server still listens on :8150 - close it yourself (./stop.sh) and rerun." >&2
    exit 3
fi
if pgrep -f '[d]s41_generate.*--serve' >/dev/null; then
    echo "ab-rco-iq2xxs: a ds41_generate --serve process is alive - close the chat yourself first." >&2
    exit 3
fi
# only the arms actually asked for need their files: the antirez Q2 is off this box, so MODELS="rco rcogen" must run
# without it (the q2 arm fails its check only when it is in MODELS)
MODES=${MODELS:-rco rcogen}
for m in $MODES; do
    case $m in
        q2)          f=$Q2M;      ov="" ;;
        rco|rcogen)  f=$RCOM;     ov=$RCOMETA ;;
        *) echo "ab-rco-iq2xxs: unknown mode '$m' (q2, rco, rcogen)" >&2; exit 5 ;;
    esac
    [ -r "$f" ] || { echo "ab-rco-iq2xxs: mode $m: missing model $f" >&2; exit 4; }
    [ -z "$ov" ] && continue
    [ -r "$ov" ] || { echo "ab-rco-iq2xxs: mode $m: missing metadata overlay $ov (tools/v41/ds41_pack_meta.py)" >&2; exit 4; }
done

mkdir -p "$OUT"
echo "AB_RCO_START $(date '+%F %T') reps=$REPS models=${MODELS:-rco rcogen}" >> "$OUT/summary.txt"

for mode in $MODES; do
    case $mode in
        q2)     Q2="$Q2M"; EXTRA="DS41_META=" ;;
        rco)    Q2="$RCOM"; EXTRA="DS41_META=$RCOMETA" ;;
        rcogen) Q2="$RCOM"; EXTRA="STRATA_NO_IQ512=1 DS41_META=$RCOMETA" ;;
        *)      echo "ab-rco-iq2xxs: unknown mode '$mode' (q2, rco, rcogen)" >&2; exit 5 ;;
    esac
    export Q2
    for r in $(seq 1 "$REPS"); do
        bash "$S" "$OUT" local/prompts/huge17k.i32 128 "$mode-r$r" "$E $EXTRA" $R --ctx 24576
        bash "$S" "$OUT" local/prompts/sposi.i32 300 "$mode-r$r" "$E $EXTRA" $R
    done
done

# Means per mode and prompt from the decode lines of every run's .out.
{
for mode in $MODES; do
    for pn in huge17k sposi; do
        files=$(ls -1 "$OUT/$pn-$mode-r"*.out 2>/dev/null || true)
        [ -z "$files" ] && continue
        echo "== $mode $pn"
        printf '%s\n' "$files" | while read -r f; do grep '^decode:' "$f" | tail -1; done \
            | sed -E 's/.* = ([0-9.]+) tok\/s.*/\1/' \
            | awk '{v[NR]=$1;s+=$1} END{if(NR){mn=v[1];mx=v[1];for(i=2;i<=NR;i++){if(v[i]<mn)mn=v[i];if(v[i]>mx)mx=v[i]} printf "   runs %d: mean %.2f min %.2f max %.2f tok/s\n", NR, s/NR, mn, mx}}'
    done
done
} >> "$OUT/ab-summary.txt"

echo "AB_RCO_DONE $(date '+%F %T')" >> "$OUT/summary.txt"
echo "ab-rco-iq2xxs: done -> $OUT (summary.txt, ab-summary.txt); start the chat again with ./run.sh."