#!/usr/bin/env bash
# local/exact-sweep.sh (2026-10-17): velocità e fedeltà della modalità exact e delle sue varianti, stesso prompt
# (circles.i32), greedy, 200 token.  Per ogni run: tok/s e quanti token coincidono con la run exact.
#   CTX=32768 (default; la chat usa 262144)   Q2=<gguf>   OUT=<cartella>
# Ferma la chat su 8150 (tiene l'arena).  ~2-4 minuti per run.
set -u
cd "$(dirname "$0")/.."
Q2="${Q2:-/run/media/vincenzo/SSD NVME SAMSUNG/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf}"
OUT="${OUT:-local/sweep-$(date +%m%d-%H%M)}"
CTX="${CTX:-32768}"
PY=/run/media/vincenzo/Dati/projects/other/Strata/.venv/bin/python
mkdir -p "$OUT" ~/.quetza-data/conductor
fuser -k 8150/tcp 2>/dev/null; sleep 2
base=(./build-hip-fused/ds41_generate -m "$Q2" --ids-file local/prompts/circles.i32 -n 200 --temp 0
      --backend cuda --experts gpu --arena-gib 90 --ctx "$CTX" --kv-q8 --threads 24 --rank local/rank-q2.bin
      --engram models/engram/layers14:models/engram/layers1 --arena-skip-vram)
run() {   # run TAG "VAR=x VAR=y" [extra args]
  local tag=$1 envs=$2; shift 2
  env $envs bash tools/ds4/memguard.sh 110 82 -- "${base[@]}" "$@" > "$OUT/$tag.out" 2>&1
  grep -E '^[0-9]+$' "$OUT/$tag.out" > "$OUT/$tag.ids"
  local tps; tps=$(grep 'decode:' "$OUT/$tag.out" | tail -1)
  local agree="(riferimento)"
  if [ "$tag" != exact ] && [ -s "$OUT/exact.ids" ]; then
    agree=$(paste -d' ' "$OUT/exact.ids" "$OUT/$tag.ids" | awk '{n++; if($1==$2) s++; else if(!d) d=n}
            END{printf "%d/%d token uguali a exact, prima differenza: %s", s, n, (d ? "token " d : "nessuna")}')
  fi
  printf '%-12s | %s | %s\n' "$tag" "$tps" "$agree" | tee -a "$OUT/summary.txt"
  "$PY" local/detok.py < "$OUT/$tag.ids" > "$OUT/$tag.txt" 2>/dev/null
}
run exact      ""                                   --route-bias 0 --route-bias-file 0
run exact-prof "STRATA_DEC_PROF=1"                  --route-bias 0 --route-bias-file 0
run spec4      "STRATA_SPEC_TOP=4 STRATA_SPEC_AHEAD=2" --route-bias 0 --route-bias-file 0
run spec8      "STRATA_SPEC_TOP=8 STRATA_SPEC_AHEAD=3" --route-bias 0 --route-bias-file 0
for d in 0 0.01 0.02 0.05; do
  run "margin$d" "STRATA_RB_MARGIN=$d"              --route-bias 0.5 --route-bias-file 0.3
done
run bias-flat  ""                                   --route-bias 0.5 --route-bias-file 0.3
echo; echo "riepilogo: $OUT/summary.txt   testi: $OUT/*.txt   profilo: grep 'decode step' $OUT/exact-prof.out"
