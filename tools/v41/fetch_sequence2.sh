#!/bin/bash
# fetch_sequence2.sh - Engram-first download order matched to the GSQ pipeline.
#   47 (layer-1 Engram, Crucible) is the only Engram shard GSQ needs for layers 0-13, so fetch the
#   backbone shards for those layers (5-16 = layers 2-13) FIRST so GSQ can run while downloading,
#   then pack 48 (layer-14 Engram, Team CX2) in well before layer 14, then the rest of the backbone
#   (17-46 = layers 14-39 + head/MTP).  Log: $MAIN/sequence2.log
set -u
F=~/AI/Strata-DS4/tools/v41/fetch_v41.sh
MAIN=/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8
E47=/media/mal/Crucible/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8
E48="/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
LOG=$MAIN/sequence2.log
log() { local m; m="$(date '+%F %T') $*"; echo "$m"; echo "$m" >> "$LOG"; }

log "seq2: waiting for shard 47 (layer-1 Engram) at $E47"
until [ -f "$E47/model-00047-of-00048.safetensors.ok" ]; do
  if ! pgrep -f "fetch_v41.sh $E47" >/dev/null 2>&1; then   # old sequence killed -> take over 47 ourselves
    log "seq2: no active 47 fetch -> (re)starting it"
    ( cd "$E47" && JOBS=1 FIT=1 MARGIN_GB=10 ONLY='model-00047-of' "$F" "$E47" ) || log "WARN 47 rc=$?"
  fi
  sleep 60
done

log "seq2: 47 verified -> backbone layers 2-13 (shards 5-16)"
( cd "$MAIN" && JOBS=1 FIT=1 MARGIN_GB=10 ONLY='model-000(0[5-9]|1[0-6])-of' "$F" "$MAIN" ) || log "WARN 5-16 rc=$?"

log "seq2: shard 48 (layer-14 Engram) -> $E48"
mkdir -p "$E48"
( cd "$E48" && JOBS=1 FIT=1 MARGIN_GB=10 ONLY='model-00048-of' "$F" "$E48" ) || log "WARN 48 rc=$?"

log "seq2: remaining backbone (shards 17-46, Engram skipped) -> $MAIN"
( cd "$MAIN" && JOBS=1 FIT=1 MARGIN_GB=10 SKIP='model-0004[78]-of' "$F" "$MAIN" ) || log "WARN rest rc=$?"
log "seq2: done"
