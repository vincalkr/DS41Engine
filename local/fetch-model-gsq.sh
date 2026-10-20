#!/usr/bin/env bash
# (The September GSQ, the default before 2026-10-20; local/fetch-model.sh now fetches the ds4 1017 Q2.)
# fetch-model.sh [DEST] - fetch DeepSeek-V4.1-Flash (GSQ) into the layout this branch uses:
#
#   DEST/V41-GSQ/V41-GSQ.gguf              the joined 152 GiB model      (from the GSQ repo, 16 parts)
#   DEST/V41-GSQ/V41-MTP-Vision.Q2_K.gguf  the DSpark draft
#   DEST/V41-GSQ/tokenizer/{vocab.json,merges.txt,token_type.json,chat_template.jinja}
#   DEST/V41-GSQ/{rank.bin,V41-GSQ.gguf.experts.json,SHA256SUMS,reassemble.sh}
#   DEST/engram/layers1/engram-L1.safetensors    the layer-1 Engram table  (from the FP8 repo, shard 47)
#   DEST/engram/layers14/engram-L14.safetensors  the layer-14 Engram table (shard 48) + the merged index
#
# DEST defaults to <repo>/models.  Needs curl and python3.
#
# Resumable and checkable: every file's expected size is read from the Hugging Face API, a file counts as complete
# only when its size matches (so a partial download is resumed with curl -C - rather than trusted), and the 16 parts
# are joined by the repo's own reassemble.sh, which checks their sha256 before appending and verifies the joined
# file at the end.  DRY_RUN=1 prints what would be fetched and changes nothing.
#
# Space: the GGUF arrives as 163 GB of parts and the join drops each part as it appends it (DELETE_PARTS=0 keeps
# them and needs the space twice), the Engram is 101 GB per table plus the 101 GB shard it comes from.  On one disk
# that is the default, ONE_AT_A_TIME=1 (fetch shard 47 -> extract layer 1 -> delete 47 -> fetch 48 -> extract layer
# 14 -> delete 48), peak about 163 + 101 + 101 GB.  ONE_AT_A_TIME=0 fetches both shards first (203 GB) and then
# extracts both, so it needs 163 + 203 + 202 GB - only for a machine with room.  DEST_LAYERS1 / DEST_LAYERS14 put
# the two tables on different disks.
#
# Env: GSQ_REPO, FP8_REPO, DELETE_PARTS=1 (0 keeps the parts), KEEP_FP8=0 (1 keeps the FP8 shards after the
#      extraction, 101 GB each), ONE_AT_A_TIME=1 (0 = both shards first, needs 163 + 203 + 202 GB), DRY_RUN=0 (1 = print
#      the plan and change nothing), FORCE_JOIN=0 (1 = remove an interrupted joined file and join again),
#      JOBS=1 and CONN=4 (one file at a time, four connections to it.  Measured on this repo: 1 connection = 5.7
#      MB/s, 4 = ~27 MB/s, 8 = 140 MiB/s; JOBS adds files in parallel), ARIA2=0 to force curl,
#      DEST_LAYERS1, DEST_LAYERS14 (put the two Engram tables on different disks), HF_TOKEN (gated repos only),
#      VERIFY_ROWS=200 (Engram rows checked against the source), WORK=default DEST (scratch for the FP8 shards and
#      the size lists).  The README's "Getting the model files" section has the same table.
set -euo pipefail

# Force English output: aria2c and curl follow the system locale, so without this their summaries come out in
# Italian on this machine (and any wording would change with the locale).
export LC_ALL=C LANG=C

# The parts are fetched as background children, so Ctrl+C must take the whole process group with it: without
# this the curls keep downloading after the script has gone (measured: three of them survived a Ctrl+C).
trap 'trap - INT TERM; kill 0 2>/dev/null' INT TERM

HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$HERE/models}
GSQ_REPO=${GSQ_REPO:-IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored}
FP8_REPO=${FP8_REPO:-dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8}
DELETE_PARTS=${DELETE_PARTS:-0}   # 1 = the author's in-place deletion (tight disk); 0 = keep the parts through the
                                 #     join and remove them only once the joined file has been verified
KEEP_FP8=${KEEP_FP8:-0}
ONE_AT_A_TIME=${ONE_AT_A_TIME:-1}   # 1 = Engram shard by shard (one disk); 0 = both shards first (needs room)
DRY_RUN=${DRY_RUN:-0}
FORCE_JOIN=${FORCE_JOIN:-0}
VERIFY_ROWS=${VERIFY_ROWS:-200}
WORK=${WORK:-$DEST}
TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
AUTH=(); ARIA2_HDR=()
if [ -n "$TOKEN" ]; then AUTH=(-H "Authorization: Bearer $TOKEN"); ARIA2_HDR=(--header="Authorization: Bearer $TOKEN"); fi

# aria2c is about 50x faster here (measured 270 MiB/s with 8 connections against 5.7 MB/s for one curl on the same
# file: the Hugging Face CDN throttles each connection, not the account).  ARIA2=0 forces curl.
ARIA2=${ARIA2:-}; if [ -z "$ARIA2" ]; then if command -v aria2c >/dev/null 2>&1; then ARIA2=1; else ARIA2=0; fi; fi
# Connections per file.  Measured on this repo (Xet CDN): 1 = 5.7 MB/s, 4 = ~27 MB/s, 8 = 140 MiB/s.
CONN=${CONN:-4}

GSQ=$DEST/V41-GSQ
ENGRAM=$DEST/engram
DEST_L1=${DEST_LAYERS1:-$ENGRAM/layers1}
DEST_L14=${DEST_LAYERS14:-$ENGRAM/layers14}
mkdir -p "$GSQ" "$DEST_L1" "$DEST_L14" "$WORK"

say() { echo "$(date '+%F %T') $*"; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
size() { stat -c%s "$1" 2>/dev/null || echo 0; }

# sizes.py <repo> -> "<path> <bytes>" per file, straight from the Hugging Face API (cached in $WORK)
sizes_of() {
  local slug=${1//\//_}
  [ -s "$WORK/.sizes.$slug" ] || curl -sSf "${AUTH[@]}" \
      "https://huggingface.co/api/models/$1/tree/main?recursive=1" \
    | python3 -c 'import json,sys; [print(f["path"], f.get("size", 0)) for f in json.load(sys.stdin) if f.get("type") == "file"]' \
    > "$WORK/.sizes.$slug"
  echo "$WORK/.sizes.$slug"
}
want_of() { awk -v p="$2" '$1 == p { print $2; found = 1 } END { if (!found) print "" }' "$1"; }

# get <sizes-file> <repo> <path-in-repo> <dest-dir>: complete only when the size matches, else resume it.
get() {
  local sizes=$1 repo=$2 path=$3 dest=$4
  local want out
  want=$(want_of "$sizes" "$path")
  out="$dest/$path"
  mkdir -p "$(dirname "$out")"
  if [ -n "$want" ] && [ "$(size "$out")" = "$want" ]; then say "have  $path"; return 0; fi
  say "get   $path (have $(size "$out") of ${want:-?} B, $(( $(free_gb "$dest") )) GB free)"
  [ "$DRY_RUN" = 1 ] && return 0
  url="https://huggingface.co/$repo/resolve/main/$path"
  if [ "$ARIA2" = 1 ] && [ "${want:-0}" -ge 52428800 ]; then   # 8 connections help on big files, drown small ones
    aria2c -c -x$CONN -s$CONN -k4M --file-allocation=none --console-log-level=warn --summary-interval=0 \
           --show-console-readout=true --retry-wait=5 -m5 -d "$(dirname "$out")" -o "$(basename "$out")" "${ARIA2_HDR[@]}" "$url" || true
  else
    curl -L --fail --retry 5 --retry-delay 5 -C - "${AUTH[@]}" -o "$out" "$url"
  fi
  if [ -n "$want" ] && [ "$(size "$out")" != "$want" ]; then
    say "ERROR $path is $(size "$out") B, expected $want - re-run to resume it"; return 1
  fi
  say "done  $path ($(( $(size "$out") / 1000000 )) MB)"
}

say "=== 0/4 the file lists (sizes from the API; DRY_RUN=$DRY_RUN)"
GSQ_SIZES=$(sizes_of "$GSQ_REPO")
FP8_SIZES=$(sizes_of "$FP8_REPO")

say "=== plan"
say "    destination       $DEST"
say "    GGUF from         $GSQ_REPO"
say "    Engram from       $FP8_REPO  (shards 47 -> layer 1, 48 -> layer 14)"
say "    layout            $GSQ  and  $DEST_L1 , $DEST_L14"
say "    fetcher           $([ "$ARIA2" = 1 ] && echo "aria2c, ${CONN:-4} connection(s) per file" || echo curl), ${JOBS:-1} file(s) at a time"
say "    free space        $WORK: $(free_gb "$WORK") GB   ${DEST_L1%/*}: $(free_gb "${DEST_L1}") GB"
say "    DRY_RUN=$DRY_RUN  DELETE_PARTS=$DELETE_PARTS  KEEP_FP8=$KEEP_FP8  ONE_AT_A_TIME=$ONE_AT_A_TIME"

want_gguf=$(awk '/^V41-GSQ\.gguf\.part-/ { s += $2 } END { print s + 0 }' "$GSQ_SIZES")
say "=== 1/4 the GGUF and its small files -> $GSQ"
for f in SHA256SUMS reassemble.sh rank.bin V41-GSQ.gguf.experts.json V41-MTP-Vision.Q2_K.gguf \
         tokenizer/vocab.json tokenizer/merges.txt tokenizer/token_type.json tokenizer/chat_template.jinja; do
  get "$GSQ_SIZES" "$GSQ_REPO" "$f" "$GSQ"
done
# One file at a time, CONN connections to it (4 by default: measured 5.7 MB/s with one, ~27 MB/s with four and 140
# MiB/s with eight here).  JOBS fetches several files at once.  A check pass follows either way, for anything a
# failed job left partial.
JOBS=${JOBS:-1}   # one file at a time
say "      $(awk '/^V41-GSQ[.]gguf[.]part-/ { s += $2 } END { printf "%.0f", s/1e9 }' "$GSQ_SIZES") GB expected in the 16 parts"
export JOBS
if [ -f "$GSQ/V41-GSQ.gguf" ] && [ "$(size "$GSQ/V41-GSQ.gguf")" = "$want_gguf" ]; then
  say "have  V41-GSQ.gguf ($(size "$GSQ/V41-GSQ.gguf") bytes, the complete model): the 16 parts are NOT needed,"
  say "      skipping their download (they are 163 GB).  Delete the file to re-join from the parts."
else
  say "      $JOBS parts at a time"
  n=0
  for part in $(seq -w 1 16); do
    get "$GSQ_SIZES" "$GSQ_REPO" "V41-GSQ.gguf.part-$part-of-16" "$GSQ" &
    n=$((n + 1))
    if [ $((n % JOBS)) -eq 0 ]; then wait; fi
  done
  wait
  for part in $(seq -w 1 16); do get "$GSQ_SIZES" "$GSQ_REPO" "V41-GSQ.gguf.part-$part-of-16" "$GSQ"; done
  say "      parts on disk: $(cat "$GSQ"/V41-GSQ.gguf.part-* 2>/dev/null | wc -c | awk '{printf "%.1f", $1/1e9}') GB"
fi

say "=== 2/4 the 16 parts are in $GSQ"
say "    reassemble.sh checks the sha256 of all 16 parts BEFORE writing anything, appends them in order into"
say "    V41-GSQ.gguf and verifies the joined file at the end (expected $want_gguf bytes, $(awk -v w="$want_gguf" 'BEGIN{printf "%.1f", w/1e9}') GB)"
if [ -f "$GSQ/V41-GSQ.gguf" ] && [ "$(size "$GSQ/V41-GSQ.gguf")" = "$want_gguf" ]; then
  say "have  V41-GSQ.gguf"
elif [ -f "$GSQ/V41-GSQ.gguf" ]; then
  say "V41-GSQ.gguf exists but is $(size "$GSQ/V41-GSQ.gguf") B, expected $want_gguf (an interrupted join?)"
  if [ "$FORCE_JOIN" = 1 ]; then rm -f "$GSQ/V41-GSQ.gguf"; say "removed it (FORCE_JOIN=1)"
  else say "remove it and re-run (the parts still on disk are checked again), or set FORCE_JOIN=1"; exit 1; fi
fi
if [ ! -f "$GSQ/V41-GSQ.gguf" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    say "would join the 16 parts"
  else
    # Keeping the parts through the join is the safe mode (an interruption can be re-run), but it needs the parts AND
    # the joined file on disk at the same time: 2x the model.  With less space, join in place instead - reassemble.sh
    # then drops each part as it appends it, which is tighter but loses the consumed parts if it is interrupted.
    free_now=$(free_gb "$WORK"); need_gb=$(( want_gguf / 1000000000 )); keep_gb=$(( 2 * need_gb + 10 ))
    mode=""
    if [ "$DELETE_PARTS" = 1 ]; then
      mode=inplace
    elif [ "$free_now" -ge "$keep_gb" ]; then
      mode=keep
    elif [ "$free_now" -ge "$(( need_gb + 10 ))" ]; then
      mode=inplace
      say "    NOTE: ${free_now} GB free, joining in place (parts are dropped as they are appended).  To keep them,"
      say "          free ${keep_gb} GB first; an interruption in this mode loses the parts already appended."
    else
      say "ERROR: ${free_now} GB free in $WORK.  The join needs at least $(( need_gb + 10 )) GB, $keep_gb GB to keep the parts."
      say "       Free space, or put the model on another disk (the first argument is the destination)."
      exit 1
    fi
    say "    join mode: $mode"
    t0=$(date +%s)
    if [ "$mode" = inplace ]; then
      bash "$GSQ/reassemble.sh" "$GSQ" --delete-parts
    else
      bash "$GSQ/reassemble.sh" "$GSQ"
    fi
    say "    join finished in $(( $(date +%s) - t0 )) s; free space now $(free_gb "$WORK") GB"
    if [ "$(size "$GSQ/V41-GSQ.gguf")" = "$want_gguf" ]; then
      say "    V41-GSQ.gguf is $(size "$GSQ/V41-GSQ.gguf") bytes, as expected: the model is complete"
      if [ "$mode" = keep ]; then
        say "    removing the 16 parts and the aria2c control files, the joined file is verified"
        rm -f "$GSQ"/V41-GSQ.gguf.part-* "$GSQ"/*.aria2
        say "    parts removed; free space now $(free_gb "$WORK") GB"
      fi
    else
      say "    ERROR: the joined file is $(size "$GSQ/V41-GSQ.gguf") bytes, expected $want_gguf - leaving everything, re-run"
      exit 1
    fi
  fi
fi

say "=== 3/4 the two Engram tables (they are NOT in the GSQ repo: shards 47/48 of $FP8_REPO)"
drop_shard() {  # drop one shard and its marker; the index stays, the next layer still needs it
  [ "$KEEP_FP8" = 1 ] && return 0
  rm -f "$WORK"/fp8/model-000$1-of-00048.safetensors* 2>/dev/null || true
}
clean_fp8() {   # at the end: the index and fetch_v41.sh's markers go too, so nothing is left behind
  [ "$KEEP_FP8" = 1 ] && return 0
  rm -f "$WORK"/fp8/model-*.safetensors* "$WORK"/fp8/model.safetensors.index.json "$WORK"/fp8/.tree.json "$WORK"/fp8/.files.tsv "$WORK"/fp8/*.ok 2>/dev/null || true
  rmdir "$WORK/fp8" 2>/dev/null || true
}
extract_shard() {  # extract_shard <shard> <layer> <dest-layers>
  local shard=$1 layer=$2 dest=$3
  local table="$dest/engram-L$layer.safetensors"
  if [ "$(size "$table")" -gt 100000000000 ]; then say "have  engram-L$layer.safetensors"; return 0; fi
  [ "$DRY_RUN" = 1 ] && { say "would fetch shard $shard and extract layer $layer into $dest"; return 0; }
  # extract_engram.py validates every shard the index names, so both must be present even to extract one table
  # (measured: "shard not found: model-00048-of-00048.safetensors" with only 47 on disk).
  get "$FP8_SIZES" "$FP8_REPO" "model.safetensors.index.json" "$WORK/fp8"
  get "$FP8_SIZES" "$FP8_REPO" "model-00047-of-00048.safetensors" "$WORK/fp8"
  get "$FP8_SIZES" "$FP8_REPO" "model-00048-of-00048.safetensors" "$WORK/fp8"
  say "    extracting layer $layer from shard $shard into $dest (verifying $VERIFY_ROWS random rows), free there: $(free_gb "$dest") GB"
  python3 "$HERE/tools/v41/extract_engram.py" "$WORK/fp8" --to "layers.$layer=$dest" --verify "$VERIFY_ROWS"
  # the shards stay: the other layer still needs them (they go at the end, in clean_fp8)
}
# extract_engram.py works on the whole set: it validates every shard the index names and needs a --to for every
# Engram layer, so both shards and both destinations go in ONE call (measured otherwise: "shard not found:
# model-00048-of-00048.safetensors" with only 47 present, and "no --to for layers.14" with only one --to).
if [ "$(size "$DEST_L1/engram-L1.safetensors")" -gt 100000000000 ] && [ "$(size "$DEST_L14/engram-L14.safetensors")" -gt 100000000000 ]; then
  say "have  both Engram tables"
elif [ "$DRY_RUN" = 1 ]; then
  say "would fetch shards 47 and 48 and extract layers 1 and 14"
else
  get "$FP8_SIZES" "$FP8_REPO" "model.safetensors.index.json" "$WORK/fp8"
  get "$FP8_SIZES" "$FP8_REPO" "model-00047-of-00048.safetensors" "$WORK/fp8"
  get "$FP8_SIZES" "$FP8_REPO" "model-00048-of-00048.safetensors" "$WORK/fp8"
  say "    extracting layers 1 and 14 into $DEST_L1 and $DEST_L14 (verifying $VERIFY_ROWS random rows)"
  python3 "$HERE/tools/v41/extract_engram.py" "$WORK/fp8" --to "layers.1=$DEST_L1" --to "layers.14=$DEST_L14" --verify "$VERIFY_ROWS"
  clean_fp8
fi

say "=== 4/4 the merged index the engine reads (it resolves every tensor across all --engram dirs)"
if [ "$DRY_RUN" = 1 ]; then say "would write the merged index in $DEST_L1 and $DEST_L14"
else
  say "    writing the merged model.safetensors.index.json in $DEST_L1 and $DEST_L14 (the engine reads the first one"
  say "    and resolves every tensor across both directories)"
  python3 "$HERE/local/engram_index.py" "$DEST_L1" "$DEST_L14"
  for d in "$DEST_L1" "$DEST_L14"; do
    say "    $d: $(grep -c engram "$d/model.safetensors.index.json" 2>/dev/null || echo 0) engram tensors in the index, $(wc -c < "$d/model.safetensors.index.json" 2>/dev/null || echo 0) bytes"
  done
fi

say "=== layout under $DEST"
du -sh "$GSQ" "$DEST_L1" "$DEST_L14" 2>/dev/null
ls -1 "$GSQ" "$DEST_L1" "$DEST_L14" | sed '/^$/d'

say "=== done.  Point the config at these two directories, for example:"
say "    --engram $DEST_L1:$DEST_L14"
say "    -m $GSQ/V41-GSQ.gguf  --rank $GSQ/rank.bin"
say "(local/ds41-local.json is our configuration; local/cfg.py edits it.)"
