#!/usr/bin/env bash
# fetch-model.sh [DEST] - DeepSeek-V4.1-Flash-1017 as antirez/ds4's Q2 (IQ2_XXS gate/up, Q2_K down, imatrix) into the
# layout the default configuration (local/ds41-local.json) reads:
#
#   DEST/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf   366 GB (from antirez/deepseek-v4.1-flash-gguf)
#   DEST/ds4-1017/tokenizer/{vocab.json,merges.txt,token_type.json,chat_template.jinja}
#   DEST/engram-1017/layers1/engram-L1.safetensors     the two Engram tables, extracted from the GGUF
#   DEST/engram-1017/layers14/engram-L14.safetensors   (local/ds4-engram-extract.py, 101 GB each) + their index
#
# DEST defaults to <repo>/models.  Needs curl, python3 with numpy (the extraction), and aria2c for a fast download.
# Space: 366 GB for the GGUF + 203 GB for the tables = ~570 GB, all of it kept (the engine reads the GGUF's weights
# and the tables' rows).  Resumable: a file counts as complete when its size matches the Hugging Face API's; the GGUF
# is checked against its sha256 (the API's LFS id) once.  The tokenizer tables are V4.1's (the GGUF's tokenizer
# metadata is the September checkpoint's, byte for byte) and come from the GSQ repo, where they are separate files.
#
# Env: CONN=4 (aria2c connections per file), ARIA2=0 (force curl), DRY_RUN=1 (print the plan, change nothing),
#      SKIP_SHA=1 (skip the 366 GB sha256), HF_TOKEN, VISION=1 (also the image encoder: DEST/vision-1017, 0.9 GiB,
#      built from the 1017 checkpoint by tools/v41/vision/make_vision_gguf.py; docs/ds4/V41_VISION.md).
set -euo pipefail
export LC_ALL=C LANG=C
trap 'trap - INT TERM; kill 0 2>/dev/null' INT TERM

HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$HERE/models}
Q2_REPO=antirez/deepseek-v4.1-flash-gguf
Q2_FILE=DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
TOK_REPO=IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
CONN=${CONN:-4}
DRY_RUN=${DRY_RUN:-0}
TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
AUTH=(); ARIA2_HDR=()
if [ -n "$TOKEN" ]; then AUTH=(-H "Authorization: Bearer $TOKEN"); ARIA2_HDR=(--header="Authorization: Bearer $TOKEN"); fi
ARIA2=${ARIA2:-}; if [ -z "$ARIA2" ]; then if command -v aria2c >/dev/null 2>&1; then ARIA2=1; else ARIA2=0; fi; fi
PY=python3; [ -x "$HERE/.venv/bin/python" ] && "$HERE/.venv/bin/python" -c 'import numpy' 2>/dev/null && PY="$HERE/.venv/bin/python"

Q2=$DEST/ds4-1017
ENG=$DEST/engram-1017
mkdir -p "$Q2/tokenizer" "$ENG"
say() { echo "$(date '+%F %T') $*"; }
size() { stat -c%s "$1" 2>/dev/null || echo 0; }
meta() {   # meta <repo> <path> -> "<size> <sha256 or ->"
  curl -sSf "${AUTH[@]}" "https://huggingface.co/api/models/$1/tree/main?recursive=1" | "$PY" -c '
import json, sys
for f in json.load(sys.stdin):
    if f.get("path") == sys.argv[1]: print(f.get("size", 0), (f.get("lfs") or {}).get("oid", "-"))' "$2"
}
get() {   # get <repo> <path-in-repo> <dest-file> <size>
  local repo=$1 path=$2 out=$3 want=$4
  if [ "$(size "$out")" = "$want" ]; then say "have  $path"; return 0; fi
  say "get   $path ($(size "$out") of $want B)"
  [ "$DRY_RUN" = 1 ] && return 0
  local url="https://huggingface.co/$repo/resolve/main/$path"
  if [ "$ARIA2" = 1 ] && [ "$want" -ge 52428800 ]; then
    aria2c -c -x"$CONN" -s"$CONN" -k4M --file-allocation=none --console-log-level=warn --summary-interval=60 \
           --retry-wait=5 -m5 -d "$(dirname "$out")" -o "$(basename "$out")" "${ARIA2_HDR[@]}" "$url"
  else
    curl -L --fail --retry 5 --retry-delay 5 -C - "${AUTH[@]}" -o "$out" "$url"
  fi
  [ "$(size "$out")" = "$want" ] || { say "ERROR $path is $(size "$out") B, expected $want - re-run to resume"; return 1; }
  say "done  $path"
}

say "=== 1/3 tokenizer -> $Q2/tokenizer"
for f in vocab.json merges.txt token_type.json chat_template.jinja; do
  read -r s _ < <(meta "$TOK_REPO" "tokenizer/$f")
  get "$TOK_REPO" "tokenizer/$f" "$Q2/tokenizer/$f" "$s"
done

say "=== 2/3 the GGUF -> $Q2 ($(df -BG --output=avail "$DEST" | tail -1 | tr -dc 0-9) GB free; it needs 366 GB, the tables 203 more)"
read -r want sha < <(meta "$Q2_REPO" "$Q2_FILE")
get "$Q2_REPO" "$Q2_FILE" "$Q2/$Q2_FILE" "$want"
if [ "$DRY_RUN" != 1 ] && [ -z "${SKIP_SHA:-}" ] && [ ! -f "$Q2/$Q2_FILE.sha-ok" ]; then
  say "      sha256 of the GGUF (expected $sha) ..."
  [ "$(sha256sum "$Q2/$Q2_FILE" | cut -d' ' -f1)" = "$sha" ] || { say "ERROR sha256 mismatch: delete the file and re-run"; exit 1; }
  touch "$Q2/$Q2_FILE.sha-ok"; say "      sha256 ok"
fi

say "=== 3/3 the Engram tables -> $ENG"
if [ "$(size "$ENG/layers1/engram-L1.safetensors")" -gt 100000000000 ] && [ "$(size "$ENG/layers14/engram-L14.safetensors")" -gt 100000000000 ] \
   && [ -f "$ENG/layers1/model.safetensors.index.json" ]; then
  say "have  both tables"
elif [ "$DRY_RUN" != 1 ]; then
  "$PY" "$HERE/local/ds4-engram-extract.py" "$Q2/$Q2_FILE" "$ENG"
fi
# optional: the image encoder (VISION=1) - the 1017 checkpoint's vision tower as a GGUF, 0.9 GiB of HTTP range reads
if [ "${VISION:-0}" = 1 ]; then
  say "=== vision encoder -> $DEST/vision-1017 (VISION=1)"
  if [ -s "$DEST/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf" ]; then say "have  the vision GGUF"
  elif [ "$DRY_RUN" != 1 ]; then
    python3 "$HERE/tools/v41/vision/make_vision_gguf.py" --out "$DEST/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf"
  fi
fi
say "=== done: ./run.sh starts the server on it (local/ds41-local.json)"
