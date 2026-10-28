#!/usr/bin/env bash
# fetch-rco-1017.sh [DEST] - the published GSQ-RCO IQ2_XXS pack (ISTA-DASLab) into the layout the engine runs:
#
#   DEST/../rco-iq2xxs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1..4}-of-00004.gguf   138.2 GB
#   DEST/../rco-iq2xxs/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf                              110.6 GB, the pack's own
#                                                                                          n-gram tables (IQ4_NL)
#   local/rco-iq2xxs-meta/meta.gguf      the 511 KB geometry overlay (tools/v41/ds41_pack_meta.py), written last
#   DEST/../vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf   0.9 GiB, the image encoder, built from the 1017
#                                        checkpoint's vision tower (tools/v41/vision/make_vision_gguf.py; VISION=0 skips)
#
# The image routing bias the pack lacks is in the repo (local/rco-iq2xxs-meta/vl-bias.gguf, 61 KB, from ISTA-DASLab's
# exp_probs_b_vl.f32 = the checkpoint's ffn.gate.bias_vl, by tools/v41/ds41_vl_bias.py), so with the encoder
# ./run.sh starts with images on.
#
# The files come from revision $REV (2026-10-25): on 2026-10-27 the repository switched to llama.cpp's upstream
# six-shard layout, which this engine does not read.
# The engine reads the GGUF's weights, the Engram tables in either form (this IQ4_NL file, or the checkpoint's FP8 rows
# from local/fetch-model.sh) and the overlay through DS41_META, because this pack carries llama.cpp/HF metadata
# instead of the deepseek41.* keys; the overlay is built from local/rco-iq2xxs-meta/geometry-keys.json (in the repo).
#
# Resumable: a file counts as complete when its size matches the Hugging Face API's, and each is checked against the
# API's LFS sha256 once.  Env: CONN=4 (aria2c connections per file), SKIP_SHA=1, HF_TOKEN, DRY_RUN=1.
set -euo pipefail
export LC_ALL=C LANG=C
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$HERE/models/rco-iq2xxs}
REPO=ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
REV=${REV:-653a9874aa38c356233c5f3cc8c96b19a43b57eb}
REV_FIRST=ad1e899190574bee6c66d0112afc00a945737869
BASE=https://huggingface.co/$REPO/resolve/$REV
FILES=${FILES:-"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf
DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf
DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf
DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf
DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"}
CONN=${CONN:-4}
DRY_RUN=${DRY_RUN:-0}
TOKEN=${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "Authorization: Bearer $TOKEN")
mkdir -p "$DEST"

for f in $FILES; do
    meta=$(curl -sf "${AUTH[@]}" "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=false" \
           | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if x['path']=='$f':
        print(x['size'], (x.get('lfs') or {}).get('sha256') or x.get('sha256') or '')
        break")
    [ -n "$meta" ] || { echo "fetch-rco: the API has no $f" >&2; exit 1; }
    want=${meta%% *}; sha=${meta#* }
    cur=$(stat -c%s "$DEST/$f" 2>/dev/null || echo 0)
    if [ "$cur" = "$want" ]; then echo "have   $f ($want B)"; continue; fi
    # a download from the first upload (2026-10-23) is complete too: the later revisions changed only the chat
    # template shard 1 embeds, which the server does not read (it renders with tools/v41/dsv41_encoding.py)
    if [ "$cur" != 0 ]; then
        old=$(curl -sf "${AUTH[@]}" "https://huggingface.co/api/models/$REPO/tree/$REV_FIRST?recursive=false" \
              | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if x['path']=='$f': print(x['size']); break")
        if [ "$cur" = "$old" ]; then echo "have   $f ($cur B, the 2026-10-23 upload)"; continue; fi
    fi
    echo "get    $f ($cur of $want B, $CONN connections)"
    [ "$DRY_RUN" = 1 ] && continue
    aria2c -c --max-concurrent-downloads=1 -x"$CONN" -s"$CONN" -k4M --file-allocation=none \
           --console-log-level=warn --summary-interval=60 --retry-wait=5 -m8 \
           ${TOKEN:+--header="Authorization: Bearer $TOKEN"} -d "$DEST" -o "$f" "$BASE/$f?download=true"
    if [ -n "$sha" ] && [ "${SKIP_SHA:-0}" != 1 ]; then
        echo "$sha  $DEST/$f" | sha256sum -c -
    elif [ -n "$sha" ]; then
        echo "skip   sha256 $f"
    fi
done

# The tokenizer: V4.1's tables as separate files (vocab, merges, token types, chat template), from a repository that
# hosts them, kept beside the pack so a fresh clone has what the server reads.
TOKREPO=IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored
for f in vocab.json merges.txt token_type.json chat_template.jinja; do
    tmeta=$(curl -sf "${AUTH[@]}" "https://huggingface.co/api/models/$TOKREPO/tree/main?recursive=1" \
            | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if x['path']=='tokenizer/$f': print(x['size']); break")
    [ -n "$tmeta" ] || { echo "fetch-rco: the API has no tokenizer/$f" >&2; exit 1; }
    cur=$(stat -c%s "$DEST/tokenizer/$f" 2>/dev/null || echo 0)
    if [ "$cur" = "$tmeta" ]; then echo "have   tokenizer/$f"; continue; fi
    echo "get    tokenizer/$f ($tmeta B)"
    [ "$DRY_RUN" = 1 ] && continue
    mkdir -p "$DEST/tokenizer"
    curl -L --fail --retry 5 --retry-delay 5 -C - ${TOKEN:+-H "Authorization: Bearer $TOKEN"} \
         -o "$DEST/tokenizer/$f" "https://huggingface.co/$TOKREPO/resolve/main/tokenizer/$f?download=true"
    [ "$(stat -c%s "$DEST/tokenizer/$f")" = "$tmeta" ] || { echo "fetch-rco: tokenizer/$f is short - re-run to resume" >&2; exit 1; }
done

META=local/rco-iq2xxs-meta
if [ "$DRY_RUN" != 1 ]; then
    mkdir -p "$HERE/$META"
    # The overlay's architecture keys live in $META/geometry-keys.json (68 keys, 8 KB, in the repo), so this works with
    # only the pack: no reference download. The Engram table row counts come from the pack's own Engram GGUF when it is
    # there, else from the same JSON.
    ENGRAMARG=(); [ -s "$DEST/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" ] && ENGRAMARG=(--engram "$DEST/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf")
    python3 "$HERE/tools/v41/ds41_pack_meta.py" --pack "$DEST"/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf \
        --ref-keys "$HERE/$META/geometry-keys.json" "${ENGRAMARG[@]}" --out "$HERE/$META/meta.gguf"
    python3 "$HERE/tools/v41/engine/v41_pack_map.py" \
        "$DEST"/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf > "$HERE/local/rco-iq2xxs-map.json"
    echo "fetch-rco: pack + tokenizer in $DEST, overlay in $HERE/$META/meta.gguf (the configurations set DS41_META to it)"
    if [ "${VISION:-1}" != 0 ]; then
        VIS="$DEST/../vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf"
        if [ -s "$VIS" ]; then echo "have   the vision GGUF"
        else
            mkdir -p "$(dirname "$VIS")"
            python3 "$HERE/tools/v41/vision/make_vision_gguf.py" --out "$VIS"
        fi
    fi
    echo "fetch-rco: --engram - the configurations name the checkpoint's FP8 rows in models/engram-1017 (203 GB, from"
    echo "           local/fetch-model.sh + local/ds4-engram-extract.py). Without them ./run.sh still starts: the engine"
    echo "           falls back to the pack's own $DEST/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf and prints that it does."
fi