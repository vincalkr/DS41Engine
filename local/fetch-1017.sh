#!/usr/bin/env bash
# fetch-1017.sh - DeepSeek-V4.1-Flash-1017 (the 2026-10-17 checkpoint) for this engine, next to the current model:
#
#   Q/DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf   lxe's 2.03 bpw quant, GSQ format (PTQ1_0 / Q2_0 / Q3_K experts,
#                                               Q8_0 dense), 148 GB from 4 parts      Q   = <repo>/models/V41-1017-2bit by default
#   Q/expert-rank.json, Q/README.md, Q/results/ the quant's expert ranking and its author's measurements
#   E/model-0004{7,8}-of-00048.safetensors      the 1017 Engram tables, the official shards as they are (203 GB):
#   E/model.safetensors.index.json              the engine resolves layers.{1,14}.engram.* through the index, so
#                                               no extraction step (and no second copy) is needed
#                                               E   = <repo>/models/engram-1017 by default
#
#   local/fetch-1017.sh [Q] [E]
#
# Both revisions are pinned.  Every big file is checked against the sha256 Hugging Face lists for it (its LFS oid);
# a file whose size is short is resumed (aria2c -c).  The four parts are checked, then appended one by one and each
# dropped after it is appended (148 GB + one 40 GB part at the peak, not 2 x 148 GB), and the joined file is checked
# against the quant's own SHA256SUMS.  The tokenizer is the same as the September checkpoint's (tokenizer.json and
# tokenizer_config.json have the same sha256), so models/V41-GSQ/tokenizer serves both.
set -euo pipefail
export LC_ALL=C LANG=C
trap 'trap - INT TERM; kill 0 2>/dev/null' INT TERM

HERE=$(cd "$(dirname "$0")/.." && pwd)
Q=${1:-$HERE/models/V41-1017-2bit}
E=${2:-$HERE/models/engram-1017}
QREPO=lxe/DeepSeek-V4.1-Flash-1017-ds41-2bit
QREV=4d60d3af5e40921f7ec4c84d163464b298981813
DREPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
DREV=1bae0e1088993b9b82de08f59de5be5a34e42170
CONN=${CONN:-4}
GGUF=DeepSeek-V4.1-Flash-1017-ds41-2bit.gguf
mkdir -p "$Q" "$E"

say() { echo "$(date '+%F %T') $*"; }
size() { stat -c%s "$1" 2>/dev/null || echo 0; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }

# meta <repo> <rev> <path> -> "<size> <sha256 or empty>"
meta() {
  curl -sSf "https://huggingface.co/api/models/$1/tree/$2?recursive=1" | python3 -c '
import json, sys
p = sys.argv[1]
for f in json.load(sys.stdin):
    if f.get("path") == p: print(f.get("size", 0), (f.get("lfs") or {}).get("oid", "")); break' "$3"
}

# get <repo> <rev> <path> <dir>: download (resuming) until the size matches, then check the sha256 when listed
get() {
  local repo=$1 rev=$2 path=$3 dir=$4 m want sha out
  m=$(meta "$repo" "$rev" "$path"); want=${m%% *}; sha=${m#* }; [ "$sha" = "$m" ] && sha=""
  out="$dir/$path"; mkdir -p "$(dirname "$out")"
  if [ "$(size "$out")" != "$want" ]; then
    say "get   $path ($(size "$out") of $want B, $(free_gb "$dir") GB free in $dir)"
    if [ "$want" -ge 52428800 ]; then
      aria2c -c -x"$CONN" -s"$CONN" -k8M --file-allocation=none --console-log-level=warn --summary-interval=60 \
             --retry-wait=5 -m10 -d "$(dirname "$out")" -o "$(basename "$out")" \
             "https://huggingface.co/$repo/resolve/$rev/$path"
    else
      curl -sSfL --retry 5 -o "$out" "https://huggingface.co/$repo/resolve/$rev/$path"
    fi
    [ "$(size "$out")" = "$want" ] || { say "ERROR $path is $(size "$out") B, expected $want - re-run to resume"; exit 1; }
  fi
  if [ -n "$sha" ] && [ ! -f "$out.sha-ok" ]; then
    say "check $path (sha256)"
    echo "$sha  $out" | sha256sum -c --quiet - || { say "ERROR $path: sha256 mismatch - delete it and re-run"; exit 1; }
    touch "$out.sha-ok"
  fi
  say "have  $path"
}

say "=== the quant -> $Q ($(free_gb "$Q") GB free), the Engram -> $E ($(free_gb "$E") GB free)"
for f in README.md SHA256SUMS expert-rank.json reassemble.sh results/summary.json results/quality-and-bias.log \
         results/quality-decode-path.log results/speed-gen-3000.log results/decode-profile.log; do
  get "$QREPO" "$QREV" "$f" "$Q"
done

if [ -f "$Q/$GGUF.ok" ]; then
  say "have  $GGUF (joined and checked)"
else
  for i in 1 2 3 4; do
    p="$GGUF.part-0$i-of-04"
    [ -f "$Q/$p.joined" ] || get "$QREPO" "$QREV" "$p" "$Q"
  done
  say "=== join (each part dropped once appended)"
  for i in 1 2 3 4; do
    p="$GGUF.part-0$i-of-04"
    [ -f "$Q/$p.joined" ] && continue
    cat "$Q/$p" >> "$Q/$GGUF"
    touch "$Q/$p.joined"; rm -f "$Q/$p" "$Q/$p.sha-ok"
    say "      appended $p ($(( $(size "$Q/$GGUF") / 1000000000 )) GB so far)"
  done
  say "check $GGUF against SHA256SUMS"
  (cd "$Q" && grep -E "  $GGUF\$" SHA256SUMS | sha256sum -c -) || { say "ERROR the joined file does not match"; exit 1; }
  touch "$Q/$GGUF.ok"; rm -f "$Q"/*.joined
fi

say "=== the 1017 Engram shards -> $E"
get "$DREPO" "$DREV" model-00047-of-00048.safetensors "$E"
get "$DREPO" "$DREV" model-00048-of-00048.safetensors "$E"
python3 - "$E" <<'EOF'
import json, struct, sys
d = sys.argv[1]
wm = {}
for s in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
    with open(f"{d}/{s}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    for k in h:   # only the hash tables: the rest of the Engram (q/k/wkv) is in the GGUF, as in models/engram
        if ".engram.embed." in k: wm[k] = s
assert sorted(wm) == sorted(f"layers.{l}.engram.embed.{t}" for l in (1, 14) for t in ("weight", "scale")), sorted(wm)
json.dump({"metadata": {}, "weight_map": wm}, open(f"{d}/model.safetensors.index.json", "w"), indent=1)
print("index:", wm)
EOF
say "=== done"
ls -la "$Q" "$E"
