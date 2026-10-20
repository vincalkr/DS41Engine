#!/bin/sh
# local/rebuild-rank.sh - rebuild the expert rank from the calibration corpus plus what the chat has routed.
#
#   local/rebuild-rank.sh [chat-weight]      (default 1: one chat selection counts as one corpus selection)
#
# The default configuration runs the engine with --dump-routes local/routes-1017: the server adds every (layer,
# expert) selection it routes and writes LNN.json there after each request (the counts survive restarts).  This
# script sums those counts (times chat-weight) with local/calib-q2/stats-merged, ranks the sum and writes
# local/rank-1017.bin.  It does not apply it: local/update-rank.sh measures it against the rank in use and applies it
# only if faster.  ROUTES, MAP (per-expert sizes, for a mixed-size quant) and OUT override the defaults.
set -eu
cd "$(dirname "$0")/.."
W="${1:-1}"
ROUTES="${ROUTES:-local/routes-1017}"
MAP="${MAP-}"
OUT="${OUT:-local/rank-1017.bin}"
STATS="${OUT%.bin}.stats"
[ -f "$ROUTES/L00.json" ] || { echo "no chat routes yet in $ROUTES (start the chat, use it)" >&2; exit 1; }
python3 - "$W" "$ROUTES" "$STATS" <<'EOF'
import json, os, sys
w = float(sys.argv[1])
L, NE = 40, 384
routes, out = sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
tot_c = tot_r = 0
for l in range(L):
    a = json.load(open(f"local/calib-q2/stats-merged/L{l:02d}.json"))["experts"]
    b = json.load(open(f"{routes}/L{l:02d}.json"))["experts"]
    ex = {}
    for e in range(NE):
        na, nb = int(a[str(e)]["n"]), int(b[str(e)]["n"])
        tot_c += na; tot_r += nb
        ex[str(e)] = {"n": na + w * nb}
    json.dump({"layer": l, "top_k": 6, "experts": ex}, open(f"{out}/L{l:02d}.json", "w"))
print(f"corpus {tot_c} selections + chat {tot_r} x {w} -> {out}")
EOF
python3 tools/v41/engine/v41_rank.py --stats "$STATS" ${MAP:+--map "$MAP"} --out "$OUT"
