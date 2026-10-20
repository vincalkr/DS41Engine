#!/bin/sh
# local/rebuild-rank.sh - rebuild the GSQ's rank from the calibration corpus plus what the chat has routed.
#
#   local/rebuild-rank.sh [chat-weight]      (default 1: one chat selection counts as one corpus selection)
#
# The chat config runs the engine with --dump-routes local/routes-chat: the server adds every (layer, expert)
# selection it routes and writes local/routes-chat/LNN.json after each request (the counts survive restarts).
# This script sums those counts (times chat-weight) with local/calib-q2/stats-merged, ranks the sum with the GSQ's
# per-expert sizes (local/gsq-alloc-map.json) and writes local/rank-gsq-chat.bin.  The chat config reads
# local/rank-gsq-mixed.bin: to use the new rank, point --rank at the new file (or copy it over) and restart.
# The ds4 1017 Q2 (the default model since 2026-10-20; one blob size, so no map):
#   ROUTES=local/routes-1017 MAP= OUT=local/rank-1017.bin local/rebuild-rank.sh
set -eu
cd "$(dirname "$0")/.."
W="${1:-1}"
ROUTES="${ROUTES:-local/routes-chat}"
MAP="${MAP-local/gsq-alloc-map.json}"
OUT="${OUT:-local/rank-gsq-chat.bin}"
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
