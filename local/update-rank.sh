#!/usr/bin/env bash
# update-rank.sh - rebuild the expert rank from what the chat has routed, measure it against the rank in use, and
# apply it only if it measured better.
#
#   local/update-rank.sh            (about 25 minutes with the defaults; it needs the GPUs)
#
# 1. local/rebuild-rank.sh: the September Q2's calibration corpus plus the chat's routing (ROUTES, default
#    local/routes-1017, which the server fills while it runs) -> local/rank-candidate.bin
# 2. local/bench-v41/ab-1017.sh: the rank in use (local/ds41-local.json's --rank) and the candidate, interleaved, REPS
#    runs each (default 3) at 17K context and on Promessi Sposi, the default configuration's flags
# 3. the candidate is applied only if neither prompt's mean is more than 2% slower and the two means together are at
#    least 1% faster (single runs here spread by 5-10%): it is copied to local/rank-1017.bin and the three default
#    configurations (local/ds41-local.json, -2gpu, -1gpu) get --rank local/rank-1017.bin.  Otherwise nothing changes.
# 4. the decision is appended to local/sweep-1017/summary.txt.
#
# A running server (127.0.0.1:8150) is stopped for the measurement and started again afterwards with
# ./run.sh (the default configuration); NO_RESTART=1 leaves it stopped.  Env: ROUTES, REPS, CHAT_WEIGHT (the
# chat's weight against the corpus, default 1).
set -euo pipefail
cd "$(dirname "$0")/.."
ROUTES=${ROUTES:-local/routes-1017}
REPS=${REPS:-3}
CAND=local/rank-candidate.bin
STAMP=$(date +%Y%m%d-%H%M)
LOG=local/sweep-1017/summary.txt
CUR=$(python3 -c 'import json; a = json.load(open("local/ds41-local.json"))["args"]; print(a[a.index("--rank") + 1])')
say() { echo "$(date '+%F %T') update-rank: $*"; }

[ -f "$ROUTES/L00.json" ] || { say "no routing in $ROUTES yet: use the server first"; exit 1; }
say "rank in use: $CUR; candidate from the corpus + $ROUTES"
ROUTES="$ROUTES" MAP= OUT="$CAND" sh local/rebuild-rank.sh "${CHAT_WEIGHT:-1}" | tail -3
if cmp -s "$CAND" "$CUR"; then say "the candidate is the rank in use: nothing to measure"; exit 0; fi

was_running=0
if pgrep -f "[d]s41_generate --serve" >/dev/null; then
    was_running=1
    say "stopping the server for the measurement"
    fuser -k 8150/tcp >/dev/null 2>&1 || true
    sleep 3
    pkill -x ds41_generate || true
    sleep 5
fi
restart() {
    if [ "$was_running" = 1 ] && [ -z "${NO_RESTART:-}" ]; then
        say "starting the server again (./run.sh)"
        setsid ./run.sh > local/serve-restart.log 2>&1 < /dev/null &
    fi
}
trap restart EXIT

for r in $(seq 1 "$REPS"); do
    REPS=1 TAG=ur$STAMP-cur-$r RANK="$CUR" bash local/bench-v41/ab-1017.sh > /dev/null
    REPS=1 TAG=ur$STAMP-new-$r RANK="$CAND" bash local/bench-v41/ab-1017.sh > /dev/null
done

verdict=$(python3 - "$STAMP" <<'EOF'
import glob, re, statistics as st, sys
stamp = sys.argv[1]
def mean(prompt, arm):
    v = []
    for p in glob.glob(f"local/sweep-1017/{prompt}-ur{stamp}-{arm}-*.out"):
        m = re.search(r"^decode: .*?= ([\d.]+) tok/s", open(p).read(), re.M)
        if m: v.append(float(m.group(1)))
    return st.mean(v) if v else 0.0
r = {p: (mean(p, "cur"), mean(p, "new")) for p in ("huge17k", "sposi")}
ok = all(c > 0 for c, _ in r.values())
ratios = [n / c for c, n in r.values()] if ok else [0]
apply = ok and min(ratios) >= 0.98 and sum(ratios) / len(ratios) >= 1.01
print(("APPLY" if apply else "KEEP") + " " + " ".join(f"{p} {c:.2f}->{n:.2f}" for p, (c, n) in r.items()))
EOF
)
say "$verdict"
if [ "${verdict%% *}" = APPLY ]; then
    cp "$CAND" local/rank-1017.bin
    python3 - <<'EOF'
import json
for f in ("local/ds41-local.json", "local/ds41-local-2gpu.json", "local/ds41-local-1gpu.json"):
    c = json.load(open(f)); a = c["args"]; a[a.index("--rank") + 1] = "local/rank-1017.bin"
    json.dump(c, open(f, "w"), indent=1); open(f, "a").write("\n")
EOF
    say "applied: local/rank-1017.bin in the three default configurations"
else
    say "kept $CUR"
fi
echo "update-rank $STAMP: $verdict (in use $CUR, candidate from $ROUTES)" >> "$LOG"
