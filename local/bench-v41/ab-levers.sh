#!/usr/bin/env bash
# ab-levers.sh - A/B of two chat-config levers against the current config (local/ds41-local.json, minus
# --dump-routes): the rank built from the chat's own routes (local/rank-gsq-chat.bin, local/rebuild-rank.sh) on the
# agent replay, and STRATA_SERVE_STEP_MAX 256 -> 96 on mid-size follow-ups (followup-sizes.py).  A fresh server per
# run on 127.0.0.1:8150; engine log segments in local/bench-v41/levers-logs/TAG.log.
cd "$(dirname "$0")/../.."
mkdir -p local/bench-v41/levers-logs
mkcfg() {   # mkcfg OUT RANK [K=V ...]
    python3 - "$@" <<'PY'
import json, sys
out, rank = sys.argv[1:3]
c = json.load(open("local/ds41-local.json"))
a = c["args"]
if "--dump-routes" in a:
    i = a.index("--dump-routes"); del a[i:i + 2]
a[a.index("--rank") + 1] = rank
c["env"].update(dict(kv.split("=", 1) for kv in sys.argv[3:]))
json.dump(c, open(out, "w"), indent=1)
PY
}
stop() { fuser -k 8150/tcp >/dev/null 2>&1; sleep 3; pkill -x ds41_generate; sleep 5; }
run() {   # run TAG BENCH RANK [K=V ...]
    local tag=$1 bench=$2; shift 2
    local cfg=local/bench-v41/levers-logs/cfg-$tag.json
    mkcfg $cfg "$@"; stop
    local n0=$(wc -l < local/serve-engine.log)
    STRATA_HOME=$PWD setsid nohup .venv/bin/python tools/v41/serve/serve_ds41.py --engine strata --config $PWD/$cfg \
        --port 8150 > local/bench-v41/levers-logs/$tag.serve.log 2>&1 < /dev/null &
    sleep 30
    timeout 900 bash -c 'until tail -3 local/serve-engine.log | grep -q "serve: ready"; do sleep 5; done'
    if [ "$bench" = replay ]; then python3 local/bench-v41/agent-replay.py $tag 15; else python3 local/bench-v41/followup-sizes.py $tag; fi
    sleep 2; tail -n +$((n0 + 1)) local/serve-engine.log > local/bench-v41/levers-logs/$tag.log
}
M=local/rank-gsq-mixed.bin; C=local/rank-gsq-chat.bin
for r in 1 2; do
    run step256-r$r sizes $M
    run step96-r$r sizes $M STRATA_SERVE_STEP_MAX=96
    run rankmix-r$r replay $M
    run rankchat-r$r replay $C
done
stop
echo LEVERSDONE >> local/bench-v41/agent-replay.txt
