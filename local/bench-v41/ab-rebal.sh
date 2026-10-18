#!/usr/bin/env bash
# A/B on the agent replay (25K first turn + 4 follow-ups, 15 s between turns): the build before the upstream
# commits (local/bench-v41/ds41_generate.base), the new build with the conversation-adaptive arena off, and on.
# Each run is a fresh server on 127.0.0.1:8150 with the chat config minus --dump-routes (the chat's own route
# counts stay untouched).  Per run: local/bench-v41/agent-replay.txt lines, the answers in agent-replay-TAG.json,
# the engine log segment in local/bench-v41/rebal-logs/TAG.log.
cd "$(dirname "$0")/../.."
mkdir -p local/bench-v41/rebal-logs
mkcfg() {   # mkcfg OUT EXE REBAL [K=V ...]
    python3 - "$@" <<'EOF'
import json, sys
out, exe, rebal = sys.argv[1:4]
extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
c = json.load(open("local/ds41-local.json"))
a = c["args"]
if "--dump-routes" in a:
    i = a.index("--dump-routes"); del a[i:i + 2]
c["env"]["DS41_EXE"] = exe
c["env"]["STRATA_ARENA_REBAL"] = rebal
c["env"].update(extra)
json.dump(c, open(out, "w"), indent=1)
EOF
}
stop() { fuser -k 8150/tcp >/dev/null 2>&1; sleep 3; pkill -x ds41_generate; sleep 5; }
run() {   # run TAG EXE REBAL [K=V ...]
    local cfg=local/bench-v41/cfg-$1.json
    mkcfg $cfg "${@:2}"
    stop
    local n0=$(wc -l < local/serve-engine.log)
    STRATA_HOME=$PWD nohup .venv/bin/python tools/v41/serve/serve_ds41.py --engine strata --config $PWD/$cfg --port 8150 \
        > local/bench-v41/rebal-logs/$1.serve.log 2>&1 &
    sleep 30
    timeout 900 bash -c 'until tail -3 local/serve-engine.log | grep -q "serve: ready"; do sleep 5; done'
    python3 local/bench-v41/agent-replay.py $1 15
    sleep 2
    tail -n +$((n0 + 1)) local/serve-engine.log > local/bench-v41/rebal-logs/$1.log
}
if [ "${1:-}" = p512 ]; then   # third round: the lazy prestage only below 512 rows
    for r in 5 6; do run p512-r$r build-hip-fused/ds41_generate 0 STRATA_PRESTAGE_MIN=512; done
elif [ "${1:-}" = pmin ]; then   # second round: the lazy prestage off (STRATA_PRESTAGE_MIN=0) against base and off
    for r in 3 4; do
        run base-r$r local/bench-v41/ds41_generate.base 0
        run pmin0-r$r build-hip-fused/ds41_generate 0 STRATA_PRESTAGE_MIN=0
        run off-r$r build-hip-fused/ds41_generate 0
    done
else
    for r in 1 2; do
        run base-r$r local/bench-v41/ds41_generate.base 0
        run off-r$r build-hip-fused/ds41_generate 0
        run on-r$r build-hip-fused/ds41_generate 1
    done
fi
stop
echo ABDONE >> local/bench-v41/agent-replay.txt
