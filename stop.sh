#!/bin/sh
# Stops the ds41e server(s) of this folder that ./run.sh started, the way Ctrl+C would: the server asks its engine to
# quit, run.sh switches power-tune off once no server is left.  Waits up to TIMEOUT seconds (default 60) for the server
# and its engine to end; --force then kills what is left.
#   ./stop.sh            stop and wait
#   ./stop.sh --force    the same, then kill a server or engine still running after TIMEOUT
R=$(cd "$(dirname "$0")" && pwd) || exit 1
FORCE=0; [ "${1:-}" = --force ] && FORCE=1
TIMEOUT=${TIMEOUT:-60}

# this folder's servers and engines only (another checkout's are left alone)
servers() { pgrep -f "$R/tools/v41/serve/serve_ds41[.]py" 2>/dev/null; }
engines() {
    for p in $(pgrep -x ds41_generate 2>/dev/null); do
        [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$R" ] && echo "$p"
    done
}

pids=$(servers)
if [ -z "$pids" ] && [ -z "$(engines)" ]; then
    echo "stop.sh: no ds41e server is running"
    exit 0
fi
for p in $pids; do kill -INT "$p" 2>/dev/null; done
echo "stop.sh: stopping the server ($(echo $pids | tr ' ' ','))"
i=0
while [ "$i" -lt "$TIMEOUT" ]; do
    [ -z "$(servers)" ] && [ -z "$(engines)" ] && break
    sleep 1
    i=$((i + 1))
done
left="$(servers) $(engines)"
if [ -z "$(echo $left)" ]; then
    echo "stop.sh: stopped in ${i} s"
    exit 0
fi
if [ "$FORCE" = 1 ]; then
    for p in $left; do kill -KILL "$p" 2>/dev/null; done
    echo "stop.sh: still running after ${TIMEOUT} s: killed $(echo $left | tr ' ' ',')"
    # run.sh may have been killed with them: switch power-tune off here if it is installed
    [ -x /usr/local/sbin/ds41e-power-tune ] && sudo -n /usr/local/sbin/ds41e-power-tune off >/dev/null 2>&1 &&
        echo "stop.sh: power-tune off"
    exit 0
fi
echo "stop.sh: still running after ${TIMEOUT} s ($(echo $left | tr ' ' ',')); ./stop.sh --force kills it" >&2
exit 1
