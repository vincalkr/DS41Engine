#!/bin/sh
# DeepSeek-V4.1-Flash (GSQ) on two RX 7900 XTX (gfx1100, ROCm/HIP); one card: ./run-1gpu.sh.
# The model layout, the settings and the measured numbers: local/ds41-local.json and docs/ds4/.
# Extra arguments are passed to the server, for example:  ./run.sh --port 8151
# The repository root is wherever this script lives, so the launcher works from a clone anywhere.
R=$(cd "$(dirname "$0")" && pwd) || exit 1
cd "$R" || exit 1

# STRATA_HOME must be this repository: the shim imports serve/, tools/ and strata_tokenizer from it, and its own
# default is this directory anyway - set here so the launcher works from any working directory.
STRATA_HOME="$R"
export STRATA_HOME

# The server needs the standard library plus pillow, psutil, regex and jinja2 (strata_tokenizer imports regex;
# serve/server.py imports PIL, psutil and jinja2).  This repository's own venv, a venv/ next to it, or any python3
# that has them - no other checkout is involved; STRATA_PY=/path/to/python forces one.
PY=""
for cand in "${STRATA_PY:-}" "$R/.venv/bin/python" "$(command -v python3)" "$R/venv/bin/python"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    if "$cand" -c 'import PIL, psutil, regex, jinja2' 2>/dev/null; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
    echo "run.sh: no python with pillow, psutil, regex and jinja2 found." >&2
    echo "  make one:        python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2" >&2
    echo "  or point at one: STRATA_PY=/path/to/python ./run.sh" >&2
    exit 1
fi
echo "run.sh: STRATA_HOME=$STRATA_HOME, python $PY"

# The model files the configuration names (-m, --engram, its tokenizer) must be there.  --config in the arguments
# replaces the default, as the server takes the last one.
CFG="$R/local/ds41-local.json"
prev=""
for a in "$@"; do [ "$prev" = "--config" ] && CFG="$a"; prev="$a"; done
missing=$("$PY" - "$CFG" "$R" <<'EOF'
import json, os, sys
cfg, root = sys.argv[1], sys.argv[2]
c = json.load(open(cfg))
a = c.get("args", [])
need = [a[a.index("-m") + 1]] if "-m" in a else []
if "--engram" in a: need += a[a.index("--engram") + 1].split(":")
if c.get("tokenizer"): need.append(c["tokenizer"])
for p in need:
    q = p if os.path.isabs(p) else os.path.join(root, p)
    if not os.path.exists(q) or (os.path.isfile(q) and os.path.getsize(q) == 0): print(q)
EOF
)
if [ -n "$missing" ]; then
    echo "run.sh: missing for $CFG:" >&2
    echo "$missing" | sed 's/^/  /' >&2
    echo "fetch the model first - see the README, '1. Get the model files'." >&2
    exit 1
fi

# local/power-tune.sh for as long as the server runs (+11% short answers, +21% at ~25K context, measured 2026-10-21):
# only when local/install-power-tune.sh has been run once with sudo (a root-owned copy and a sudo rule for it alone);
# otherwise nothing changes.  DS41_POWER_TUNE=0 leaves the machine's settings alone.
PT=/usr/local/sbin/ds41e-power-tune
tuned=0
if [ "${DS41_POWER_TUNE:-1}" != 0 ] && [ -x "$PT" ] && sudo -n "$PT" on >/dev/null 2>&1; then
    tuned=1
    echo "run.sh: power-tune on (NVMe APST off, CPU C3 off, performance governor); off again when the server ends"
elif [ "${DS41_POWER_TUNE:-1}" != 0 ] && [ ! -x "$PT" ]; then
    echo "run.sh: power-tune not installed (sudo local/install-power-tune.sh switches it on and off with the server)"
fi

if [ "$tuned" = 0 ]; then
    exec "$PY" "$R/tools/v41/serve/serve_ds41.py" \
         "--engine" "strata" \
         "--config" "$R/local/ds41-local.json" \
         "--port" "8150" "$@"
fi
# The server runs in the foreground (Ctrl+C reaches it as before); this shell waits and switches power-tune off
# however the server ends.  A no-op INT/TERM/HUP trap keeps the shell alive until then (an ignored signal would be
# inherited by the server).
trap ':' INT TERM HUP
rc=0
"$PY" "$R/tools/v41/serve/serve_ds41.py" \
    "--engine" "strata" \
    "--config" "$R/local/ds41-local.json" \
    "--port" "8150" "$@" || rc=$?
# Another ds41e server may have started meanwhile (a restart, an A/B that stops one and starts the next): it switched
# power-tune on and still needs it, so it stays on until the last server ends.
if pgrep -f "[t]ools/v41/serve/serve_ds41.py" >/dev/null 2>&1; then
    echo "run.sh: another ds41e server is running: power-tune stays on"
else
    sudo -n "$PT" off >/dev/null 2>&1 && echo "run.sh: power-tune off"
fi
exit "$rc"
