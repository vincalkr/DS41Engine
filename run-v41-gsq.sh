#!/bin/sh
# DeepSeek-V4.1-Flash (GSQ) on two RX 7900 XTX (gfx1100, ROCm/HIP); one card: ./run-v41-gsq-1gpu.sh.
# The model layout, the settings and the measured numbers: local/ds41-local.json and docs/ds4/.
# Extra arguments are passed to the server, for example:  ./run-v41-gsq.sh --port 8151
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
    echo "run-v41-gsq.sh: no python with pillow, psutil, regex and jinja2 found." >&2
    echo "  make one:        python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2" >&2
    echo "  or point at one: STRATA_PY=/path/to/python ./run-v41-gsq.sh" >&2
    exit 1
fi
echo "run-v41-gsq.sh: STRATA_HOME=$STRATA_HOME, python $PY"

for f in "$R/models/V41-GSQ/V41-GSQ.gguf" \
         "$R/models/engram/layers1/engram-L1.safetensors" \
         "$R/models/engram/layers14/engram-L14.safetensors"; do
    if [ ! -s "$f" ]; then
        echo "run-v41-gsq.sh: $f is missing (or empty)." >&2
        echo "run local/fetch-model.sh first - see the README, 'Getting the model files'." >&2
        exit 1
    fi
done

exec "$PY" "$R/tools/v41/serve/serve_ds41.py" \
     "--engine" "strata" \
     "--config" "$R/local/ds41-local.json" \
     "--port" "8150" "$@"
