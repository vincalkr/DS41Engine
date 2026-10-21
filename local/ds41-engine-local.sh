#!/usr/bin/env bash
# local/ds41-engine-local.sh - the engine as the server starts it (the configs' "exe").
#
#   ds41-engine-local.sh <engine args>        (the server appends its own: GEN/STOP/QUIT line protocol)
#
# It adds --serve for a standalone call and picks the binary (DS41_EXE); everything else comes from the config's
# "args" and "env".
set -euo pipefail
cd "$(dirname "$0")/.."
EXE="${DS41_EXE:-$PWD/build-hip-fused/ds41_generate}"
[ -x "$EXE" ] || { echo "ds41-engine-local: $EXE missing (cmake --build build-hip-fused --target ds41_generate -j)" >&2; exit 1; }
# Strata's server already passes --serve (serve/server.py's popen); only a standalone call needs it added
serve=0
for a in "$@"; do [ "$a" = "--serve" ] && serve=1; done
if [ "$serve" = 1 ]; then exec "$EXE" "$@"; else exec "$EXE" --serve "$@"; fi
