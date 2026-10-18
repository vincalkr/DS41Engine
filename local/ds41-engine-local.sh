#!/usr/bin/env bash
# local/ds41-engine-local.sh - the engine as Strata's server starts it on THIS machine.
#
#   ds41-engine-local.sh <engine args>        (the server appends its own: GEN/STOP/QUIT line protocol)
#
# Why a wrapper instead of the engine binary directly: the engine speaks the protocol only with --serve, and this
# machine has no systemd user bus for the upstream ds41-engine.sh (memguard.sh's systemd-run), so the RAM cap and
# the GPU lock are left to the caller.  Everything the run needs comes from the config's "env" block
# (models/V41-GSQ/... paths, blob reads through the page cache, the CPU expert tier's host routed sums).
set -euo pipefail
cd "$(dirname "$0")/.."
EXE="${DS41_EXE:-$PWD/build-hip-fused/ds41_generate}"
[ -x "$EXE" ] || { echo "ds41-engine-local: $EXE missing (cmake --build build-hip-fused --target ds41_generate -j)" >&2; exit 1; }
# Strata's server already passes --serve (serve/server.py's popen); only a standalone call needs it added
serve=0
for a in "$@"; do [ "$a" = "--serve" ] && serve=1; done
if [ "$serve" = 1 ]; then exec "$EXE" "$@"; else exec "$EXE" --serve "$@"; fi
