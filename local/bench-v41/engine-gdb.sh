#!/usr/bin/env bash
# The chat engine under gdb (diagnostic): gdb's own output goes to local/bench-v41/gdb.log, the engine's protocol is untouched.
cd "$(dirname "$0")/../.."
EXE="${DS41_EXE:-$PWD/build-hip-fused/ds41_generate}"
serve=0; for a in "$@"; do [ "$a" = "--serve" ] && serve=1; done
[ "$serve" = 1 ] || set -- --serve "$@"
exec gdb -q -batch -nx -ex "set logging file $PWD/local/bench-v41/gdb.log" -ex "set logging overwrite on" \
  -ex "set logging redirect on" -ex "set logging enabled on" -ex "set pagination off" -ex "handle SIGPIPE nostop noprint" \
  -ex run -ex "bt 40" -ex "info threads" -ex "thread apply all bt 12" --args "$EXE" "$@"
