#!/bin/sh
# DeepSeek-V4.1-Flash (GSQ) on ONE RX 7900 XTX: ./run.sh with local/ds41-local-1gpu.json (no second card).
# Extra arguments are passed to the server, for example:  ./run-1gpu.sh --port 8151
exec "$(cd "$(dirname "$0")" && pwd)/run.sh" --config "$(cd "$(dirname "$0")" && pwd)/local/ds41-local-1gpu.json" "$@"
