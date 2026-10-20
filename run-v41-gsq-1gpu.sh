#!/bin/sh
# DeepSeek-V4.1-Flash (GSQ) on ONE RX 7900 XTX: ./run-v41-gsq.sh with local/ds41-local-1gpu.json (no second card).
# Extra arguments are passed to the server, for example:  ./run-v41-gsq-1gpu.sh --port 8151
exec "$(cd "$(dirname "$0")" && pwd)/run-v41-gsq.sh" --config "$(cd "$(dirname "$0")" && pwd)/local/ds41-local-1gpu.json" "$@"
