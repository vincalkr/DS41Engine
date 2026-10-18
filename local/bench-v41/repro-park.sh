#!/usr/bin/env bash
# Reproduce the 2026-10-18 card fault: a 3-turn conversation, then a NEW long prompt (thinking on) after the park.
# Uses whatever server runs on 8150.  Prints the reuse lines and whether the engine survived.
cd "$(dirname "$0")/../.."
python3 local/bench-v41/multiturn.py local/bench-v41/long.txt || true
python3 - <<'PY' || true
import json, urllib.request
doc = open("local/bench-v41/long.txt").read()
msgs = [{"role": "system", "content": "Sei un assistente tecnico."}, {"role": "user", "content": doc + "\n\nQuali sono i 3 limiti principali descritti?"}]
body = {"model": "x", "stream": False, "max_tokens": 200, "temperature": 0.0, "messages": msgs, "chat_template_kwargs": {"enable_thinking": True}}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
print("4th request OK, prompt_tokens", r["usage"]["prompt_tokens"])
PY
grep -E "serve: reuse|launch failure|parked" local/serve-engine.log | tail -6 | cut -c1-160
pgrep -x ds41_generate >/dev/null && echo "ENGINE ALIVE" || echo "ENGINE DEAD"
