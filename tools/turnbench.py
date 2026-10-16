#!/usr/bin/env python3
"""turnbench.py - the agent-loop prefill benchmark: what a SMALL appended turn costs on a served engine.

An agent re-sends its whole conversation every step; with prefix reuse the engine only prefills the new tail, so
the number that matters is the cost of a +N-token turn on top of a long context, not one-shot prefill.

  turnbench.py <url> [--base 4000] [--deltas 64,182,512,1024,2048] [--log /tmp/strata-ds41.log] [--json out.json]

Turn 0 sends a ~base-token user message (code); every next turn appends the reply and a new user message of ~delta
tokens. Each request asks for 1 token with thinking off, so the wall time is the prompt. Prints per turn: new
tokens, reused tokens, prefill seconds, new-token rate - and, when --log names the engine's log, its own
"prompt phases" line for that turn (dense / experts / cache-open split).
"""
import glob
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def corpus():
    """Deterministic, varied text (repo source) so tokens are realistic and every turn's filler is distinct."""
    files = sorted(glob.glob(os.path.join(ROOT, "serve", "*.py"))) + sorted(glob.glob(os.path.join(ROOT, "tools", "gauntlet", "*.py")))
    return "".join(open(f, errors="replace").read() for f in files)


def main():
    url = sys.argv[1]
    base, deltas = int(arg("--base", 4000)), [int(x) for x in arg("--deltas", "64,182,512,1024,2048").split(",")]
    log, out = arg("--log", None), arg("--json", None)
    text, cur = corpus(), 0

    def take(ntok):
        nonlocal cur
        n = int(ntok * 3.6)   # ~3.6 chars a token on this code
        s = text[cur:cur + n]
        cur += n
        return s

    msgs = [{"role": "system", "content": "You are a terse assistant. Reply with one word."},
            {"role": "user", "content": "Read this code:\n" + take(base)}]
    rows = []
    for i, d in enumerate([0] + deltas):
        if d:
            msgs.append({"role": "user", "content": "More code:\n" + take(d)})
        logpos = os.path.getsize(log) if log and os.path.exists(log) else 0
        body = {"model": "x", "messages": msgs, "max_tokens": 1, "temperature": 0, "reasoning_effort": "none", "stream": False}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        j = json.load(urllib.request.urlopen(req, timeout=3600))
        wall = time.time() - t0
        u, tm = j.get("usage", {}), j.get("timings", {})
        reused = tm.get("cache_n", (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        new = u.get("prompt_tokens", 0) - reused
        pms = tm.get("prompt_ms", wall * 1000)
        phases = ""
        if log and os.path.exists(log):
            time.sleep(0.3)
            with open(log, errors="replace") as f:
                f.seek(logpos)
                phases = " | ".join(l.strip()[7:] for l in f if l.startswith("serve: prompt phases"))
        row = {"turn": i, "delta_req": d, "prompt": u.get("prompt_tokens", 0), "reused": reused, "new": new,
               "prompt_s": round(pms / 1000, 2), "new_tok_s": round(new / max(pms / 1000, 1e-9), 1), "wall_s": round(wall, 2),
               "phases": phases}
        rows.append(row)
        print(f"turn {i}: +{new:5d} new ({reused:6d} reused)  prefill {row['prompt_s']:6.2f} s = {row['new_tok_s']:7.1f} tok/s"
              f"{'   ' + phases if phases else ''}", flush=True)
        msgs.append({"role": "assistant", "content": (j["choices"][0]["message"].get("content") or "ok").strip() or "ok"})
    if out:
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
