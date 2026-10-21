#!/usr/bin/env python3
"""serve-bench.py TAG [N] - decode speed through the running server (127.0.0.1:8150), the same requests every time so
two machine states can be compared without restarting it: N short answers (a fixed Italian prompt, 300 tokens) and N
answers after the ~22K-token document in local/bench-v41/long22k.txt (300 tokens; the first long request also pays the
prompt, the next ones reuse it).  Temperature 0, thinking off.  Each request's decode speed is the server's own
"serve: N tokens in X s" line in local/serve-engine.log.  Appends one line per request to
local/bench-v41/serve-bench.txt and prints the means."""
import json, statistics, sys, time, urllib.request

TAG = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
URL = "http://127.0.0.1:8150/v1/chat/completions"
LOG = "local/serve-engine.log"
DOC = open("local/bench-v41/long22k.txt").read()
SHORT = "Raccontami la trama dei Promessi Sposi, capitolo per capitolo."
LONG = DOC + "\n\nRiassumi il testo qui sopra in dieci punti dettagliati."


def ask(text):
    n0 = sum(1 for _ in open(LOG, errors="replace"))
    body = {"model": "x", "max_tokens": 300, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                       {"Content-Type": "application/json"}), timeout=3600) as r:
        json.load(r)
    wall = time.time() - t0
    time.sleep(1)
    lines = open(LOG, errors="replace").read().splitlines()[n0:]
    dec = [l for l in lines if l.startswith("serve: ") and " tokens in " in l and "prompt" not in l]
    pr = [l for l in lines if l.startswith("serve: prompt ")]
    tps = float(dec[-1].split("= ")[1].split()[0]) if dec else None
    return tps, wall, (pr[-1][:110] if pr else "")


res = {"short": [], "long": []}
with open("local/bench-v41/serve-bench.txt", "a") as out:
    for kind, text in (("short", SHORT), ("long", LONG)):
        for i in range(N + (1 if kind == "long" else 0)):
            tps, wall, pr = ask(text)
            warm = kind == "long" and i == 0
            line = f"{TAG} {kind}{' (prompt run)' if warm else ''} {i}: decode {tps} tok/s, wall {wall:.1f} s | {pr}"
            print(line, flush=True)
            out.write(line + "\n")
            if not warm and tps:
                res[kind].append(tps)
for k, v in res.items():
    if v:
        print(f"{TAG} {k}: mean {statistics.mean(v):.2f} tok/s over {len(v)} ({', '.join(f'{x:.2f}' for x in v)})")
