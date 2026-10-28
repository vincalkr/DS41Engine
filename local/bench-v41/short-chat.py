#!/usr/bin/env python3
"""local/bench-v41/short-chat.py - decode tok/s on short chats (the default configuration, routing exact).

One server start (./run.sh, the default two-card configuration, no conversation disk), then a short chat of three
turns (a ~60-token question, two follow-ups; the context stays under ~1,500 tokens) and a fresh ~1,200-token
prompt, greedy, 256 output tokens each, REPS times; each request's decode tok/s from /metrics.  For comparing with
engines that report speed on short chats.  Output: local/sweep-short/summary.txt.

    .venv/bin/python local/bench-v41/short-chat.py [REPS=3]
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "local/sweep-short"
OUT.mkdir(exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
URL = "http://127.0.0.1:8150"
CHAT = ["Explain in a few paragraphs how a hash map handles collisions, with a small Python example.",
        "Now compare open addressing and chaining: when is each one faster?",
        "Scrivi la stessa spiegazione in italiano, in modo semplice, per uno studente."]
LONG = ("Summarize this source file and point out anything that looks wrong:\n\n" +
        (ROOT / "tools/v41/engine/ds41_conv_pool.cpp").read_text()[:4500])


def config():
    c = json.loads((ROOT / "local/ds41-local.json").read_text())
    a = c["args"]
    for flag in ("--conversation-disk", "--conversation-disk-gib"):
        if flag in a:
            i = a.index(flag)
            del a[i:i + 2]
    c["log"] = "local/sweep-short/engine.log"
    p = OUT / "config.json"
    p.write_text(json.dumps(c, indent=2))
    return p


def get(path):
    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.loads(r.read())


def chat(msgs):
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": 256, "temperature": 0,
                       "reasoning_effort": "low", "stream": False}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        m = json.loads(r.read())["choices"][0]["message"]
    r = get("/metrics")["requests"][0]
    return m.get("content") or m.get("reasoning_content") or "", (r["prompt_tokens"] if "prompt_tokens" in r else r["prompt_read"], r["decode_tok_s"], r["hit_rate"])


srv = subprocess.Popen([str(ROOT / "run.sh"), "--config", str(config())], cwd=ROOT,
                       stdout=open(OUT / "serve.out", "w"), stderr=subprocess.STDOUT)
rows = []
try:
    for _ in range(600):
        try:
            if get("/metrics")["live"]["state"] == "idle":
                break
        except Exception:
            pass
        time.sleep(2)
    for rep in range(REPS):
        msgs, res = [], []
        for q in CHAT:
            msgs.append({"role": "user", "content": q})
            txt, r = chat(msgs)
            msgs.append({"role": "assistant", "content": txt})
            res.append(r)
        res.append(chat([{"role": "user", "content": LONG}])[1])
        line = f"rep {rep}: " + "  ".join(f"prompt {p} -> {t} tok/s (hit {h})" for p, t, h in res)
        print(line, flush=True)
        rows.append(line)
        (OUT / "summary.txt").write_text("\n".join(rows) + "\n")
finally:
    subprocess.run([str(ROOT / "stop.sh")], cwd=ROOT)
    srv.wait(timeout=180)
