#!/usr/bin/env python3
"""local/bench-v41/ab-vram-conv.py - STRATA_VRAM_CONV off / on through the server, the way a coding session uses it.

Each run starts the server (./run.sh --config <variant>) without the conversation disk, sends the dev-code prompt
(a code review over two engine sources, ~16.7K tokens) and then three follow-up turns of ~200 tokens each (enough to
run as a batch, which re-opens the VRAM cache), all greedy with 128 output tokens, and reads each request's decode
tok/s from /metrics.  Variants interleave, REPS times.  Output: local/sweep-vram-conv/summary.txt.

    .venv/bin/python local/bench-v41/ab-vram-conv.py [REPS=3] [SHARES=0,500]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "local/sweep-vram-conv"
OUT.mkdir(exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SHARES = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,500").split(",")]
URL = "http://127.0.0.1:8150"

rd = lambda f: (ROOT / f).read_text()
PROMPT = ("You are working on a C++/HIP inference engine. Here are two source files.\n\n"
          "=== tools/v41/engine/ds41_conv_disk.hpp ===\n" + rd("tools/v41/engine/ds41_conv_disk.hpp") +
          "\n=== tools/v41/engine/ds41_conv_pool.cpp ===\n" + rd("tools/v41/engine/ds41_conv_pool.cpp") +
          "\n\nReview this code: list the concrete bugs or risks you find (file, function, why), then write a patch in "
          "unified diff format that fixes the most important one, with a short explanation.")
pool = rd("tools/v41/engine/ds41_conv_pool.cpp")
FOLLOW = [
    "Now look at this part again and tell me whether the locking is correct:\n\n" + pool[2000:2800],
    "Rewrite this function so that it handles a short write and returns an error, keep the style:\n\n" + pool[4000:4800],
    "Write a unit test plan for this code (cases, inputs, expected results):\n\n" + pool[6000:6800],
]


def config(share):
    c = json.loads((ROOT / "local/ds41-local.json").read_text())
    a = c["args"]
    for flag in ("--conversation-disk", "--conversation-disk-gib"):
        if flag in a:
            i = a.index(flag)
            del a[i:i + 2]
    c["env"]["STRATA_VRAM_CONV"] = str(share)
    c["log"] = f"local/sweep-vram-conv/engine-{share}.log"
    p = OUT / f"config-{share}.json"
    p.write_text(json.dumps(c, indent=2))
    return p


def get(path):
    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.loads(r.read())


def chat(msgs):
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": 128, "temperature": 0,
                       "reasoning_effort": "low", "stream": False}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return m.get("content") or m.get("reasoning_content") or ""


def run(share, rep):
    srv = subprocess.Popen([str(ROOT / "run.sh"), "--config", str(config(share))], cwd=ROOT,
                           stdout=open(OUT / f"serve-{share}-{rep}.out", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                if get("/metrics")["live"]["state"] == "idle":
                    break
            except Exception:
                pass
            time.sleep(2)
        msgs, res = [{"role": "user", "content": PROMPT}], []
        for turn in range(1 + len(FOLLOW)):
            msgs.append({"role": "assistant", "content": chat(msgs)})
            r = get("/metrics")["requests"][0]
            res.append((r["prompt_read"], r["decode_tok_s"], r["hit_rate"]))
            if turn < len(FOLLOW):
                msgs.append({"role": "user", "content": FOLLOW[turn]})
        return res
    finally:
        subprocess.run([str(ROOT / "stop.sh")], cwd=ROOT)
        srv.wait(timeout=180)


rows = []
for rep in range(REPS):
    for share in SHARES:
        res = run(share, rep)
        line = f"STRATA_VRAM_CONV={share} rep {rep}: " + "  ".join(f"read {p} -> {t} tok/s (hit {h})" for p, t, h in res)
        print(line, flush=True)
        rows.append(line)
        (OUT / "summary.txt").write_text("\n".join(rows) + "\n")
