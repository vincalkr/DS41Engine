#!/usr/bin/env python3
"""local/bench-v41/ab-dspark.py - DSpark (STRATA_SERVE_DSPARK) off / on through the server.

Each run starts the server (./run.sh --config <variant>) without the conversation disk, sends the dev-code prompt
(a code review over two engine sources, ~16.7K tokens), three follow-up turns of ~200 tokens each and then a prose
request, all greedy with 256 output tokens, and reads each request's decode tok/s from /metrics.  Variants
interleave, REPS times.  Output: local/sweep-dspark/summary.txt.

    .venv/bin/python local/bench-v41/ab-dspark.py [REPS=2] [VARIANTS=off,c06d4,c06d2]

Variants: off = no drafter; cXXdN = the drafter with STRATA_DSPARK_CONF=0.XX and STRATA_DSPARK_DEPTH=N.
DRAFTER (env) = the sidecar GGUF, default models/dspark-1017/V41-MTP-Vision-1017.Q2_K.gguf.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "local/sweep-dspark"
OUT.mkdir(exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
VARIANTS = (sys.argv[2] if len(sys.argv) > 2 else "off,c06d4,c06d2").split(",")
DRAFTER = os.environ.get("DRAFTER", str(ROOT / "models/dspark-1017/V41-MTP-Vision-1017.Q2_K.gguf"))
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
    "Now something different. Scrivi un racconto breve in italiano, in prosa, su un vecchio guardiano del faro che "
    "una notte d'inverno vede una luce sul mare. Almeno sei paragrafi.",
]


def config(v):
    c = json.loads((ROOT / "local/ds41-local.json").read_text())
    a = c["args"]
    for flag in ("--conversation-disk", "--conversation-disk-gib"):
        if flag in a:
            i = a.index(flag)
            del a[i:i + 2]
    if v != "off":
        c["env"]["STRATA_SERVE_DSPARK"] = DRAFTER
        c["env"]["STRATA_DSPARK_CONF"] = "0." + v[1:3]
        c["env"]["STRATA_DSPARK_DEPTH"] = v[4:]
    c["log"] = f"local/sweep-dspark/engine-{v}.log"
    p = OUT / f"config-{v}.json"
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
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return m.get("content") or m.get("reasoning_content") or ""


def run(v, rep):
    srv = subprocess.Popen([str(ROOT / "run.sh"), "--config", str(config(v))], cwd=ROOT,
                           stdout=open(OUT / f"serve-{v}-{rep}.out", "w"), stderr=subprocess.STDOUT)
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
    for v in VARIANTS:
        res = run(v, rep)
        line = f"{v} rep {rep}: " + "  ".join(f"read {p} -> {t} tok/s (hit {h})" for p, t, h in res)
        print(line, flush=True)
        rows.append(line)
        (OUT / "summary.txt").write_text("\n".join(rows) + "\n")
