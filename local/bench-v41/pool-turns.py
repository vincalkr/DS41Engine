#!/usr/bin/env python3
"""pool-turns.py [--pool MIB] [--out DIR] - two long conversations taking turns on one engine (the default two-card
configuration's flags, ctx 65536): A = the 17K prompt, B = 25.5K tokens of the 64K document from position 30000;
six turns A B A B A B, each adding ~200 tokens of the document, 16 tokens decoded.  Prints each turn's resumed
positions and prompt seconds; --pool N adds --conversation-cache-mib N (0 = the single park, the default engine).
Results appended to local/sweep-pool/summary.txt."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/v41/engine"))
import check_conv_reuse as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "local/sweep-pool"))
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(ROOT / "local/ds41-local.json"))
args = list(cfg["args"])
i = args.index("--ctx"); args[i + 1] = "65536"
if "--dump-routes" in args:
    i = args.index("--dump-routes"); del args[i:i + 2]
args = [str(ROOT / x) if x.startswith(("models/", "local/")) else x for x in args]
if a.pool > 0:
    args += ["--conversation-cache-mib", str(a.pool)]
env = {k: str(v) for k, v in cfg["env"].items() if k not in ("DS41_EXE", "STRATA_TEXT_LOG")}
env["HIP_VISIBLE_DEVICES"] = "0,1"

doc = np.fromfile(ROOT / "local/prompts/huge64k.i32", dtype=np.int32).tolist()
A = np.fromfile(ROOT / "local/prompts/huge17k.i32", dtype=np.int32).tolist()
B = doc[30000:55500]
tail = {"A": doc[56000:], "B": doc[58000:]}
conv = {"A": A, "B": B}
k = {"A": 0, "B": 0}

eng = C.Engine(ROOT / "build-hip-fused/ds41_generate", args, env, out, f"pool{a.pool}", cpu=False)
lines = []
try:
    for turn, c in enumerate(["A", "B", "A", "B", "A", "B"]):
        if turn >= 2:
            conv[c] = conv[c] + tail[c][k[c]:k[c] + 200]
            k[c] += 200
        r = eng.gen(conv[c], 16)
        s = (f"pool={a.pool:<5d} turn {turn} {c}: prompt {len(conv[c])} resumed {r['resume']} prompt {r['prompt_ms'] / 1000:.2f} s "
             f"request {r['sec']:.1f} s")
        print(s, flush=True)
        lines.append(s)
finally:
    eng.close()
with open(out / "summary.txt", "a") as f:
    f.write("\n".join(lines) + "\n")
