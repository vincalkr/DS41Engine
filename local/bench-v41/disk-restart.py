#!/usr/bin/env python3
"""disk-restart.py [--out DIR] - a long conversation across an engine restart, with --conversation-disk: the default
configuration's flags (local/ds41-local.json: two cards, the conversation pool), ctx 65536; B = 25.5K tokens of the 64K
document, then B + 200 tokens, the engine ended and started again, then B + 400 tokens.  Prints each turn's resumed
positions and prompt seconds, and the engine's "conversation disk" lines (the save and restore times).  Results
appended to local/sweep-disk/summary.txt."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/v41/engine"))
import check_conv_reuse as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "local/sweep-disk"))
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
disk = out / "conv-disk"
shutil.rmtree(disk, ignore_errors=True)

cfg = json.load(open(ROOT / "local/ds41-local.json"))
args = list(cfg["args"])
i = args.index("--ctx"); args[i + 1] = "65536"
if "--dump-routes" in args:
    i = args.index("--dump-routes"); del args[i:i + 2]
args = [str(ROOT / x) if x.startswith(("models/", "local/")) else x for x in args]
args += ["--conversation-disk", str(disk)]
env = {k: str(v) for k, v in cfg["env"].items() if k not in ("DS41_EXE", "STRATA_TEXT_LOG")}
env["HIP_VISIBLE_DEVICES"] = "0,1"

doc = np.fromfile(ROOT / "local/prompts/huge64k.i32", dtype=np.int32).tolist()
B = doc[30000:55500]
steps = [B, B + doc[56000:56200], None, B + doc[56000:56400]]
lines = []
eng = C.Engine(ROOT / "build-hip-fused/ds41_generate", args, env, out, "disk0", cpu=False)
try:
    for k, ids in enumerate(steps):
        if ids is None:
            eng.close()
            eng = C.Engine(ROOT / "build-hip-fused/ds41_generate", args, env, out, "disk1", cpu=False)
            lines.append("engine restarted")
            print(lines[-1], flush=True)
            continue
        r = eng.gen(ids, 16)
        lines.append(f"turn {k}: prompt {len(ids)} resumed {r['resume']} prompt {r['prompt_ms'] / 1000:.2f} s "
                     f"request {r['sec']:.1f} s")
        print(lines[-1], flush=True)
finally:
    eng.close()
for name in ("disk0", "disk1"):
    for ln in (out / f"{name}.log").read_text(errors="replace").splitlines():
        if "conversation disk:" in ln:
            lines.append(f"{name}: " + ln.split("conversation disk: ", 1)[1].split("; records")[0])
size = sum(p.stat().st_size for p in disk.rglob("*") if p.is_file())
lines.append(f"on disk: {size / 1e6:.1f} MB")
print("\n".join(lines[-6:]))
with open(out / "summary.txt", "a") as f:
    f.write("\n".join(lines) + "\n")
