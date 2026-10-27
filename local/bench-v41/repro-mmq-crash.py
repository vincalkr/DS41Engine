#!/usr/bin/env python3
"""repro-mmq-crash.py [LONG NEW] - the 2026-10-27 crash's shape: a ~123K-token conversation, then a turn that adds ~3.9K new tokens as one
prompt batch.  The chat configuration's flags (local/ds41-local.json, ctx 262144), no conversation disk."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/v41/engine"))
import check_conv_reuse as C  # noqa: E402
out = ROOT / "local/sweep-crash"
out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(ROOT / "local/ds41-local.json"))
args = list(cfg["args"])
for flag in ("--dump-routes", "--conversation-disk", "--conversation-disk-gib"):
    if flag in args:
        i = args.index(flag); del args[i:i + 2]
args = [str(ROOT / x) if x.startswith(("models/", "local/")) else x for x in args]
env = {k: str(v) for k, v in cfg["env"].items() if k not in ("DS41_EXE", "STRATA_TEXT_LOG")}
env["HIP_VISIBLE_DEVICES"] = "0,1"
import os
args += os.environ.get("REPRO_ARGS", "").split()
for kv in os.environ.get("REPRO_ENV", "").split():
    k, v = kv.split("=", 1); env[k] = v
env["DS41_META"] = str(ROOT / env["DS41_META"])
doc = np.fromfile(ROOT / "local/prompts/huge64k.i32", dtype=np.int32).tolist()
long_ = (doc * 3)[:int(sys.argv[1]) if len(sys.argv) > 1 else 123416]
new = (doc[1000:] * 2)[:int(sys.argv[2]) if len(sys.argv) > 2 else 3918]
eng = C.Engine(ROOT / "build-hip-fused/ds41_generate", args, env, out, "crash", cpu=False)
try:
    for k, ids in enumerate([long_, long_ + new, long_ + new + doc[5000:9000]]):
        r = eng.gen(ids, 16)
        print(f"turn {k}: prompt {len(ids)} resumed {r['resume']} prompt {r['prompt_ms'] / 1000:.1f} s", flush=True)
finally:
    eng.close()
