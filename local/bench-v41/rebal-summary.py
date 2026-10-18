#!/usr/bin/env python3
"""rebal-summary.py - one table from local/bench-v41/rebal-logs/*.log (ab-rebal.sh): per run and turn, the prompt's
new tokens, prompt seconds, file-tier experts, decode tok/s and decode file-tier experts; then whether the answers of
each run match base-r1 (temperature 0)."""
import glob, json, os, re
P = re.compile(r"serve: prompt (\d+) tokens \((\d+) reused.*?in ([\d.]+) s .*?/ (\d+) file\)")
D = re.compile(r"serve: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s.*?, (\d+) file")
for f in sorted(glob.glob("local/bench-v41/rebal-logs/*.log")):
    tag = os.path.basename(f)[:-4]
    if tag.endswith(".serve"): continue
    txt = open(f).read()
    ps, ds = P.findall(txt), D.findall(txt)
    cells = []
    for (tot, reu, s, ff), (n, ds_, tps, df) in zip(ps, ds):
        cells.append(f"+{int(tot) - int(reu):5d} {float(s):5.1f}s f{int(ff):5d} | {float(tps):5.2f} f{int(df):4d}")
    print(f"{tag:8s} " + "  ||  ".join(cells))
ref = "local/bench-v41/agent-replay-base-r1.json"
if os.path.exists(ref):
    a = [m["content"] for m in json.load(open(ref)) if m["role"] == "assistant"]
    for f in sorted(glob.glob("local/bench-v41/agent-replay-*-r*.json")):
        b = [m["content"] for m in json.load(open(f)) if m["role"] == "assistant"]
        same = [x == y for x, y in zip(a, b)]
        pre = [len(os.path.commonprefix([x, y])) for x, y in zip(a, b)]
        print(f"{os.path.basename(f)}: identical to base-r1 per turn {same}, common prefix chars {pre}")
