#!/usr/bin/env python3
"""ab-summary.py PREFIX [SUMMARY] - decode tok/s per variant and prompt from local/sweep-prof/summary.txt lines whose
tag starts with PREFIX (sweep.sh tags PREFIX<variant>-r<rep>): n, mean, standard deviation, median, min, max, and the
forced-token argmax differences (diff=) per run."""
import collections, re, statistics, sys

prefix = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else "local/sweep-prof/summary.txt"
runs = collections.defaultdict(list)
for line in open(path):
    f = line.split()
    if len(f) < 3 or not f[0].startswith(prefix):
        continue
    m = re.search(r"= ([0-9.]+) tok/s", line)
    d = re.search(r"diff=(\d+)", line)
    if not m:
        continue
    variant = re.sub(r"-r\d+$", "", f[0][len(prefix):])
    runs[(f[1], variant)].append((float(m.group(1)), int(d.group(1)) if d else -1))
for (prompt, variant), v in sorted(runs.items()):
    t = [x for x, _ in v]
    sd = statistics.stdev(t) if len(t) > 1 else 0.0
    print(f"{prompt:10s} {variant:10s} n={len(t)}  mean {statistics.mean(t):6.2f}  sd {sd:5.2f}  median {statistics.median(t):6.2f}"
          f"  min {min(t):6.2f}  max {max(t):6.2f} tok/s  diff {[d for _, d in v]}")
