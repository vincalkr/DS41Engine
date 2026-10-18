#!/usr/bin/env python3
"""steady.py FILE.out... - steady-state decode from STRATA_DEC_PROF's cumulative lines (steps 128 -> 256): the first
tokens of a fresh process pay a warm-up (VRAM seed, graph captures) that a long-lived server pays once."""
import re, sys
for f in sys.argv[1:]:
    rows = {}
    for l in open(f, errors='replace'):
        m = re.search(r'begin ([\d.]+) predict ([\d.]+) attn_router ([\d.]+) tier ([\d.]+) finish ([\d.]+) spec ([\d.]+) \((\d+) steps\)', l)
        if m: rows[int(m.group(7))] = sum(float(x) for x in m.groups()[:6])
    if 128 in rows and 256 in rows:
        ms = (256 * rows[256] - 128 * rows[128]) / 128
        print(f"{f.split('/')[-1]:40s} steady {ms:6.1f} ms/token = {1000/ms:5.2f} tok/s   (cumulative 256: {1000/rows[256]:5.2f})")
