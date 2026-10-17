#!/usr/bin/env python3
"""Merge ds41_generate --dump-routes stats directories (sum experts[e].n) into one v41_rank.py input.

  python3 local/calib-q2/merge_stats.py --out local/calib-q2/stats-merged DIR [DIR ...]
"""
import argparse
import json
import os

L, NE = 40, 384


def load(d):
    c = {}
    for l in range(L):
        p = os.path.join(d, f"L{l:02d}.json")
        ex = json.load(open(p))["experts"]
        for e in range(NE):
            c[(l, e)] = int(ex[str(e)]["n"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    tot = {k: 0 for k in load(a.dirs[0])}
    for d in a.dirs:
        c = load(d)
        for k, n in c.items():
            tot[k] += n
    os.makedirs(a.out, exist_ok=True)
    for l in range(L):
        with open(os.path.join(a.out, f"L{l:02d}.json"), "w") as f:
            f.write('{"layer": %d, "top_k": 6, "experts": {' % l)
            f.write(", ".join('"%d": {"n": %d}' % (e, tot[(l, e)]) for e in range(NE)))
            f.write("}}\n")
    n = sum(tot.values())
    print(f"merged {len(a.dirs)} dirs -> {a.out}: {n} selections = {n / (L * 6):.0f} tokens")


if __name__ == "__main__":
    main()
