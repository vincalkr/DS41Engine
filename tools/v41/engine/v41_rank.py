#!/usr/bin/env python3
"""v41_rank.py - rank every (layer, expert) for the engine's VRAM cache / RAM arena seed (ds41_generate --rank).

Order: calibration routing count per BYTE (a VRAM slot or arena slot costs the expert's blob, Q3_K experts are ~1.5x a
Q2_0 one), hottest first.  Counts = stats/LNN.json experts[e].n (tokens routed to e over the 256K-token agentic-first
calibration corpus); bytes from the tier map (alloc_full.json) or, without one, all Q2_0.

    python3 tools/v41/engine/v41_rank.py --stats /media/Crucial1TB/v41-calib/stats [--map alloc_full.json] --out rank.bin
writes int32 (layer, expert) pairs, and prints the cumulative routed share covered by the first N GiB.
"""
import argparse
import json
import os

import numpy as np

D, FF, NE = 5120, 2304, 384
# bytes per value of the expert formats (block bytes / block values): the mixed packs are ranked per BYTE of
# blob, so an IQ1_M expert (56 B a 256) and a Q2_K one (84 B) must not be priced the same.
BPW = {"q2_0": 18 / 64, "q3_k": 110 / 256, "ptq1": 28 / 128,
       "iq1_m": 56 / 256, "iq2_xxs": 66 / 256, "iq2_xs": 74 / 256, "q2_k": 84 / 256,
       "iq2_s": 98 / 256, "iq3_xxs": 110 / 256, "iq3_s": 104 / 256, "iq4_nl": 18 / 32, "iq4_xs": 136 / 256}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="/media/Crucial1TB/v41-calib/stats")
    ap.add_argument("--map")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    amap = json.load(open(a.map))["map"] if a.map else None
    rows = []
    total = 0
    for L in range(a.layers):
        p = os.path.join(a.stats, f"L{L:02d}.json")
        if not os.path.exists(p):
            print(f"warning: no stats for layer {L} (ranked last, by index)")
            for e in range(NE):
                rows.append((0.0, L, e, 1))
            continue
        ex = json.load(open(p))["experts"]
        for e in range(NE):
            n = float(ex[str(e)]["n"]) if str(e) in ex else 0.0
            if amap and str(L) in amap:
                m = amap[str(L)]
                if isinstance(m, dict) and str(e) in m:          # v41_alloc --per-proj: {e: {gu, dn}}
                    gu, dn = m[str(e)]["gu"], m[str(e)]["dn"]
                elif isinstance(m, dict):
                    gu, dn = m["gu"][e], m["dn"][e]
                else:
                    gu = dn = m[e]
            else:
                gu = dn = "q2_0"
            b = 2 * D * FF * BPW[gu] + D * FF * BPW[dn]
            rows.append((n / b, L, e, b))
            total += n
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    np.array([(r[1], r[2]) for r in rows], np.int32).tofile(a.out)
    cum_n, cum_b, marks = 0.0, 0.0, [8, 16, 32, 48, 64, 80, 100, 120, 150]
    for r in rows:
        cum_n += r[0] * r[3]
        cum_b += r[3]
        while marks and cum_b >= marks[0] * 2**30:
            print(f"first {marks[0]:3d} GiB of experts cover {100 * cum_n / max(total, 1):5.1f}% of routed tokens")
            marks.pop(0)
    print(f"wrote {len(rows)} pairs to {a.out}; all experts {cum_b / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
