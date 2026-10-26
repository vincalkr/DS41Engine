#!/usr/bin/env python3
"""v41_prune_list.py RANK FRAC OUT - the coldest FRAC of the experts in a v41_rank.py ranking (int32 (layer, expert)
pairs, hottest first) as "layer expert" lines: ds41_generate --prune OUT [--prune-penalty X] (soft pruning, GLM's recipe:
the listed experts take no VRAM/arena space and get a selection bias -X; 0 = never routed)."""
import sys

import numpy as np

rank, frac, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
r = np.fromfile(rank, np.int32).reshape(-1, 2)
n = int(round(len(r) * frac))
cold = r[len(r) - n:]
with open(out, "w") as f:
    for l, e in cold:
        f.write(f"{int(l)} {int(e)}\n")
per_layer = np.bincount(cold[:, 0], minlength=int(r[:, 0].max()) + 1)
print(f"{n} of {len(r)} experts ({frac:.0%}) -> {out}; per layer min {per_layer.min()} max {per_layer.max()}")
