#!/usr/bin/env python3
"""dump-main-diff.py A B [N_EMBD] - two STRATA_DUMP_MAIN files of the same forced decode ({int32 token, int32 pos,
3 x n_embd f32} per step: the HC-mean of the state after layers 36, 37, 38) compared step by step: the relative L2
difference ||a - b|| / ||a|| per layer row, mean / p90 / max over the steps, and the steps whose tokens differ (a
forced run feeds the same tokens, so none should)."""
import sys

import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5120
rec = np.dtype([("tok", "<i4"), ("pos", "<i4"), ("h", "<f4", (3, n))])
A, B = np.fromfile(a_path, rec), np.fromfile(b_path, rec)
m = min(len(A), len(B))
A, B = A[:m], B[:m]
same = (A["tok"] == B["tok"]) & (A["pos"] == B["pos"])
print(f"{m} steps, {int((~same).sum())} with a different token/pos")
for j, layer in enumerate((36, 37, 38)):
    ha, hb = A["h"][:, j, :].astype(np.float64), B["h"][:, j, :].astype(np.float64)
    rel = np.linalg.norm(ha - hb, axis=1) / np.maximum(np.linalg.norm(ha, axis=1), 1e-30)
    print(f"after layer {layer}: relative L2 difference mean {rel.mean():.2e}  p90 {np.percentile(rel, 90):.2e}  "
          f"max {rel.max():.2e}")
