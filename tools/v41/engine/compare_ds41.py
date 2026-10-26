#!/usr/bin/env python3
"""compare_ds41.py GOLDEN_DIR ENGINE_DIR [--mode decode|prefill] - per layer: cosine / max rel error of the block
output and pre-mix, router top-6 set agreement and weight error; logits cosine + top-1 agreement per position."""
import argparse
import os

import numpy as np


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("engine")
    ap.add_argument("--mode", default="decode")
    ap.add_argument("--min-cos", type=float, default=0.9999)
    a = ap.parse_args()
    g = np.load(os.path.join(a.golden, "golden.npz"))
    d = os.path.join(a.engine, a.mode)
    S = len(g["tokens"])
    L = 0
    worst = 1.0
    bad = 0
    while f"x{L}" in g.files and os.path.exists(f"{d}/x{L}.f32"):
        gx = g[f"x{L}"]
        ex = np.fromfile(f"{d}/x{L}.f32", np.float32).reshape(gx.shape)
        gp, ep = g[f"pre{L}"], np.fromfile(f"{d}/pre{L}.f32", np.float32).reshape(g[f"pre{L}"].shape)
        gi, ei = g[f"ids{L}"], np.fromfile(f"{d}/ids{L}.i32", np.int32).reshape(g[f"ids{L}"].shape)
        gw, ew = g[f"wts{L}"], np.fromfile(f"{d}/wts{L}.f32", np.float32).reshape(g[f"wts{L}"].shape)
        same = sum(set(gi[t]) == set(ei[t]) for t in range(S))
        wd = max(abs(np.sort(gw[t]) - np.sort(ew[t])).max() for t in range(S) if set(gi[t]) == set(ei[t])) if same else float("nan")
        cx = min(cos(gx[t], ex[t]) for t in range(S))
        rel = float(np.abs(gx - ex).max() / np.abs(gx).max())
        cp = cos(gp, ep)
        worst = min(worst, cx)
        flag = "" if cx >= a.min_cos and same == S else "  <-- check"
        bad += bool(flag)
        print(f"L{L:2d}  x cos(min over pos) {cx:.7f}  max|d|/max {rel:.2e}  pre cos {cp:.7f}  route sets {same}/{S}  "
              f"w maxdiff {wd:.2e}{flag}")
        L += 1
    if os.path.exists(f"{d}/logits.f32") and "logits" in g.files:
        gl = g["logits"]
        el = np.fromfile(f"{d}/logits.f32", np.float32).reshape(gl.shape)
        cs = [cos(gl[t], el[t]) for t in range(S)]
        top = sum(int(gl[t].argmax() == el[t].argmax()) for t in range(S))
        print(f"logits cos min {min(cs):.7f} mean {np.mean(cs):.7f}  top-1 {top}/{S}")
    print("PASS" if not bad else f"{bad} layer(s) below cos {a.min_cos} or with routing differences")


if __name__ == "__main__":
    main()
