#!/usr/bin/env python3
"""score.py DIR - DIR/seq{0..7}.f32 (this engine's logits, 1024 x vocab each) against the FP8 1017 teacher's top-256
log-probs (eval/teacher_8x1024.npz: idx, lp for the 8 x 1023 next-token rows).  Prints ppl, top-1 agreement with the
teacher and KL(teacher || ours) over the teacher's top-256 (teacher renormalized there), the measures lxe and Lucebox
publish (lxe's quant on his engine: ppl 11.66, top-1 71.5 %, KL 0.462; teacher ppl 8.56)."""
import json, os, sys
import numpy as np

d = sys.argv[1]
e = os.path.dirname(os.path.abspath(__file__))
toks = np.load(f"{e}/heldout_8x1024.npy")
z = np.load(f"{e}/teacher_8x1024.npz")
tidx, tlp = z["idx"], z["lp"]
V = 129280
nll = kl = 0.0
top1 = n = 0
for s in range(8):
    lg = np.fromfile(f"{d}/seq{s}.f32", np.float32)
    lg = lg.reshape(-1, lg.size // len(toks[s]))[:, :V].astype(np.float64)
    lg -= lg.max(1, keepdims=True)
    ls = lg - np.log(np.exp(lg).sum(1, keepdims=True))          # our log-softmax, rows 0..1023
    for t in range(len(toks[s]) - 1):
        r = s * (len(toks[s]) - 1) + t
        nll -= ls[t, toks[s][t + 1]]
        ti, tl = tidx[r], tlp[r].astype(np.float64)
        p = np.exp(tl - tl.max()); p /= p.sum()
        kl += float((p * (np.log(p + 1e-30) - ls[t, ti])).sum())
        top1 += int(np.argmax(ls[t]) == ti[np.argmax(tl)])
        n += 1
res = {"tokens": n, "ppl": float(np.exp(nll / n)), "top1_agree": top1 / n, "kl_top256": kl / n}
print(json.dumps(res))
json.dump(res, open(f"{d}/score.json", "w"))
