#!/usr/bin/env python3
"""score.py DIR [--against DIR2] - DIR/seq{0..7}.f32 (this engine's logits, 1024 x vocab each).

Always printed: ppl against the held-out tokens (`heldout_8x1024.npy`) and this run's own top-1 next-token accuracy.
When the FP8 1017 teacher's top-256 log-probs are present (`teacher_8x1024.npz`: idx, lp for the 8 x 1023 next-token
rows) it also prints top-1 agreement with the teacher and KL(teacher || ours) over that top-256 (teacher renormalized
there), the measures lxe and Lucebox publish (lxe's quant on his engine: ppl 11.66, top-1 71.5 %, KL 0.462; teacher ppl
8.56).  That npz came off the 1017 FP8 on 2026-10-20 and is not on this box any more, so those two print only when it
is there; the numbers of the runs that had it are in those runs' score.json.

--against DIR2 compares the same rows with another run's logits instead: KL(other || ours) over the whole vocab and
top-1 agreement between the two - how one change (an Engram table form, a rank, a kernel) is read without a teacher.
Both runs' logits must be kept: run-eval.sh does that with KEEP_LOGITS=1.
"""
import json, os, sys
import numpy as np

args = sys.argv[1:]
d = args[0]
against = args[args.index("--against") + 1].rstrip("/") + "/" if "--against" in args else None
e = os.path.dirname(os.path.abspath(__file__))
toks = np.load(f"{e}/heldout_8x1024.npy")
V = 129280
tpath = f"{e}/teacher_8x1024.npz"
z = np.load(tpath) if os.path.exists(tpath) else None
tidx, tlp = (z["idx"], z["lp"]) if z is not None else (None, None)


def logsoftmax(f):
    lg = np.fromfile(f, np.float32)
    lg = lg.reshape(-1, lg.size // 1024)[:, :V].astype(np.float64)
    lg -= lg.max(1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(1, keepdims=True))


nll = kl = kl2 = 0.0
top1 = agree = agree2 = n = n2 = 0
for s in range(8):
    ls = logsoftmax(f"{d}/seq{s}.f32")
    ls2 = logsoftmax(f"{against}/seq{s}.f32") if against else None
    am, am2 = ls.argmax(1), (ls2.argmax(1) if ls2 is not None else None)
    for t in range(len(toks[s]) - 1):
        r = s * (len(toks[s]) - 1) + t
        nll -= ls[t, toks[s][t + 1]]
        top1 += int(am[t] == toks[s][t + 1])
        n += 1
        if z is not None:
            ti, tl = tidx[r], tlp[r].astype(np.float64)
            p = np.exp(tl - tl.max()); p /= p.sum()
            kl += float((p * (np.log(p + 1e-30) - ls[t, ti])).sum())
            agree += int(am[t] == ti[np.argmax(tl)])
        if ls2 is not None:
            q = np.exp(ls2[t])
            kl2 += float((q * (ls2[t] - ls[t])).sum())     # KL(other || ours)
            agree2 += int(am2[t] == am[t])
            n2 += 1
res = {"tokens": n, "ppl": float(np.exp(nll / n)), "top1_own": top1 / n,
       "teacher": os.path.basename(tpath) if z is not None else "absent"}
if z is not None:
    res["top1_agree"], res["kl_top256"] = agree / n, kl / n
if against:
    res["against"] = os.path.basename(against.rstrip("/"))
    res["kl_against"], res["top1_against"] = kl2 / n2, agree2 / n2
print(json.dumps(res))
json.dump(res, open(f"{d}/score.json", "w"))
