#!/usr/bin/env python3
"""local/bench_ds41.py - a repeatable A/B bench for this machine's ds41_generate.

  local/bench_ds41.py --prompt local/prompts/long.i32 --n 64 --reps 3 \
      --base "--backend cuda --experts gpu --arena-gib 60 --ctx 262144 --kv-q8 --engram D1:D2" \
      --config "arena90=--arena-gib 90" --config "pcie50=--pcie 0.50"

Why: one-shot runs on this box vary run to run (GPU clocks, page cache, whatever else is running), so a single A/B is
worth nothing.  This runs the configurations ROUND ROBIN (A,B,A,B,...) for --reps rounds, prints every run and the
MEDIAN per configuration for prompt tok/s, decode tok/s and the per-pass dense/tier split, which is what the
configuration decisions are made on.

Each run is a fresh process (the arena seeds the same way every time), and the model's load time is not counted:
only the engine's own "prompt:"/"decode:" totals are.
"""
import argparse
import re
import statistics
import subprocess
import sys
import time

ROOT = None  # set in main
EXE_DEFAULT = "build-hip-pre/ds41_generate"
MODEL_DEFAULT = "models/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
RANK_DEFAULT = "local/rank-q2.bin"

RE_PROMPT = re.compile(r"^prompt: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s")
RE_DECODE = re.compile(r"^decode: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s")
RE_PASS = re.compile(r"pass_v split \(all (\d+) passes.*dense ([\d.]+) ms, tier ([\d.]+) ms")
RE_PSPLIT = re.compile(r"prompt split: dense ([\d.]+) s, experts ([\d.]+) s \(file wait ([\d.]+) s\)")


def run_once(exe, args, log_path, env):
    t0 = time.time()
    p = subprocess.run([str(exe)] + args, capture_output=True, text=True, timeout=3600, env=env)
    wall = time.time() - t0
    out = p.stdout + p.stderr
    open(log_path, "w").write(out)
    r = {"rc": p.returncode, "wall_s": wall, "prompt": None, "decode": None, "dense_ms": None, "tier_ms": None,
         "pdense": None, "pexperts": None, "pfilewait": None}
    for line in out.splitlines():
        m = RE_PROMPT.match(line)
        if m:
            r["prompt"] = (int(m.group(1)), float(m.group(3)))
        m = RE_DECODE.match(line)
        if m:
            r["decode"] = (int(m.group(1)), float(m.group(3)))
        m = RE_PASS.search(line)
        if m:
            r["dense_ms"], r["tier_ms"] = float(m.group(2)), float(m.group(3))
        m = RE_PSPLIT.search(line)
        if m:
            r["pdense"], r["pexperts"], r["pfilewait"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return r


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--exe", default=EXE_DEFAULT)
    ap.add_argument("--base", default="")
    ap.add_argument("--config", action="append", default=[], help="label=extra args (repeatable)")
    ap.add_argument("--out", default="local/bench")
    ap.add_argument("--env", action="append", default=[], help="NAME=value for the engine (repeatable)")
    a = ap.parse_args()

    global ROOT
    ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
    exe = ROOT / a.exe
    base = f"-m {ROOT / MODEL_DEFAULT} --ids-file {ROOT / a.prompt} -n {a.n} " \
           f"--rank {ROOT / RANK_DEFAULT} --threads 24 --stop 1 " + " ".join(
               t for t in a.base.split() if not t.startswith("ENV:"))

    env = dict(__import__("os").environ)
    env["STRATA_NO_LARGEPAGES"] = "1"
    for kv in a.env + [t for t in a.base.split() if t.startswith("ENV:")]:
        k, v = kv.replace("ENV:", "").split("=", 1)
        env[k] = v
    configs = [("base", "")] + [(c.split("=", 1)[0], c.split("=", 1)[1]) for c in a.config]
    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    results = {label: [] for label, _ in configs}
    for rep in range(a.reps):
        for label, extra in configs:
            toks = [t for t in extra.split() if not t.startswith("ENV:")]
            cenv = dict(env)
            for t in extra.split():
                if t.startswith("ENV:"):
                    k, v = t[4:].split("=", 1)
                    cenv[k] = v
            args = base.split() + toks
            log = outdir / f"{label}-r{rep}.log"
            sys.stderr.write(f"[{rep + 1}/{a.reps}] {label}: {' '.join(toks) or '(base)'}\n")
            sys.stderr.flush()
            results[label].append(run_once(exe, args, log, cenv))

    print(f"\n{'config':>16} | {'prompt tok/s':>21} | {'decode tok/s':>21} | {'p-split dense/exp/file s':>25} | {'pass dense/tier ms':>19}")
    print("-" * 118)
    for label, _ in configs:
        rs = results[label]
        pr = " ".join(f"{r['prompt'][1]:.1f}" if r["prompt"] else "ERR" for r in rs)
        de = " ".join(f"{r['decode'][1]:.1f}" if r["decode"] else "ERR" for r in rs)
        ps = "%.1f/%.1f/%.1f" % (med([r["pdense"] for r in rs]), med([r["pexperts"] for r in rs]),
                                 med([r["pfilewait"] for r in rs])) if any(r["pdense"] for r in rs) else "-"
        dm, tm = med([r["dense_ms"] for r in rs]), med([r["tier_ms"] for r in rs])
        pass_s = "%.1f/%.1f" % (dm, tm) if dm == dm else "-"
        print(f"{label:>16} | {med([r['prompt'][1] for r in rs if r['prompt']]):>7.1f} [{pr}] "
              f"| {med([r['decode'][1] for r in rs if r['decode']]):>7.1f} [{de}] | {ps:>25} | {pass_s:>19}")
    print("\n(medians first, then every run's value in brackets; logs in %s)" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
