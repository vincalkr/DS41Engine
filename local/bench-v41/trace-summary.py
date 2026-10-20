#!/usr/bin/env python3
"""trace-summary.py SHORT_DIR LONG_DIR TOKENS - per-token GPU work of decode from two rocprofv3 traces (csv) of the same
run with SHORT and SHORT+TOKENS decode tokens: the difference of the two is TOKENS tokens of decode, with the load and
the prompt cancelled out.  Prints, per agent, kernels and busy time per token, the kernels by total time, and the
memory copies by direction."""
import collections, csv, glob, os, re, sys

def load(d):
    ks, cs = [], []
    for f in glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True):
        ks += list(csv.DictReader(open(f)))
    for f in glob.glob(os.path.join(d, "**", "*memory_copy_trace.csv"), recursive=True):
        cs += list(csv.DictReader(open(f)))
    return ks, cs

def short(name):
    n = re.sub(r"\(.*", "", name)                      # drop the argument list
    n = re.sub(r"<.*", "", n) if len(n) > 70 else n    # long templates: the base name
    return n.strip()[:70]

def stats(ks, cs):
    agent = collections.Counter(); busy = collections.Counter()
    byk = collections.defaultdict(lambda: [0, 0])
    for k in ks:
        a = k.get("Agent_Id", "?"); d = int(k["End_Timestamp"]) - int(k["Start_Timestamp"])
        agent[a] += 1; busy[a] += d
        e = byk[(a, short(k["Kernel_Name"]))]; e[0] += 1; e[1] += d
    cp = collections.defaultdict(lambda: [0, 0, 0])
    for c in cs:
        key = c.get("Direction") or f'{c.get("Source_Agent_Id")}->{c.get("Destination_Agent_Id")}'
        e = cp[key]; e[0] += 1; e[1] += int(c["End_Timestamp"]) - int(c["Start_Timestamp"])
        e[2] += int(c.get("Bytes") or c.get("Size") or 0)
    return agent, busy, byk, cp

a_dir, b_dir, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
A, B = stats(*load(a_dir)), stats(*load(b_dir))
print(f"per decode token ({ntok} tokens = {b_dir} - {a_dir}):")
for ag in sorted(set(A[0]) | set(B[0])):
    n = (B[0][ag] - A[0][ag]) / ntok; t = (B[1][ag] - A[1][ag]) / ntok / 1e6
    print(f"  agent {ag}: {n:.0f} kernels, {t:.2f} ms busy")
rows = []
for key in set(A[2]) | set(B[2]):
    n = (B[2][key][0] - A[2][key][0]) / ntok; t = (B[2][key][1] - A[2][key][1]) / ntok / 1e6
    if n > 0.5 or t > 0.02: rows.append((t, n, key))
print("  kernels by time per token (agent, name: count, ms, us each):")
for t, n, (ag, name) in sorted(rows, reverse=True)[:40]:
    print(f"    {ag:>3} {name:70s} {n:7.1f} {t:7.3f} {1000 * t / max(n, 1e-9):8.2f}")
print("  copies per token (direction: count, ms, MB):")
for key in sorted(set(A[3]) | set(B[3])):
    n = (B[3][key][0] - A[3][key][0]) / ntok; t = (B[3][key][1] - A[3][key][1]) / ntok / 1e6
    mb = (B[3][key][2] - A[3][key][2]) / ntok / 1e6
    print(f"    {key:40s} {n:7.1f} {t:7.3f} {mb:8.2f}")
