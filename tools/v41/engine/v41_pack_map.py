#!/usr/bin/env python3
"""v41_pack_map.py - the per-expert quant types of a GGUF pack, as ds41_generate --rank's byte map.

    python3 tools/v41/engine/v41_pack_map.py PACK-00001-of-NNNNN.gguf > map.json

`v41_rank.py` ranks (layer, expert) by routed tokens per BYTE of the blob an expert occupies, so a pack that mixes
formats needs to say which format each expert's gate/up and down are in.  Two pack shapes exist: one stacked tensor per
role for a whole layer (uniform types - the GSQ-RCO pack), or a tensor per type plus the `blk.N.ffn_exps_{gu,down}_type`
I32 maps (the mixed packs).  Both are read here from the headers, plus the 1536-byte I32 maps when present: no tensor
payload is touched.  Output is the form v41_rank.py accepts:

    {"map": {"<layer>": {"gu": [<NE ggml type names>], "dn": [<NE type names>]}}}

A split pack's other shards are resolved from the `-of-NNNNN` name; pass them explicitly if they were renamed.
"""
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ds41_pack_meta import Header                                            # noqa: E402

I32 = 26   # GGML_TYPE_I32


def ggml_type_names():
    """ggml's type enum as id -> lowercase name, read from the vendored header."""
    h = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                     "third_party", "llama.cpp", "ggml", "include", "ggml.h")
    body = open(h).read().split("enum ggml_type {", 1)[1].split("};", 1)[0]
    return {int(m.group(2)): m.group(1)[len("GGML_TYPE_"):].lower()
            for m in re.finditer(r"(GGML_TYPE_[A-Z0-9_]+)\s*=\s*(\d+)", body)}


def shard_paths(first):
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    n = int(m.group(2))
    return [re.sub(r"-\d{5}-of-\d{5}\.gguf$", "-%05d-of-%05d.gguf" % (i, n), first) for i in range(1, n + 1)]


def read_i32(path, off, n):
    with open(path, "rb") as f:
        return list(struct.unpack("<%di" % n, os.pread(f.fileno(), 4 * n, off)))


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    TN = ggml_type_names()
    files, T = [], {}
    paths = sys.argv[1:]
    if len(paths) == 1:
        paths = shard_paths(paths[0])          # resolve the -of-NNNNN family of a split pack
    for p in paths:
        h = Header(p)
        files.append(p)
        for n, (ne, ty, off) in h.tensors.items():
            T[n] = (p, ne, ty, off)
    layers = sorted({int(m.group(1)) for n in T for m in [re.match(r"blk\.(\d+)\.", n)] if m})
    out, warn = {}, []
    for L in layers:
        b = f"blk.{L}."
        r = T.get(b + "ffn_gate_inp.weight")
        if not r or len(r[1]) != 2:
            continue                       # not a routed layer
        NE = r[1][1]
        roles = {role: T.get(b + f"ffn_{role}_exps.weight") for role in ("gate", "up", "down")}
        maps = {k: T.get(b + f"ffn_exps_{k}_type") for k in ("gu", "down")}
        if all(roles.values()):            # uniform: one stacked tensor per role, its type is every expert's
            gu, up, dn = (TN[roles[k][2]] for k in ("gate", "up", "down"))
            if gu != up:
                warn.append(f"layer {L}: gate {gu} != up {up} in a uniform pack, gate taken")
            out[str(L)] = {"gu": [gu] * NE, "dn": [TN[roles['down'][2]]] * NE}
        elif maps["gu"] and maps["down"]:  # per-type: the I32 maps hold each expert's type id
            for key, dst in (("gu", "gu"), ("down", "dn")):
                p, ne, ty, off = maps[key]
                if ty != I32 or ne != [NE]:
                    warn.append(f"{b}ffn_exps_{key}_type: type {ty} shape {ne}, layer {L} skipped")
                    break
                out.setdefault(str(L), {})[dst] = [TN.get(v, "unknown") for v in read_i32(p, off, NE)]
            if "gu" not in out.get(str(L), {}) or "dn" not in out.get(str(L), {}):
                out.pop(str(L), None)
        else:
            warn.append(f"layer {L}: neither uniform role tensors nor the I32 type maps")
    for w in warn:
        print("warning:", w, file=sys.stderr)
    print(f"read {len(files)} file(s), {len(T)} tensors, {len(out)} routed layers", file=sys.stderr)
    json.dump({"map": out}, sys.stdout, separators=(",", ":"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
