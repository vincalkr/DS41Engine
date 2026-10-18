#!/usr/bin/env python3
"""gguf-expert-map.py GGUF OUT.json - the per-expert tier map v41_rank.py --map reads, straight from a GSQ-format GGUF.

A GSQ layer whose experts mix types stores I32 maps `blk.L.ffn_exps_gu_type` / `blk.L.ffn_exps_down_type` [384]
(ggml type per expert: 11 = Q3_K, 42 = Q2_0, 143 = PTQ1_0, tools/v41/v41_gguf.py); a layer with one type for a role
has a plain stacked `blk.L.ffn_{gate,down}_exps.weight` of that type.  Writes {"map": {"L": {"gu": [384], "dn": [384]}}}
with v41_rank.py's names (q3_k / q2_0 / ptq1), the format local/gsq-alloc-map.json has."""
import json, struct, sys

NAME = {11: "q3_k", 42: "q2_0", 143: "ptq1"}
NE = 384


def read_gguf(path):
    f = open(path, "rb")
    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
    def rstr():
        (n,) = rd("Q")
        return f.read(n).decode("utf-8", "replace")
    scal = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            (at,), (n,) = rd("I"), rd("Q")
            return [rval(at) for _ in range(n)]
        return rd(scal[t])[0]
    magic, ver = f.read(4), rd("I")[0]
    assert magic == b"GGUF", magic
    nt, nkv = rd("Q")[0], rd("Q")[0]
    kv = {}
    for _ in range(nkv):
        k = rstr(); (t,) = rd("I"); kv[k] = rval(t)
    tens = {}
    for _ in range(nt):
        name = rstr(); (nd,) = rd("I"); dims = rd("Q" * nd); (typ,) = rd("I"); (off,) = rd("Q")
        tens[name] = (typ, dims, off)
    align = int(kv.get("general.alignment", 32))
    data0 = (f.tell() + align - 1) // align * align
    return f, kv, tens, data0


def main():
    src, out = sys.argv[1], sys.argv[2]
    f, kv, tens, data0 = read_gguf(src)
    nl = max(int(n.split(".")[1]) for n in tens if n.startswith("blk.")) + 1
    res = {}
    for L in range(nl):
        roles = {}
        for role, mapn, plain in (("gu", "ffn_exps_gu_type", "ffn_gate_exps.weight"),
                                  ("dn", "ffn_exps_down_type", "ffn_down_exps.weight")):
            m = tens.get(f"blk.{L}.{mapn}")
            if m:
                f.seek(data0 + m[2])
                ids = list(struct.unpack(f"<{NE}i", f.read(4 * NE)))
            elif f"blk.{L}.{plain}" in tens:
                ids = [tens[f"blk.{L}.{plain}"][0]] * NE
            else:
                ids = None
            if ids is not None: roles[role] = [NAME.get(i, str(i)) for i in ids]
        if len(roles) == 2: res[str(L)] = roles
    json.dump({"map": res, "source": src}, open(out, "w"))
    cnt = {}
    for L in res.values():
        for t in L["gu"]: cnt[t] = cnt.get(t, 0) + 1
    print(f"{len(res)} routed layers; gate/up types over all experts: {cnt}")


if __name__ == "__main__":
    main()
