#!/usr/bin/env python3
"""ds4-engram-extract.py GGUF OUTDIR - the two Engram tables of a ds4 `deepseek41` GGUF (antirez/ds4: tensors
blk.{1,14}.engram_embd.weight, I8 [264, rows], encoding e4m3_e8m0_32_row264) as the safetensors files the engine's
--engram reads (OUTDIR/layers1/engram-L1.safetensors, OUTDIR/layers14/engram-L14.safetensors, each with its
model.safetensors.index.json).

A GGUF row is 256 E4M3 weight bytes then 8 E8M0 scale bytes (checked against the extracted tables of
the September checkpoint: rows 0-3 of the ds4 Q2 GGUF match weight+scale byte for byte); the safetensors file holds
`layers.N.engram.embed.scale` [rows, 8] F8_E8M0 first, then `layers.N.engram.embed.weight` [rows, 256] F8_E4M3.
Same bytes, regrouped; nothing is converted.  Needs numpy."""
import json, os, struct, sys
import numpy as np

def gguf_tensors(path):
    f = open(path, "rb")
    b = f.read(64 << 20)
    o = [4]
    def rd(fmt):
        v = struct.unpack_from("<" + fmt, b, o[0]); o[0] += struct.calcsize("<" + fmt); return v[0]
    def rs():
        n = rd("Q"); s = b[o[0]:o[0] + n].decode("utf-8", "replace"); o[0] += n; return s
    SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def skip(t):
        if t == 8: rs(); return
        if t == 9:
            et = rd("I"); n = rd("Q")
            for _ in range(n): skip(et)
            return
        rd(SZ[t])
    assert b[:4] == b"GGUF", "not a GGUF"
    rd("I"); nt = rd("Q"); nkv = rd("Q"); align = 32; enc = None
    for _ in range(nkv):
        k = rs(); t = rd("I")
        if k == "general.alignment": align = rd("I")
        elif k == "deepseek41.engram.encoding": enc = rs()
        else: skip(t)
    ts = {}
    for _ in range(nt):
        n = rs(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); off = rd("Q"); ts[n] = (dims, t, off)
    start = (o[0] + align - 1) // align * align
    return enc, start, ts

def extract(gguf, layer, out_dir):
    enc, start, ts = gguf_tensors(gguf)
    assert enc == "e4m3_e8m0_32_row264", f"unexpected engram encoding {enc}"
    dims, t, off = ts[f"blk.{layer}.engram_embd.weight"]
    assert dims[0] == 264 and t == 24, (dims, t)
    rows = dims[1]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"engram-L{layer}.safetensors")
    w_bytes, s_bytes = rows * 256, rows * 8
    hdr = json.dumps({f"layers.{layer}.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [rows, 8], "data_offsets": [0, s_bytes]},
                      f"layers.{layer}.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [rows, 256], "data_offsets": [s_bytes, s_bytes + w_bytes]}},
                     separators=(", ", ": ")).encode()
    hdr += b" " * ((8 - (8 + len(hdr)) % 8) % 8)
    base = 8 + len(hdr)
    src = open(gguf, "rb")
    with open(out, "wb") as fo, open(out, "r+b") as fw:
        fo.write(struct.pack("<Q", len(hdr)) + hdr)
        fo.truncate(base + s_bytes + w_bytes)
        fo.seek(base); fw.seek(base + s_bytes)
        step = 1 << 20   # rows per chunk (264 MiB)
        for r0 in range(0, rows, step):
            n = min(step, rows - r0)
            src.seek(start + off + r0 * 264)
            a = np.frombuffer(src.read(n * 264), dtype=np.uint8).reshape(n, 264)
            fw.write(a[:, :256].tobytes())
            fo.write(a[:, 256:].tobytes())
            if (r0 // step) % 32 == 0: print(f"  L{layer}: {r0 + n}/{rows} rows", flush=True)
    print(f"wrote {out} ({base + s_bytes + w_bytes} bytes)")
    return out

if __name__ == "__main__":
    gguf, out = sys.argv[1], sys.argv[2]
    paths = {l: extract(gguf, l, os.path.join(out, f"layers{l}")) for l in (1, 14)}
    # the merged index both directories carry (the engine reads the first --engram directory's)
    wm = {}
    for l in (1, 14):
        wm[f"layers.{l}.engram.embed.scale"] = f"engram-L{l}.safetensors"
        wm[f"layers.{l}.engram.embed.weight"] = f"engram-L{l}.safetensors"
    _, _, ts = gguf_tensors(gguf)
    total = sum(ts[f"blk.{l}.engram_embd.weight"][0][1] * 264 for l in (1, 14))
    for l in (1, 14):
        with open(os.path.join(out, f"layers{l}", "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": wm}, f, indent=1)
            f.write("\n")
