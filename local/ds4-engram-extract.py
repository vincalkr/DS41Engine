#!/usr/bin/env python3
"""ds4-engram-extract.py GGUF OUTDIR - the two Engram tables of a `deepseek41` GGUF that carries them (tensors
blk.{1,14}.engram_embd.weight, I8 [264, rows], encoding e4m3_e8m0_32_row264) as the safetensors files the engine's
--engram reads (OUTDIR/layers1/engram-L1.safetensors, OUTDIR/layers14/engram-L14.safetensors, each with its
model.safetensors.index.json).

A GGUF row is 256 E4M3 weight bytes then 8 E8M0 scale bytes (the checkpoint's own rows, byte for byte); the
safetensors file holds
`layers.N.engram.embed.scale` [rows, 8] F8_E8M0 first, then `layers.N.engram.embed.weight` [rows, 256] F8_E4M3.
Same bytes, regrouped; nothing is converted.  Needs numpy.

Env LAYERS ("1,14" by default) picks the layers to write, so one interrupted layer can be redone alone.  A table is
written as <name>.safetensors.part and renamed only when complete (the file's size is reserved up front, so a run
that dies - out of disk, Ctrl+C - must not leave something that looks finished), and a layer already on disk with the
exact expected size is skipped.  Free space is checked before writing: a layer needs rows x 264 bytes, and the pack
itself is only a source - once both tables are out, the GGUF can be deleted (or re-fetched by local/fetch-model.sh)."""
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

def st_header(layer, rows):
    """the safetensors header (scale [rows, 8] first, then weight [rows, 256]) and where the data starts"""
    w_bytes, s_bytes = rows * 256, rows * 8
    hdr = json.dumps({f"layers.{layer}.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [rows, 8], "data_offsets": [0, s_bytes]},
                      f"layers.{layer}.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [rows, 256], "data_offsets": [s_bytes, s_bytes + w_bytes]}},
                     separators=(", ", ": ")).encode()
    hdr += b" " * ((8 - (8 + len(hdr)) % 8) % 8)
    return hdr, 8 + len(hdr)

def expected_size(layer, rows):
    _, base = st_header(layer, rows)
    return base + rows * 264

def complete(path, layer, rows):
    """size alone is not enough: an interrupted run reserved the whole size with truncate(), so a table that died
    out of disk reads as zeros past the hole.  Sample the weight rows, like the engine does at open."""
    if not os.path.exists(path) or os.path.getsize(path) != expected_size(layer, rows):
        return False
    _, base = st_header(layer, rows)
    stride, zeros, sampled = max(1, rows // 256), 0, 0
    with open(path, "rb") as f:
        for r in range(0, rows, stride):
            if sampled >= 256: break
            f.seek(base + rows * 8 + r * 256)
            b = f.read(256)
            if len(b) != 256: return False
            sampled += 1
            zeros += not any(b)
    if sampled and zeros * 100 > sampled:
        print(f"{path}: {zeros} of {sampled} sampled rows are all zero - hole-filled, extracting it again")
        return False
    return True

def table_path(out, layer):
    return os.path.join(out, f"layers{layer}", f"engram-L{layer}.safetensors")

def extract(gguf, layer, out_dir, rows):
    enc, start, ts = gguf_tensors(gguf)
    assert enc == "e4m3_e8m0_32_row264", f"unexpected engram encoding {enc}"
    dims, t, off = ts[f"blk.{layer}.engram_embd.weight"]
    assert dims[0] == 264 and t == 24 and dims[1] == rows, (dims, t)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"engram-L{layer}.safetensors")
    hdr, base = st_header(layer, rows)
    src = open(gguf, "rb")
    tmp = out + ".part"                       # the size is reserved up front: an interrupted file must not look whole
    with open(tmp, "wb") as fo, open(tmp, "r+b") as fw:
        fo.write(struct.pack("<Q", len(hdr)) + hdr)
        fo.truncate(base + rows * 264)
        fo.seek(base); fw.seek(base + rows * 8)
        step = 1 << 20   # rows per chunk (264 MiB)
        for r0 in range(0, rows, step):
            n = min(step, rows - r0)
            src.seek(start + off + r0 * 264)
            a = np.frombuffer(src.read(n * 264), dtype=np.uint8).reshape(n, 264)
            fw.write(a[:, :256].tobytes())
            fo.write(a[:, 256:].tobytes())
            if (r0 // step) % 32 == 0: print(f"  L{layer}: {r0 + n}/{rows} rows", flush=True)
        fo.flush(); os.fsync(fo.fileno())
    os.replace(tmp, out)
    print(f"wrote {out} ({base + rows * 264} bytes)")
    return out

def write_index(out, layers, total):
    """the merged index both directories carry (the engine reads the first --engram directory's)"""
    wm = {}
    for l in layers:
        wm[f"layers.{l}.engram.embed.scale"] = f"engram-L{l}.safetensors"
        wm[f"layers.{l}.engram.embed.weight"] = f"engram-L{l}.safetensors"
    for l in layers:
        d = os.path.join(out, f"layers{l}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": wm}, f, indent=1)
            f.write("\n")

if __name__ == "__main__":
    gguf, out = sys.argv[1], sys.argv[2]
    _, _, tensors = gguf_tensors(gguf)
    all_layers = sorted(int(n.split(".")[1]) for n in tensors if n.endswith("engram_embd.weight"))
    rows = {l: tensors[f"blk.{l}.engram_embd.weight"][0][1] for l in all_layers}
    layers = [int(x) for x in os.environ.get("LAYERS", "").replace(" ", ",").split(",") if x.strip()] or all_layers
    missing = [l for l in layers if l not in all_layers]
    if missing: sys.exit(f"the pack has no blk.{missing}.engram_embd.weight (it has {all_layers})")
    todo = []
    for l in layers:                          # a layer already complete on disk costs nothing
        f = table_path(out, l)
        if complete(f, l, rows[l]):
            print(f"have L{l} ({os.path.getsize(f)} B)")
        else:
            todo.append(l)
    need = sum(rows[l] * 264 for l in todo)
    free = os.statvfs(out if os.path.isdir(out) else os.path.dirname(os.path.abspath(out))).f_bavail * os.statvfs(".").f_frsize
    if need and free < need + (2 << 30):
        sys.exit(f"ds4-engram-extract: not enough room: L{todo} need {need / 2**30:.0f} GiB, "
                 f"{free / 2**30:.0f} GiB free (the pack is only the source: with both tables written it can go)")
    for l in todo:
        extract(gguf, l, os.path.join(out, f"layers{l}"), rows[l])
    # the index always covers every Engram layer of the pack: the engine reads one directory's index and looks both up
    write_index(out, all_layers, sum(rows[l] * 264 for l in all_layers))
