#!/usr/bin/env python3
"""make_vision_gguf.py - the DeepSeek-V4.1-Flash-1017 vision encoder as a GGUF, straight from the checkpoint.

The 1017 checkpoint (deepseek-ai/DeepSeek-V4.1-Flash-1017) keeps the vision tower in BF16: shard 1 holds exactly the
263 ViT + aligner tensors (0.90 GiB), shard 2 the three span delimiters (image_start / image_newline / image_end,
BF16 [5120] each) next to embed.weight.  This copies those 266 tensors byte for byte into a GGUF with the same tensor
names and metadata keys as antirez's DeepSeek-V4.1-Flash-Vision.gguf, which is the SEPTEMBER checkpoint's tower: the
1017 tower was retrained (0 of 266 tensors equal, per-tensor cosine 0.995-0.99999; on DeepSeek's example images the
September tower's image rows differ from the 1017 ones by cosine 0.89-0.91 on average, 0.38 at worst - measured
2026-10-21, docs/ds4/V41_VISION.md).  So the 1017 model needs this file, not the September one.

Only HTTP range reads: the two safetensors headers, shard 1's data (0.90 GiB) and 30 KB of shard 2.  --shard1 FILE
uses a shard 1 already on disk instead of downloading it.  Standard library only.

    python3 tools/v41/vision/make_vision_gguf.py [--out models/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf]
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"   # the revision the Q2 GGUF names (general.source.revision)
SHARD1 = "model-00001-of-00048.safetensors"
SHARD2 = "model-00002-of-00048.safetensors"
SHARD1_SHA256 = "13bd121075f0b68cb7e8e8021a1011d42a81a83f557a2602fd9c07d52fddb0ec"   # HF LFS id at REVISION
DELIMITERS = ("image_start", "image_newline", "image_end")
GGML_BF16 = 30
ALIGN = 32

# the vision config (inference/config.json at REVISION; the Q2 GGUF's deepseek41.config carries the same values)
KV = [
    ("general.architecture", "str", "deepseek4-vision"),
    ("general.name", "str", "DeepSeek V4.1 Flash 1017 Vision Encoder"),
    ("general.alignment", "u32", ALIGN),
    ("general.source.url", "str", f"https://huggingface.co/{REPO}"),
    ("general.source.revision", "str", REVISION),
    ("deepseek4-vision.checkpoint_variant", "str", "v4.1-flash-1017"),
    ("deepseek4-vision.rope.freq_base", "f32", 10000.0),
    ("deepseek4-vision.attention.layer_norm_rms_epsilon", "f32", 1e-6),
    ("deepseek4-vision.block_count", "u32", 32),
    ("deepseek4-vision.embedding_length", "u32", 1024),
    ("deepseek4-vision.feed_forward_length", "u32", 2816),
    ("deepseek4-vision.attention.head_count", "u32", 16),
    ("deepseek4-vision.projection_length", "u32", 5120),
    ("deepseek4-vision.patch_size", "u32", 14),
    ("deepseek4-vision.downsample_ratio", "u32", 3),
    ("deepseek4-vision.image.max_tokens", "u32", 1024),
    ("deepseek4-vision.image.min_pixels", "u32", 295936),
    ("deepseek4-vision.image.max_width_height_ratio", "u32", 0),
    ("deepseek4-vision.image.token_id", "u32", 129264),
    ("deepseek4-vision.language.block_count", "u32", 40),
    ("deepseek4-vision.language.expert_count", "u32", 384),
]


def url(name):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def token_headers():
    tok = os.environ.get("HF_TOKEN") or ""
    p = os.path.expanduser("~/.cache/huggingface/token")
    if not tok and os.path.exists(p):
        tok = open(p).read().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def http_range(u, a, b):
    """Bytes [a, b) of a URL."""
    req = urllib.request.Request(u, headers={**token_headers(), "Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != b - a:
        raise IOError(f"{u}: got {len(data)} bytes of range {a}-{b - 1}")
    return data


class Source:
    """A safetensors shard, local or remote: its header and byte ranges of its data."""

    def __init__(self, name, path=None):
        self.name, self.path = name, path
        n = struct.unpack("<Q", self.read_abs(0, 8))[0]
        self.header = json.loads(self.read_abs(8, 8 + n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def read_abs(self, a, b):
        if self.path:
            with open(self.path, "rb") as f:
                f.seek(a)
                d = f.read(b - a)
            if len(d) != b - a:
                raise IOError(f"{self.path}: short read")
            return d
        return http_range(url(self.name), a, b)

    def stream(self, a, b, chunk=64 << 20):
        """Yield the data bytes [a, b) (relative to the data start) in chunks."""
        pos = a
        while pos < b:
            e = min(b, pos + chunk)
            yield self.read_abs(self.base + pos, self.base + e)
            pos = e


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf_kv(key, ty, val):
    code = {"u32": 4, "f32": 6, "str": 8}[ty]
    body = struct.pack("<I", val) if ty == "u32" else struct.pack("<f", val) if ty == "f32" else gguf_str(val)
    return gguf_str(key) + struct.pack("<I", code) + body


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--out", default=os.path.join(root, "models", "vision-1017", "DeepSeek-V4.1-Flash-1017-Vision.gguf"))
    ap.add_argument("--shard1", help=f"{SHARD1} already on disk (checked against its sha256)")
    ap.add_argument("--no-sha", action="store_true", help="skip the sha256 of a local --shard1")
    a = ap.parse_args()

    if a.shard1 and not a.no_sha:
        h = hashlib.sha256()
        with open(a.shard1, "rb") as f:
            for blk in iter(lambda: f.read(64 << 20), b""):
                h.update(blk)
        if h.hexdigest() != SHARD1_SHA256:
            sys.exit(f"{a.shard1}: sha256 {h.hexdigest()} is not {REPO}@{REVISION[:8]}'s {SHARD1} ({SHARD1_SHA256})")
    s1 = Source(SHARD1, a.shard1)
    s2 = Source(SHARD2)

    tensors = []   # (name, shape (checkpoint order), source, data start, data end)
    for name in sorted(s1.header):
        v = s1.header[name]
        if not name.startswith(("vision.", "aligner.")) or v["dtype"] != "BF16":
            sys.exit(f"{SHARD1}: unexpected tensor {name} ({v['dtype']}): not the 1017 layout this script knows")
        tensors.append((name, v["shape"], s1, *v["data_offsets"]))
    for name in DELIMITERS:
        v = s2.header[name]
        if v["dtype"] != "BF16" or v["shape"] != [5120]:
            sys.exit(f"{SHARD2}: {name} is {v['dtype']} {v['shape']}, expected BF16 [5120]")
        tensors.append((name, v["shape"], s2, *v["data_offsets"]))
    if len(tensors) != 266:
        sys.exit(f"expected 266 vision tensors, found {len(tensors)}")

    # header: KV, tensor infos (ggml order: shape reversed), offsets aligned to ALIGN
    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(KV)))
    for k, ty, v in KV:
        hdr += gguf_kv(k, ty, v)
    off = 0
    offsets = []
    for name, shape, _, a0, a1 in tensors:
        if a1 - a0 != 2 * math.prod(shape):
            sys.exit(f"{name}: {a1 - a0} bytes for shape {shape}")
        hdr += gguf_str(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape[::-1])
        hdr += struct.pack("<IQ", GGML_BF16, off)
        offsets.append(off)
        off += (a1 - a0 + ALIGN - 1) // ALIGN * ALIGN
    hdr += b"\0" * (-len(hdr) % ALIGN)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    total = sum(t[4] - t[3] for t in tensors)
    done = 0
    with open(tmp, "wb") as f:
        f.write(hdr)
        data0 = f.tell()
        # shard 1's tensors are contiguous in its data region: read them in big ranges, in file order
        for (name, shape, src, a0, a1), o in sorted(zip(tensors, offsets), key=lambda t: (t[0][2] is s2, t[0][3])):
            f.seek(data0 + o)
            for blk in src.stream(a0, a1):
                f.write(blk)
                if done * 10 // total != (done + len(blk)) * 10 // total:
                    print(f"  {(done + len(blk)) / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
                done += len(blk)
        f.seek(0, 2)
        f.write(b"\0" * (-(f.tell() - data0) % ALIGN))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(tensors)} BF16 tensors, {total / 2**30:.3f} GiB, {REPO}@{REVISION[:8]}")


if __name__ == "__main__":
    main()
