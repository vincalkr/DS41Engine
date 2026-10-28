#!/usr/bin/env python3
"""ds41_vl_bias.py - the image routing bias as a small GGUF for DS41_VL_BIAS.

The GSQ-RCO pack carries the text router bias (`blk.N.exp_probs_b.bias`, identical to the checkpoint's
`layers.N.ffn.gate.bias`) but not the image one.  ISTA-DASLab publish it next to the pack as `exp_probs_b_vl.f32`
(40 x 384 F32, the checkpoint's `layers.N.ffn.gate.bias_vl`, 61,440 bytes).  This writes it as 40 tensors
`blk.N.exp_probs_b_vl.bias` [384] F32 in one GGUF, which the engine opens after the pack's shards when DS41_VL_BIAS
names it.

    python3 tools/v41/ds41_vl_bias.py --bias exp_probs_b_vl.f32 --out local/rco-iq2xxs-meta/vl-bias.gguf
"""
import argparse
import struct
import sys

ALIGN = 32


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", required=True, help="exp_probs_b_vl.f32 (n_layer x n_expert F32)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    a = ap.parse_args()
    raw = open(a.bias, "rb").read()
    row = a.experts * 4
    if len(raw) != a.layers * row:
        sys.exit(f"{a.bias}: {len(raw)} bytes, expected {a.layers} x {a.experts} F32 = {a.layers * row}")
    kv = gstr("general.architecture") + struct.pack("<I", 8) + gstr("deepseek41")
    infos = b""
    for l in range(a.layers):
        infos += gstr(f"blk.{l}.exp_probs_b_vl.bias") + struct.pack("<I", 1) + struct.pack("<Q", a.experts)
        infos += struct.pack("<I", 0) + struct.pack("<Q", l * row)   # type 0 = F32, offset in the data section
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", a.layers, 1) + kv + infos
    head += b"\0" * (-len(head) % ALIGN)
    with open(a.out, "wb") as f:
        f.write(head + raw)
    print(f"{a.out}: {a.layers} tensors blk.N.exp_probs_b_vl.bias [{a.experts}] F32")


if __name__ == "__main__":
    main()
