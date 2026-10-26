#!/usr/bin/env python3
"""ds41_pack_meta.py - an antirez-style metadata overlay for a GSQ-RCO IQ2_XXS GGUF pack.

The ISTA-DASLab / RCO packs carry llama.cpp/HF metadata (`embedding_length`, `block_count`,
`attention.head_count_kv`, ...), not the flat `deepseek41.*` keys ds41e reads, and their Engram arrays live as I32/I64
tensors instead of KV entries. This writes ONE tiny GGUF with exactly the keys and full integer arrays that
`tools/v41/engine/ds41_geometry.hpp` expects: geometry scalars and the config JSON come from the reference antirez Q2
file (same checkpoint architecture), compress ratios/sources are cross-checked against which tensors the pack holds,
and the Engram token map/primes/multipliers are read out of the pack's own small global tensors.  The table row counts
come from the separate Engram IQ4_NL GGUF and must tile exactly under those primes.

    python3 tools/v41/ds41_pack_meta.py --pack SHARD1 --ref REF_Q2.gguf --engram ENGRAM.gguf \
        --out models/rco-iq2xxs-meta/meta.gguf

Then run the engine with DS41_META=<that file>; no tensor data is touched or copied.
"""
import argparse
import array
import json
import mmap
import os
import struct
import sys

# ggml metadata type ids and their on-disk scalar formats
SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h",
    4: "<I", 5: "<i", 6: "<f", 12: "<d",
    7: "<B", 10: "<Q", 11: "<q",
}
META_TYPE = {"u32": 4,
             "i32": 5,
             "f32": 6,
             "bool": 7,
             "u64": 10,
             "i64": 11}


class Header:
    """KV section and tensor infos only; the tokenizer arrays are skipped by length."""

    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), min(os.fstat(f.fileno()).st_size, 128 << 20), access=mmap.ACCESS_READ)
        pos = 8

        def rd(fmt):
            nonlocal pos
            sz = struct.calcsize(fmt)
            v = struct.unpack_from(fmt, mm, pos)[0]
            pos += sz
            return v

        def rstr():
            nonlocal pos
            n = rd("<Q")
            s = bytes(mm[pos:pos + n]).decode("utf8", "replace")
            pos += n
            return s

        nt, nkv = rd("<Q"), rd("<Q")
        kvv = {}
        for _ in range(nkv):
            k = rstr()
            t = rd("<I")
            if t == 9:
                at = rd("<I")
                an = rd("<Q")
                keep = (k.startswith("deepseek41.") and an <= 64) or k in (
                    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id")
                vals = None
                if keep:
                    esz = struct.calcsize(SCALAR_FMT[at])
                    vals = [int(struct.unpack_from(SCALAR_FMT[at], mm, pos + i * esz)[0]) for i in range(an)]
                    pos += esz * an
                elif at == 8:
                    for _ in range(an):
                        n = struct.unpack_from("<Q", mm, pos)[0]
                        pos += 8 + n
                else:
                    pos += struct.calcsize(SCALAR_FMT[at]) * an
                kvv[k] = ("array", at, vals)
            elif t == 7:
                kvv[k] = ("bool", None, bool(rd("<B")))
            elif t == 6:
                kvv[k] = ("num", None, float(rd("<f")))
            elif t == 10:
                kvv[k] = ("num", None, int(rd("<Q")))
            elif t == 11:
                kvv[k] = ("num", None, int(rd("<q")))
            elif t == 12:
                kvv[k] = ("num", None, float(rd("<d")))
            elif t <= 5:
                fmt = SCALAR_FMT[t]
                kvv[k] = ("num", None, int(struct.unpack_from(fmt, mm, pos)[0]))
                pos += struct.calcsize(fmt)
            else:
                n = rd("<Q")
                s = bytes(mm[pos:pos + n]).decode("utf8", "replace")
                pos += n
                kvv[k] = ("str", None, s)

        tensors = []
        for _ in range(nt):
            name = rstr()
            nd = rd("<I")
            ne = [int(rd("<Q")) for _ in range(nd)]
            ty = rd("<I")
            off = rd("<Q")
            tensors.append((name, ne, ty, off))
        self.data_start = (pos + 31) // 32 * 32
        self.tensors = {n: (ne, ty, self.data_start + off) for n, ne, ty, off in tensors}
        self.kv_wanted = kvv
        f.close()

    def get(self, key):
        if key not in self.kv_wanted:
            raise KeyError(key)
        return self.kv_wanted[key][2]


class KeyRef:
    """The reference architecture keys read from JSON instead of from a GGUF: the same `deepseek41.*` scalars, arrays
    and tokenizer ids a reference pack's header carries.  They describe the checkpoint, not the pack, so the overlay can
    be rebuilt on a fresh clone without any 366 GB reference download (`--ref-keys`, generated once by --dump-ref-keys
    from the reference header)."""

    def __init__(self, path):
        self.kv_wanted = {k: ("json", None, v) for k, v in json.load(open(path)).items()}

    def get(self, key):
        if key not in self.kv_wanted:
            raise KeyError(key)
        return self.kv_wanted[key][2]


def payload(path, at, count, width):
    with open(path, "rb") as f:
        raw = os.pread(f.fileno(), count * width, at)
    a = array.array("i" if width == 4 else "q")
    a.frombytes(raw)
    return list(a)


# ---- GGUF writer -----------------------------------------------------------
ALIGN = 256


def gstr(s):
    b = s.encode("utf8") if isinstance(s, str) else bytes(s)
    return struct.pack("<Q", len(b)) + b


META_TYPE = {"u32": 4, "i32": 5, "f32": 6,
             "bool": 7, "u64": 10, "i64": 11}

def kv(key, ty, val):
    if ty == "array":
        et, vals = val
        body = struct.pack("<I", META_TYPE[et]) + struct.pack("<Q", len(vals))
        fmt = {"u32": "<I", "i32": "<i", "u64": "<Q", "i64": "<q"}[et]
        for v in vals:
            body += struct.pack(fmt, int(v))
        code = 9
    elif ty == "str":
        body, code = gstr(val), 8
    elif ty == "bool":
        body, code = struct.pack("<B", bool(val)), 7
    else:
        body = struct.pack({"u32": "<I", "i32": "<i", "f32": "<f",
                            "u64": "<Q", "i64": "<q"}[ty], val)
        code = META_TYPE[ty]
    return gstr(key) + struct.pack("<I", code) + body


SCALARS = [
    ("hidden_size", "u32"), ("num_hidden_layers", "u32"), ("num_attention_heads", "u32"),
    ("head_dim", "u32"), ("qk_rope_head_dim", "u32"), ("q_lora_rank", "u32"),
    ("o_lora_rank", "u32"), ("o_groups", "u32"), ("sliding_window", "u32"),
    ("n_routed_experts", "u32"), ("num_experts_per_tok", "u32"), ("moe_intermediate_size", "u32"),
    ("n_shared_experts", "u32"), ("hc_mult", "u32"), ("hc_sinkhorn_iters", "u32"),
    ("index_n_heads", "u32"), ("index_head_dim", "u32"), ("index_topk", "u32"),
    ("candidate_source_layer_id", "i32"), ("candidate_topk_blocks", "u32"),
    ("candidate_block_size", "u32"), ("vocab_size", "u32"), ("max_position_embeddings", "u32"),
    ("routed_scaling_factor", "f32"), ("hc_eps", "f32"), ("rms_norm_eps", "f32"),
    ("rope_theta", "f32"), ("compress_rope_theta", "f32"),
    ("rope_scaling.factor", "f32"), ("rope_scaling.beta_fast", "f32"),
    ("rope_scaling.beta_slow", "f32"), ("rope_scaling.original_max_position_embeddings", "f32"),
    ("swiglu_limit", "f32"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="", help="first shard of the RCO pack")
    ap.add_argument("--ref-keys", default="", help="JSON with the reference deepseek41.* keys (see --dump-ref-keys)")
    ap.add_argument("--dump-ref-keys", default="", help="write --ref's keys to this JSON and stop")
    ap.add_argument("--ref", default="models/ds4-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf",
                    help="antirez Q2 GGUF carrying deepseek41.* geometry (same checkpoint architecture)")
    ap.add_argument("--engram", default="", help="the separate Engram IQ4_NL GGUF (table row counts); without it the rows come from the reference keys under deepseek41.engram.rows")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.dump_ref_keys:
        h = Header(a.ref or a.pack)
        with open(a.dump_ref_keys, "w") as f:
            json.dump({k: v for k, (_, _, v) in sorted(h.kv_wanted.items())}, f, indent=1)
            f.write("\n")
        print(f"{a.dump_ref_keys}: {len(h.kv_wanted)} reference keys")
        return 0
    if not (a.pack and a.out):
        ap.error("--pack and --out are needed (--engram is optional with --ref-keys); "
                 "--dump-ref-keys OUT.json --ref REF.gguf writes the reference keys")
    if not a.ref_keys and not os.path.exists(a.ref):
        sys.exit(f"{a.ref} is not there: use --ref-keys <geometry-keys.json> "
                 f"(tools/v41/ds41_pack_meta.py --dump-ref-keys REF.gguf keys.json builds one from any reference pack)")
    ref = KeyRef(a.ref_keys) if a.ref_keys else Header(a.ref)
    d = os.path.dirname(os.path.abspath(a.pack))
    base = os.path.basename(a.pack)
    marker = "-00001-of-"
    if marker not in base:
        sys.exit(f"{a.pack}: expected a -0000N-of-0000M split name")
    stem, tail = base.split(marker)
    nsh = int(tail[:-5])
    pack_shards = [os.path.join(d, f"{stem}-{i:05d}-of-{nsh:05d}.gguf") for i in range(1, nsh + 1)]
    headers = [Header(p) for p in pack_shards]
    present = {}
    for h, p in zip(headers, pack_shards):
        for n in h.tensors:
            present[n] = (h, p)

    def has(l, s):
        return f"blk.{l}.{s}" in present

    nl = int(ref.get("deepseek41.num_hidden_layers"))
    ratios_ref = list(ref.get("deepseek41.compress_ratios"))
    kv_ref = list(ref.get("deepseek41.kv_source_layer_ids"))
    idx_ref = list(ref.get("deepseek41.index_source_layer_ids"))
    if len(ratios_ref) != nl:
        sys.exit(f"reference compress_ratios has {len(ratios_ref)} entries for {nl} layers")

    ratio = []
    cur = 0
    for l in range(nl):
        # A compressor source starts a new KV-cache regime; layers after it share that compressed cache.
        if has(l, "attn_compressor_kv.weight"):
            cur = 2 if has(l, "attn_compressor_gate.weight") else 1
        ratio.append(cur)
    kv_src = [l for l in range(nl) if has(l, "attn_compressor_kv.weight")]
    idx_src = [l for l in range(nl) if has(l, "indexer.proj.weight") and has(l, "indexer.attn_q_b.weight")]
    ok_shape = (ratio == ratios_ref and kv_src == kv_ref and idx_src == idx_ref)
    if not ok_shape:
        print("WARNING: pack tensors disagree with the reference geometry; using tensor-derived sets", file=sys.stderr)

    def small(name):
        h, p = present[name]
        ne, ty, at = h.tensors[name]
        n = 1
        for x in ne:
            n *= x
        width = 4 if int(ty) == 26 else 8   # I32 token map vs I64 primes/multipliers/offsets
        return payload(p, at, n, width)

    tm = small("engram_token_map.weight")
    pr = small("engram_primes.weight")
    mu = small("engram_multipliers.weight")

    engr_rows = []
    if a.engram:
        efile = Header(a.engram)
        for l in (1, 14):
            name = f"blk.{l}.engram_embd.weight"
            if name not in efile.tensors:
                sys.exit(f"{a.engram}: no {name}")
            engr_rows.append(int(efile.tensors[name][0][-1]))
    else:
        # the FP8 tables' row counts (the checkpoint's own tables, which the pack's IQ4_NL file mirrors)
        engr_rows = [int(v) for v in ref.get("deepseek41.engram.rows")]
        print(f"engram table rows from the reference keys: {engr_rows}")

    cfg_str = ref.get("deepseek41.config")
    cj = json.loads(cfg_str)

    def jget(o, key):
        if isinstance(o, dict):
            if key in o:
                return o[key]
            for v in o.values():
                r = jget(v, key)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = jget(v, key)
                if r is not None:
                    return r
        return None

    nel = len(engr_rows)
    heads = jget(cj, "engram_n_heads")
    ngram = jget(cj, "engram_max_ngram_size")
    edim = jget(cj, "engram_head_dim")
    if heads is None or ngram is None:
        ngram = len(mu) // nel
        heads = len(pr) // nel // max(1, int(ngram) - 1)
        print(f"note: config lacks Engram shape keys; derived heads={heads} ngram={ngram}", file=sys.stderr)
    if edim is None:
        sys.exit("reference config has no engram_head_dim")
    heads, ngram, edim = int(heads), int(ngram), int(edim)
    per_layer = (int(ngram) - 1) * int(heads)
    if len(pr) != nel * per_layer or len(mu) != nel * ngram:
        sys.exit(f"Engram arrays do not fit layers={nel} ngram={ngram} heads={heads}: "
                 f"primes {len(pr)}, mult {len(mu)}")
    for i in range(nel):
        s = sum(pr[i * per_layer:(i + 1) * per_layer])
        if s != engr_rows[i]:
            sys.exit(f"engram layer {i}: primes tile to {s}, table has {engr_rows[i]} rows")

    out_kv = []

    def add(k, ty, v):
        out_kv.append((k, ty, v))

    add("general.architecture", "str", "deepseek41")
    add("general.name", "str", os.path.basename(a.pack))
    add("general.alignment", "u32", ALIGN)

    # The engine's flat JSON search finds these anywhere in the tree; repeat them at the root for clarity.
    cj["engram_n_heads"] = heads
    cj["engram_head_dim"] = edim
    cj["engram_max_ngram_size"] = ngram
    cfg_str = json.dumps(cj, separators=(",", ":"))

    for name, ty in SCALARS:
        key = "deepseek41." + name
        v = ref.get(key)
        add(key, ty, float(v) if ty == "f32" else int(v))
    add("deepseek41.norm_topk_prob", "bool", bool(ref.get("deepseek41.norm_topk_prob")))
    add("deepseek41.scoring_func", "str", ref.get("deepseek41.scoring_func"))
    add("deepseek41.config", "str", cfg_str)
    add("deepseek41.compress_ratios", "array", ("i32", ratios_ref if ok_shape else ratio))
    add("deepseek41.kv_source_layer_ids", "array", ("i32", kv_ref if ok_shape else kv_src))
    add("deepseek41.index_source_layer_ids", "array", ("i32", idx_ref if ok_shape else idx_src))

    def pack_key(k, fallback):
        try:
            return int(headers[0].get(k))
        except KeyError:
            return fallback

    pad_id = pack_key("deepseek41.engram.pad_token_id", 2)
    cvocab = pack_key("deepseek41.engram.compressed_vocab_size",
                      max(tm) + 1 if tm and min(tm) >= 0 else None)
    add("deepseek41.engram.pad_id", "u32", pad_id)
    add("deepseek41.engram.compressed_vocab_size", "u32", cvocab)
    add("deepseek41.engram.layer_ids", "array", ("i32", [1, 14]))
    add("deepseek41.engram.rows", "array", ("i64", engr_rows))
    add("deepseek41.engram.token_map", "array", ("i32", tm))
    add("deepseek41.engram.primes", "array", ("u64", pr))
    add("deepseek41.engram.multipliers", "array", ("u64", mu))

    for k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
        try:
            v = int(ref.get(k))
        except KeyError:
            v = -1
        add(k, "i32", v)

    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(out_kv)))
    for k, ty, v in out_kv:
        hdr += kv(k, ty, v)
    hdr += b"\0" * (-len(hdr) % ALIGN)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    with open(tmp, "wb") as f:
        f.write(hdr)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(out_kv)} keys, {len(tm)} token-map ids, primes tile rows "
          f"{engr_rows[0]} / {engr_rows[1]}, ratios match pack tensors: {ok_shape}")


if __name__ == "__main__":
    main()