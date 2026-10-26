#!/usr/bin/env python3
"""local/encode_prompt.py - tokenize text into the int32 id file ds41_generate --ids-file expects.

  local/encode_prompt.py "some prompt" out.i32
  local/encode_prompt.py --chat "hello" out.i32        (DeepSeek-V4.1's own chat encoder)

Local helper for this machine's checkout: it uses the BPE in models/rco-iq2xxs/tokenizer with llama.cpp's `joyai-llm`
pre-split (the same tables and the same pre-tokenizer serve_ds41.py gives the server).  Not part of the engine.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

# the tokenizer of whichever model is in models/ (the GSQ's and the ds4 1017 Q2's are the same tables);
# STRATA_TOKENIZER=<dir> picks one
TPATH = Path(os.environ["STRATA_TOKENIZER"]) if os.environ.get("STRATA_TOKENIZER") else next(
    (d for d in (ROOT / "models" / "rco-iq2xxs" / "tokenizer", ROOT / "models" / "V41-GSQ" / "tokenizer") if d.is_dir()),
    ROOT / "models" / "rco-iq2xxs" / "tokenizer")


class _Ds41Tokenizer(ST.Tokenizer):
    """serve_ds41.py's subclass: V4.1's special tokens (<｜User｜>, <｜end▁of▁sentence｜> ...) are not in the byte
    alphabet, so they decode to their own text, as llama.cpp detokenizes control / user-defined tokens."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._literal = {i: t.encode("utf-8") for t, i in self.special_tokens.items()}

    def token_bytes(self, i: int) -> bytes:
        lit = self._literal.get(i)
        return lit if lit is not None else super().token_bytes(i)


def load_tokenizer():
    vocab = json.loads((TPATH / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (TPATH / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((TPATH / "token_type.json").read_text())
    return _Ds41Tokenizer(tokens, merges, types, pre="joyai-llm")


def main():
    argv = sys.argv[1:]
    chat = False
    if argv and argv[0] == "--chat":
        chat = True
        argv = argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    text, out = argv
    tok = load_tokenizer()
    if chat:
        sys.path.insert(0, str(ROOT / "tools" / "v41"))
        import dsv41_encoding as E
        # "chat" = no thinking block: keeps the answer short and comparable run to run
        text = E.encode_messages([{"role": "user", "content": text}], "chat")
    ids = tok.encode(text, parse_special=True)
    np.array(ids, dtype=np.int32).tofile(out)
    print(f"{len(ids)} tokens -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
