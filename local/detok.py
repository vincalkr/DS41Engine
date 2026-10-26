#!/usr/bin/env python3
"""local/detok.py - turn ids (argv or stdin, one line of ids) back into text with V4.1's tokenizer.

  local/detok.py 671 6102 294 8760 344
  local/detok.py < out.ids
Local helper, same tables as local/encode_prompt.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_prompt import load_tokenizer  # noqa: E402

tok = load_tokenizer()
data = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
ids = [int(x) for x in data.replace(",", " ").split()]
sys.stdout.write(tok.decode(ids))
