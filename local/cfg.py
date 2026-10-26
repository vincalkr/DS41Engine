#!/usr/bin/env python3
"""local/cfg.py - edit local/ds41-local.json without mangling the arg list.

Usage:
  python3 local/cfg.py show
  python3 local/cfg.py env NAME=VALUE [NAME=VALUE ...]     # set/unset (NAME= to remove)
  python3 local/cfg.py arg --flag VALUE --flag2 ...        # replace these flags (removes old occurrences first)
  python3 local/cfg.py arg-drop --flag ...                 # remove flags (and their value if it is not a flag)
  python3 local/cfg.py save-copy NAME                      # keep a copy before an experiment
"""
import json, shutil, sys
from pathlib import Path

P = Path(__file__).resolve().parent / "ds41-local.json"
VALUE_FLAGS = {"--route-bias", "--route-bias-file", "--arena-gib", "--ctx", "--threads", "--prompt-batch",
               "--prefill-chunk", "--stop", "--vram-margin", "--peer-device", "--kv-type", "--rank", "--engram",
               "--backend", "--experts", "-m", "--slots"}


def load():
    return json.loads(P.read_text())


def save(d):
    P.write_text(json.dumps(d, indent=1) + "\n")


def drop_args(args, flags):
    """Drop the named flags and their values (a flag consumes the next token only when that token is not itself a
    flag).  The earlier version had the condition inverted and could eat the following flag or leave the value
    behind: that is how a stray --arena-gib 64 stayed in the args."""
    out, i = [], 0
    while i < len(args):
        if args[i] in flags:
            i += 1
            if i < len(args) and not args[i].startswith("--"):
                i += 1
            continue
        out.append(args[i])
        i += 1
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    cmd = sys.argv[1]
    d = load()
    if cmd == "show":
        print(" ".join(d["args"]))
        print(json.dumps(d.get("env", {}), indent=1))
        return 0
    if cmd == "save-copy":
        shutil.copy(P, P.with_suffix(".json." + sys.argv[2]))
        print("copiato in", P.with_suffix(".json." + sys.argv[2]))
        return 0
    if cmd == "env":
        for kv in sys.argv[2:]:
            k, _, v = kv.partition("=")
            if v == "":
                d.setdefault("env", {}).pop(k, None)
            else:
                d.setdefault("env", {})[k] = v
        save(d); print("env:", d.get("env", {})); return 0
    if cmd == "arg":
        new = sys.argv[2:]
        d["args"] = drop_args(d["args"], {new[i] for i in range(0, len(new), 2) if new[i].startswith("--")}) + new
        save(d); print("args:", " ".join(d["args"])); return 0
    if cmd == "arg-drop":
        d["args"] = drop_args(d["args"], set(sys.argv[2:]))
        save(d); print("args:", " ".join(d["args"])); return 0
    print("comando sconosciuto:", cmd); return 1


if __name__ == "__main__":
    sys.exit(main())
