#!/usr/bin/env python3
"""local/make-dev-prompts.py - writes the two prompts update-rank.sh measures on (local/prompts is not in git):
dev-code (a code review with a patch over two engine sources, ~16.7K tokens) and dev-docs (a README section over the
README and the HIP notes, ~10.8K tokens), as chat turns without thinking (local/encode_prompt.py --chat).  The texts
are taken from the files as they are when it runs (the ones on this box: 2026-10-27), so a later rebuild gives
slightly different prompts; --force rewrites existing ones."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "local/prompts"
P.mkdir(exist_ok=True)
rd = lambda f: (ROOT / f).read_text()
texts = {
    "dev-code": "You are working on a C++/HIP inference engine. Here are two source files.\n\n"
                "=== tools/v41/engine/ds41_conv_disk.hpp ===\n" + rd("tools/v41/engine/ds41_conv_disk.hpp") +
                "\n=== tools/v41/engine/ds41_conv_pool.cpp ===\n" + rd("tools/v41/engine/ds41_conv_pool.cpp") +
                "\n\nReview this code: list the concrete bugs or risks you find (file, function, why), then write a patch in "
                "unified diff format that fixes the most important one, with a short explanation.",
    "dev-docs": "Here is the README of a project and its notes on the AMD HIP port.\n\n=== README.md ===\n" + rd("README.md") +
                "\n=== docs/ds4/AMD_HIP_NOTES.md (first part) ===\n" + rd("docs/ds4/AMD_HIP_NOTES.md")[:24000] +
                "\n\nWrite a new README section in Markdown titled \"Troubleshooting\": the five problems a new user is most "
                "likely to hit when building and running this project, each with its symptom, its cause and the exact fix, "
                "in the README's style.",
}
for name, t in texts.items():
    if (P / f"{name}.i32").exists() and "--force" not in sys.argv:
        print(f"{name}: exists (--force rewrites it)")
        continue
    (P / f"{name}.txt").write_text(t)
    subprocess.run([sys.executable, str(ROOT / "local/encode_prompt.py"), "--chat", t, str(P / f"{name}.i32")], check=True)
