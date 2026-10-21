#!/usr/bin/env python3
"""check_images.py - ask the running server (./run.sh --config local/ds41-local-vision.json) about pictures.

    python3 tools/v41/vision/check_images.py [--url http://127.0.0.1:8150] [--question "..."] IMAGE [IMAGE ...]

Each image goes as an OpenAI image_url data: URL with the question, temperature 0, thinking off; prints the answer,
the prompt / completion tokens and the wall time.  Several images can also go in ONE request (--together).
Standard library only."""
import argparse
import base64
import json
import mimetypes
import time
import urllib.request


def ask(url, images, question, max_tokens):
    parts = [{"type": "text", "text": question}]
    for p in images:
        mime = mimetypes.guess_type(p)[0] or "image/png"
        data = base64.b64encode(open(p, "rb").read()).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": parts}], "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8150")
    ap.add_argument("--question", default="Describe this image in two or three sentences.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--together", action="store_true", help="all images in one request")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    groups = [a.images] if a.together else [[p] for p in a.images]
    for g in groups:
        out, wall = ask(a.url, g, a.question, a.max_tokens)
        u = out.get("usage", {})
        msg = out["choices"][0]["message"].get("content") or ""
        print(f"== {', '.join(g)}: prompt {u.get('prompt_tokens')} tokens, answer {u.get('completion_tokens')}, "
              f"{wall:.1f} s\n{msg.strip()}\n", flush=True)


if __name__ == "__main__":
    main()
