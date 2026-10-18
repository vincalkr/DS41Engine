#!/usr/bin/env python3
"""chatbench.py PROMPT [--think] [--max N] [--temp T] [--file F] [--quiet] - one streamed chat request to
127.0.0.1:8150: TTFT, completion tokens, decode tok/s (tokens after the first / stream time after it), the answer."""
import json, time, urllib.request, argparse
ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?"); ap.add_argument("--file"); ap.add_argument("--think", action="store_true")
ap.add_argument("--max", type=int, default=800); ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
prompt = open(a.file).read() if a.file else a.prompt
body = {"model": "x", "stream": True, "max_tokens": a.max, "temperature": a.temp,
        "messages": [{"role": "user", "content": prompt}], "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(a.think)}}
req = urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time(); t1 = None; n = 0; text = []; reas = []; usage = None; fin = None
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        j = json.loads(d)
        if j.get("usage"): usage = j["usage"]
        for c in j.get("choices", []):
            de = c.get("delta", {})
            if (de.get("content") or "") + (de.get("reasoning_content") or ""):
                if t1 is None: t1 = time.time()
                n += 1
            if de.get("content"): text.append(de["content"])
            if de.get("reasoning_content"): reas.append(de["reasoning_content"])
            if c.get("finish_reason"): fin = c["finish_reason"]
t2 = time.time()
ct = (usage or {}).get("completion_tokens", n)
print(f"prompt_tokens={(usage or {}).get('prompt_tokens')} ttft={t1 - t0:.2f}s completion_tokens={ct} "
      f"decode={(ct - 1) / max(t2 - t1, 1e-9):.2f} tok/s total={t2 - t0:.1f}s finish={fin} reasoning_chars={len(''.join(reas))}")
if not a.quiet: print("".join(text))
