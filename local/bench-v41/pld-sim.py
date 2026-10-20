#!/usr/bin/env python3
"""pld-sim.py SESSION.jsonl [...] - how many tokens prompt-lookup drafting would get right on recorded Pi agent sessions.

For every assistant message: the history is the conversation before it (user text, earlier answers and tool calls,
tool results; earlier thinking left out, as the chat template drops it) plus what the message generated so far; the
message's own output is its thinking, text and tool calls.  At each step the last 4/3/2 tokens are looked up in the
history (most recent occurrence); the K tokens that followed are the draft, and a verify pass keeps the longest prefix
the real output agrees with, plus the pass's own token.  Without a match the step is a normal single token.
Prints, per K: output tokens, steps, tokens per step, and how many tokens came from drafted passes.  Tokens are the
model's own (models/V41-GSQ/tokenizer, the server's tokenizer).  Only counts are printed, no content."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import strata_tokenizer as ST  # noqa: E402

TD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "V41-GSQ", "tokenizer")
vocab = json.load(open(os.path.join(TD, "vocab.json")))
tokens = [None] * (max(vocab.values()) + 1)
for t, i in vocab.items():
    tokens[i] = t
tokens = [t if t is not None else f"<unused{i}>" for i, t in enumerate(tokens)]
merges = [l.rstrip("\n") for l in open(os.path.join(TD, "merges.txt")) if l.strip() and not l.startswith("#")]
types = json.load(open(os.path.join(TD, "token_type.json")))
tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")
enc = lambda s: tok.encode(s) if s else []

def parts(m, with_thinking):
    out = []
    c = m.get("content")
    if isinstance(c, str):
        return [c]
    for b in c or []:
        t = b.get("type")
        if t == "text": out.append(b.get("text", ""))
        elif t == "thinking" and with_thinking: out.append(b.get("thinking", ""))
        elif t == "toolCall": out.append(b.get("name", "") + json.dumps(b.get("arguments", {}), ensure_ascii=False))
    return out

def simulate(hist, out, K):
    """steps, tokens from drafted passes, passes"""
    H = list(hist)
    idx = {n: {} for n in (2, 3, 4)}
    for n in idx:
        for e in range(n, len(H)):
            idx[n][tuple(H[e - n:e])] = e      # an n-gram ending at e, followed by H[e]
    p = steps = from_pass = passes = 0
    def push(t):
        H.append(t)
        e = len(H) - 1
        for n in idx:
            if e >= n: idx[n][tuple(H[e - n:e])] = e
    while p < len(out):
        draft = []
        for n in (4, 3, 2):
            if len(H) >= n:
                e = idx[n].get(tuple(H[-n:]))
                if e is not None and e < len(H):
                    draft = H[e:e + K]
                    if draft: break
        acc = 0
        while acc < len(draft) and p + acc < len(out) and draft[acc] == out[p + acc]:
            acc += 1
        n_new = min(acc + 1, len(out) - p)
        if draft:
            passes += 1; from_pass += n_new
        for t in out[p:p + n_new]:
            push(t)
        p += n_new; steps += 1
    return steps, from_pass, passes

for path in sys.argv[1:]:
    msgs = [json.loads(l).get("message") for l in open(path)]
    msgs = [m for m in msgs if m]
    hist = []
    res = {K: [0, 0, 0, 0] for K in (4, 8, 16)}   # out tokens, steps, tokens from passes, passes
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            out = enc("".join(parts(m, True)))
            for K in res:
                s, fp, ps = simulate(hist, out, K)
                r = res[K]; r[0] += len(out); r[1] += s; r[2] += fp; r[3] += ps
            hist += enc("".join(parts(m, False)))
        elif role in ("user", "toolResult", "system"):
            hist += enc("\n".join(parts(m, False)))
    print(os.path.basename(path)[:40], f"context at the end {len(hist)} tokens")
    for K, (n, s, fp, ps) in res.items():
        print(f"  K={K:2d}: {n} output tokens in {s} steps = {n / max(s, 1):.2f} tokens/step; drafted passes {ps}, "
              f"{fp} tokens ({100 * fp / max(n, 1):.0f}%), {fp / max(ps, 1):.2f} tokens per drafted pass")
