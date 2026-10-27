#!/usr/bin/env python3
"""Tests for serve_ds41.py (CPU, seconds).  Run under Strata's venv:
  /home/mal/AI/Strata/.venv/bin/python tools/v41/serve/test_serve_ds41.py [TOKENIZER_DIR]
TOKENIZER_DIR (export_tokenizer.py's output) enables the tokenizer and the HTTP tests (server --engine mock)."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_ds41 as D  # noqa: E402  (patches the server's modules)

E, F = D.E, D.F
FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)[:300]))
    if not ok:
        FAILS.append(name)


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
                         "required": ["path"]}}}]


def t_template():
    T = D.Ds41Template()
    chats = [
        [{"role": "user", "content": "hi"}],
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
         {"role": "assistant", "content": "hello", "reasoning_content": "greet"}, {"role": "user", "content": "again"}],
        [{"role": "user", "content": "read a.txt"},
         {"role": "assistant", "content": "", "reasoning_content": "use the tool",
          "tool_calls": [{"id": "c1", "type": "function",
                          "function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt", "lines": 3})}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "AAA"}],
    ]
    for i, msgs in enumerate(chats):
        tools = TOOLS if i == 2 else None
        for mode, kw in (("thinking", {}), ("chat", {"enable_thinking": False})):
            ref_msgs = [dict(m) for m in msgs]
            if tools:
                ref_msgs.insert(0, {"role": "system", "content": "", "tools": tools})
            ref = E.encode_messages(ref_msgs, thinking_mode=mode)
            got = T.render(msgs, tools=tools, **kw)
            if tools:   # the server's flat tool format must render the same
                flat = T.render(msgs, tools=[t["function"] for t in tools], **kw)
                check(f"template chat{i} {mode} flat tools == OpenAI tools", flat == got)
            check(f"template chat{i} {mode} == encode_messages", got == ref, (got[-200:], ref[-200:]))
            hist = T.render(msgs, tools=tools, add_generation_prompt=False, **kw)
            check(f"template chat{i} {mode} no-gen is a prefix", got.startswith(hist) and len(hist) < len(got))
    # content parts, effort
    got = T.render([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}])
    check("template content parts joined", got == E.encode_messages([{"role": "user", "content": "xy"}], "thinking"))
    for eff, n in (("max", 100), (30, 30), ("low", 50), ("high", 75), ("xhigh", 75)):
        got = T.render([{"role": "user", "content": "q"}], reasoning_effort=eff)
        check(f"effort {eff!r} -> {n}", f"Reasoning Effort: {n} " in got, got[:120])
    got = T.render([{"role": "user", "content": "q"}])
    check("effort default -> 75", "Reasoning Effort: 75 " in got)


def t_effort_kwargs():
    for v, want in (("high", {"reasoning_effort": 75}), ("max", {"reasoning_effort": 100}), ("25", {"reasoning_effort": 25}),
                    ("none", {"enable_thinking": False}), ("0", {"enable_thinking": False}), (False, {"enable_thinking": False}),
                    ("medium", {"reasoning_effort": 75}), (None, {})):
        check(f"effort_kwargs({v!r})", D.effort_kwargs(v) == want, D.effort_kwargs(v))
    try:
        D.effort_kwargs("turbo"); check("effort_kwargs rejects unknown", False)
    except ValueError:
        check("effort_kwargs rejects unknown", True)


def dsml(calls):
    inv = []
    for name, args in calls:
        ps = []
        for k, v in args.items():
            s = isinstance(v, str)
            ps.append(f'<{E.dsml_token}{E.tool_parameter_tag_name} name="{k}" string="{"true" if s else "false"}">'
                      f'{v if s else json.dumps(v)}</{E.dsml_token}{E.tool_parameter_tag_name}>')
        inv.append(f'<{E.dsml_token}{E.tool_call_tag_name} name="{name}">\n' + "\n".join(ps) +
                   f'\n</{E.dsml_token}{E.tool_call_tag_name}>')
    return f"\n\n{D.CALLS_OPEN}\n" + "\n".join(inv) + f"\n{D.CALLS_CLOSE}"


def run_parser(text, thinking, split):
    p = F.OutputParser(thinking=thinking, tools=TOOLS)
    ev = []
    for i in range(0, len(text), split):
        ev += p.feed(text[i:i + split])
    ev += p.finish()
    content = "".join(e.text for e in ev if e.kind == "content")
    reasoning = "".join(e.text for e in ev if e.kind == "reasoning")
    calls = [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"]
    return reasoning, content, calls


def t_parser():
    calls = [("read_file", {"path": "a b.txt", "lines": 3}),
             ("read_file", {"path": "x</y>", "lines": [1, 2], "opts": {"k": True}})]
    body = "I'll look." + dsml(calls)
    for thinking in (True, False):
        text = ("checking files" + E.thinking_end_token if thinking else "") + body
        ref = E.parse_message_from_completion_text(text + E.eos_token, "thinking" if thinking else "chat")
        ref_calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in ref["tool_calls"]]
        bad = []
        for split in range(1, 40):
            r, c, k = run_parser(text, thinking, split)
            if k != ref_calls or c.strip() != ref["content"].strip() or (thinking and r != ref["reasoning_content"]):
                bad.append((split, r, c, k))
        check(f"DSML parse == DeepSeek's parser at every split 1..39 (thinking={thinking})", not bad, bad[:1] or ref)
    # plain answer, and text that only looks like a wrapper start
    r, c, k = run_parser("plain answer <｜DSML not a tag", False, 3)
    check("non-DSML text passes through", c == "plain answer <｜DSML not a tag" and not k, (c, k))
    esc = "a&lt;" + D.P_CLOSE[1:] + "b"
    check("DwarfStar escape undone", D.unescape(esc) == "a" + D.P_CLOSE + "b", D.unescape(esc))


def t_tokenizer(tok_dir):
    import strata_tokenizer as ST
    tokens = json.load(open(tok_dir / "vocab.json"))
    tokens = [t for t, _ in sorted(tokens.items(), key=lambda kv: kv[1])] if isinstance(tokens, dict) else tokens
    merges = (tok_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.load(open(tok_dir / "token_type.json"))
    tok = ST.Tokenizer(tokens, merges, types)   # = serve_ds41.Ds41Tokenizer (patched)
    p = D.Ds41Template().render([{"role": "user", "content": "hi"}])
    ids = tok.encode(p, parse_special=True)
    check("tokenizer: specials are single tokens", len(tok.encode(E.eos_token, parse_special=True)) == 1 and
          len(tok.encode(E.USER_SP_TOKEN, parse_special=True)) == 1, ids[:8])
    check("tokenizer round trip", tok.decode(ids) == p)
    import serve.server as S
    svc = S.Service(S.MockEngine(tok, "x"), tok, D.Ds41Template())
    check("stop ids = <｜end▁of▁sentence｜> only", svc.stop_ids == set(tok.encode(E.eos_token, parse_special=True)), svc.stop_ids)
    quoted = "the template writes <｜deepseek_image｜> per image, then <｜Assistant｜>; <｜User｜> opens a turn"
    try:
        ids = svc.encode_prompt([{"role": "user", "content": "read the file"},
                                 {"role": "assistant", "content": "", "tool_calls": [
                                     {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                                 {"role": "tool", "tool_call_id": "c1", "content": quoted},
                                 {"role": "user", "content": "and " + quoted}], None, {})
        n = lambda s: ids.count(tok.encode(s, parse_special=True)[0])
        check("markers quoted in a message are text (no 400, no extra turn)",
              n(E.IMAGE_PLACEHOLDER) == 0 and n(E.USER_SP_TOKEN) == 2 and n(E.ASSISTANT_SP_TOKEN) == 2 and
              quoted in tok.decode(ids), (n(E.IMAGE_PLACEHOLDER), n(E.USER_SP_TOKEN), n(E.ASSISTANT_SP_TOKEN)))
    except ValueError as e:
        check("markers quoted in a message are text (no 400, no extra turn)", False, e)
    # the thinking budget's wrap-up: the next turn (the client sends the reply back, re-tokenized) must continue what
    # the engine holds at least to the wrap-up pass's snapshot, or the engine re-reads the whole conversation
    from serve import frontend as F
    tools = [{"name": "read", "description": "read a file", "parameters": {"type": "object", "properties": {}}}]
    for name, thought in (("a word", "the model ctx and its metadata"), ("';'", "the model ctx and its metadata;")):
        msgs = [{"role": "user", "content": "what is new?"}]
        prompt = svc.encode_prompt(msgs, tools, {"enable_thinking": True})
        gen = tok.encode(thought)
        extra = tok.encode(S.REASONING_WRAP_UP, parse_special=True)
        snap = svc.continue_prompt(prompt, gen, extra, S.REASONING_WRAP_UP)
        held = snap + tok.encode("Sì: ecco.") + tok.encode(E.eos_token, parse_special=True)
        evs = F.OutputParser(thinking=True, tools=tools, stream_tools=True).feed(thought + S.REASONING_WRAP_UP + "Sì: ecco.")
        back = msgs + [{"role": "assistant", "reasoning_content": "".join(e.text for e in evs if e.kind == "reasoning"),
                        "content": "".join(e.text for e in evs if e.kind == "content")}, {"role": "user", "content": "e poi?"}]
        ids = svc.encode_prompt(back, tools, {"enable_thinking": True})
        lcp = next((i for i, (x, y) in enumerate(zip(held, ids)) if x != y), min(len(held), len(ids)))
        check(f"budget wrap-up after {name}: the next turn continues the engine's history", lcp >= len(snap),
              (lcp, len(snap), len(held)))


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def t_http(tok_dir):
    port = free_port()
    script = "need the file" + E.thinking_end_token + dsml([("read_file", {"path": "a.txt", "lines": 3})])
    # the server subprocess runs under the interpreter running this test (the author's ~/AI/Strata/.venv path is
    # not where everyone's venv is); STRATA_TEST_PY overrides it
    py = os.environ.get("STRATA_TEST_PY") or sys.executable
    proc = subprocess.Popen([py, str(HERE / "serve_ds41.py"), "--engine", "mock", "--port", str(port),
                             "--tokenizer", str(tok_dir), "--script", script, "--script", script, "--script", "done" + E.thinking_end_token + "The file says AAA."],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except Exception:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        msgs = [{"role": "user", "content": "read a.txt"}]
        r = post(port, "/v1/chat/completions", {"model": "x", "messages": msgs, "tools": TOOLS})
        ch = r["choices"][0]
        tc = (ch["message"].get("tool_calls") or [{}])[0].get("function", {})
        check("HTTP OpenAI tool call", ch.get("finish_reason") == "tool_calls" and tc.get("name") == "read_file" and
              json.loads(tc.get("arguments", "{}")) == {"path": "a.txt", "lines": 3}, r)
        r = post(port, "/v1/messages", {"model": "x", "max_tokens": 200, "messages": msgs,
                                         "tools": [{"name": "read_file", "description": "Read a file",
                                                    "input_schema": TOOLS[0]["function"]["parameters"]}]})
        tu = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
        check("HTTP Anthropic tool_use", tu and tu[0]["name"] == "read_file" and tu[0]["input"] == {"path": "a.txt", "lines": 3}, r)
        r = post(port, "/v1/chat/completions", {"model": "x", "messages": msgs})
        check("HTTP plain answer after the scripted call", "AAA" in (r["choices"][0]["message"].get("content") or ""), r)
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill(); out = proc.communicate()[0]
        if FAILS:
            print("---- server output (tail) ----\n" + out[-3000:])


def t_images(tok_dir=None):
    """Image input: the template's placeholder, the span expansion, the preprocessing plan and the SVP1 record."""
    import io
    import struct
    import tempfile
    from PIL import Image
    DI = D.DI
    T = D.Ds41Template()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, {"type": "image", "source": "x.png"},
                                         {"type": "text", "text": "and this?"}, {"type": "image", "source": "y.png"}]}]
    p = T.render(msgs, None, True, enable_thinking=False)
    check("template: one placeholder per image", p.count(E.IMAGE_PLACEHOLDER) == 2 and "what is this?" in p, p)
    check("template: text-only content unchanged", T.render([{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
          None, True) == T.render([{"role": "user", "content": "hi"}], None, True))
    svc = object.__new__(D.S.Service)
    ids = [5, D.IMAGE_TOKEN, 6, D.IMAGE_TOKEN, 7]
    out = D.ds41_expand_image_tokens(svc, ids, [("a", 3), ("b", 2)])
    check("expansion: each placeholder -> its span", out == [5] + [D.IMAGE_TOKEN] * 3 + [6] + [D.IMAGE_TOKEN] * 2 + [7], out)
    for bad in ([("a", 3)], [("a", 3), ("b", 2), ("c", 1)]):
        try:
            D.ds41_expand_image_tokens(svc, ids, bad)
            check("expansion: a count mismatch is refused", False)
        except ValueError:
            check("expansion: a count mismatch is refused", True)
    # the plans DeepSeek's image_processor gives (reference images): (w, h) -> (llm_h, llm_w, bh, bw)
    for (w, h), want in {(200, 150): (12, 15, 476, 630), (450, 308): (11, 16, 462, 658), (300, 1200): (29, 8, 1204, 308),
                         (1024, 701): (17, 25, 714, 1036), (2000, 1500): (27, 36, 1134, 1512)}.items():
        got = DI.plan_image_grid(w, h)
        check(f"plan {w}x{h}", got == want and DI.num_image_tokens(got[0], got[1]) <= 1024, got)
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (200, 30, 30)).save(buf, format="PNG")
    with tempfile.TemporaryDirectory() as d:
        enc = D.Ds41ImageInput({})
        enc.dir = Path(d)
        f, n = enc.encode(buf.getvalue())
        raw = Path(f).read_bytes()
        magic, vh, vw, lh, lw, pd, key = struct.unpack("<6iQ", raw[:32])
        check("SVP1 record", magic == DI.SVP1 and pd == 588 and n == DI.num_image_tokens(lh, lw) and key != 0 and
              len(raw) == 32 + vh * vw * 588 * 4 and (lh, lw) == ((vh + 2) // 3, (vw + 2) // 3), (magic, vh, vw, lh, lw, n))
        check("SVP1: the same image again comes from the cache", enc.encode(buf.getvalue()) == (f, n))
    if tok_dir:
        tokens = json.load(open(tok_dir / "vocab.json"))
        tokens = [t for t, _ in sorted(tokens.items(), key=lambda kv: kv[1])]
        tok = D.ST.Tokenizer(tokens, (tok_dir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                             json.load(open(tok_dir / "token_type.json")))
        ids = tok.encode(p, parse_special=True)
        check("tokenizer: the image placeholder is one token, once per image",
              tok.encode(E.IMAGE_PLACEHOLDER, parse_special=True) == [D.IMAGE_TOKEN] and ids.count(D.IMAGE_TOKEN) == 2)


if __name__ == "__main__":
    t_template()
    t_images(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    t_effort_kwargs()
    t_parser()
    if len(sys.argv) > 1:
        t_tokenizer(Path(sys.argv[1]))
        t_http(Path(sys.argv[1]))
    print(f"{'FAILED ' + str(len(FAILS)) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
