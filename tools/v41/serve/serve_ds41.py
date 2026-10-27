#!/usr/bin/env python3
"""Strata's API server (~/AI/Strata/serve/server.py: OpenAI + Anthropic endpoints, streaming, sessions, llama.cpp-style
metrics) in front of the DeepSeek-V4.1-Flash engine (`ds41_generate --serve`).  The server is generic; this shim
swaps in V4.1's pieces:

- tokenizer: the GGUF's BPE (tokenizer dir from export_tokenizer.py) with llama.cpp's `joyai-llm` pre-split
  (tools/strata_tokenizer.py; == HF tokenizers on 465,645 tokens)
- chat template: DeepSeek's own encoder (dsv41_encoding.encode_messages), not Jinja
- reasoning effort: V4.1's numeric 1-100 ("Reasoning Effort: N" system prefix).  Names: high / xhigh = 75,
  max = 100, a number 1-100 as given, none / 0 = thinking off; low = 50 (DeepSeek's encoder),
  minimal / medium = 75
- end of turn: <｜end▁of▁sentence｜>
- tool calls: DSML (`<｜DSML｜ invoke name="X">` + `<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>`,
  note the LEADING SPACE in the tag names) parsed as DeepSeek's parse_tool_calls does; the `calls` wrapper is
  stripped from the stream.  A value may carry an escape of its own closing tag: undone.

  serve_ds41.py --engine strata --config local/ds41-local.json --port 8150   (./run.sh does this)
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V41 = HERE.parent
# Strata's server this shim sits in front of.  Default: the installed layout (~/AI/Strata).  STRATA_HOME points it at
# another tree, and this checkout is one (a Strata fork ships its own serve/), so a self-contained install needs no
# ~/AI/Strata at all.
# STRATA_HOME names the checkout that provides serve/, tools/ and strata_tokenizer.  This repository is one, so
# that is the default; the upstream flow pointed it at an ambient ~/AI/Strata install instead.
STRATA = Path(os.environ.get("STRATA_HOME") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(STRATA))
sys.path.insert(0, str(STRATA / "tools"))
sys.path.insert(0, str(V41.parent))          # Strata-DS4/tools: strata_tokenizer.py with the joyai-llm pre-split
sys.path.insert(0, str(V41))

import strata_tokenizer as ST  # noqa: E402
assert "joyai-llm" in getattr(ST, "PRE_PATTERNS", {}), ST.__file__
import dsv41_encoding as E  # noqa: E402
from serve import frontend as F  # noqa: E402
from serve import server as S  # noqa: E402

# ---- tokenizer: the server builds ST.Tokenizer(tokens, merges, types); give it V4.1's pre-split
_Tok = ST.Tokenizer


class Ds41Tokenizer(_Tok):
    def __init__(self, tokens, merges, token_types=None, pre="joyai-llm", **k):
        super().__init__(tokens, merges, token_types, pre="joyai-llm", **k)
        self._literal = {i: t.encode("utf-8") for t, i in self.special_tokens.items()}

    def token_bytes(self, i: int) -> bytes:
        # V4.1's special tokens (<｜User｜>, <｜end▁of▁sentence｜>, ...) are not in the byte alphabet: they decode to
        # their own text, as llama.cpp detokenizes control / user-defined tokens (ASCII ones never hit this)
        lit = self._literal.get(i)
        return lit if lit is not None else super().token_bytes(i)


ST.Tokenizer = Ds41Tokenizer

# ---- reasoning effort: client spellings -> template kwargs (reasoning_effort = an int 1-100, or thinking off)
# The model's own levels are the three the encoder knows (dsv41_encoding.REASONING_EFFORT_MAPPINGS: low 50,
# high 75, max 100, default high), and they are rendered as the encoder renders them; the clients' extra spellings
# (minimal, medium, xhigh) fold onto high, no invented budgets.
# DeepSeek-V4.1 renders the effort as "Reasoning Effort: N (range 1-100, ...)": HIGH 75, MAX 100, and low 50 as
# DeepSeek's own encoder has it. A numeric 1-100 is also accepted and rendered as given.
EFFORT = {"none": 0, "off": 0, "disabled": 0, "false": 0,
          "minimal": 75, "low": 50, "medium": 75,
          "high": 75, "xhigh": 75, "max": 100, "maximum": 100}


def effort_kwargs(value) -> dict:
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key.isdigit() and 0 <= int(key) <= 100:
        level = int(key)
    elif key in EFFORT:
        level = EFFORT[key]
    else:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium, high, max or 1-100")
    return {"enable_thinking": False} if level == 0 else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget: V4.1 has two named levels (HIGH 75, MAX 100), so a small or medium budget asks
    for HIGH and a large one for MAX."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": 75 if n < 8192 else 100}


for mod in (F, S):
    if hasattr(mod, "effort_kwargs"):
        mod.effort_kwargs = effort_kwargs
    if hasattr(mod, "budget_effort"):
        mod.budget_effort = budget_effort


def _level(effort):
    """Whatever reaches the template (ours, or the server's own names) -> 1..100, or None for the default (75)."""
    if effort is None or effort == "":
        return None
    if isinstance(effort, int):
        return max(1, min(100, effort))
    key = str(effort).strip().lower()
    if key.isdigit():
        return max(1, min(100, int(key)))
    return EFFORT.get(key) or None


# ---- chat template: DeepSeek's encoder
GEN_SUFFIXES = (E.ASSISTANT_SP_TOKEN + E.thinking_start_token, E.ASSISTANT_SP_TOKEN + E.thinking_end_token)


def _text(content):
    if content is None or isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text", None))


def _content(content):
    """A message's content for DeepSeek's encoder: text, or - when it holds images (the server's {"type": "image",
    "source"} items) - text and image blocks in order; the encoder writes one <｜deepseek_image｜> per image block
    (dsv41_encoding.process_image_messages), in the order images_of() lists the sources."""
    if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image" for p in content):
        return [{"type": "image", "source": p.get("source") or "image"} if p.get("type") == "image" else {"type": "text", "text": p.get("text", "")}
                for p in content if isinstance(p, dict) and p.get("type") in ("image", "text", "input_text", None)]
    return _text(content)


def to_encoder_messages(messages, tools):
    out = []
    for m in messages:
        m = dict(m)
        m["content"] = _content(m.get("content"))
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                fn.setdefault("arguments", "{}")
                calls.append({"type": "function", "id": c.get("id"), "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    if tools:   # the server passes flat {"name", "description", "parameters"}; the encoder wants OpenAI's wrapper
        tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        if out and out[0].get("role") == "system":
            out[0] = dict(out[0], tools=tools)
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out


class Ds41Template:
    """Stands in for F.ChatTemplate: render(messages, tools, add_generation_prompt, **kwargs) -> prompt text."""
    source = "{# DeepSeek-V4.1: rendered by serve_ds41.py with DeepSeek's encoder (dsv41_encoding.py) #}"

    def __init__(self, path=None):
        pass

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        mode = "chat" if kw.get("enable_thinking") is False else "thinking"
        prompt = E.encode_messages(to_encoder_messages(messages, tools), thinking_mode=mode,
                                   reasoning_effort=_level(kw.get("reasoning_effort")))
        if not add_generation_prompt:
            for s in GEN_SUFFIXES:
                if prompt.endswith(s):
                    return prompt[:-len(s)]
        return prompt


F.ChatTemplate = Ds41Template
S.ChatTemplate = Ds41Template

# ---- end of turn
S.IM_END = E.eos_token
_svc_init = S.Service.__init__


def _svc_init_ds41(self, engine, tokenizer, template, *a, **k):
    _svc_init(self, engine, tokenizer, template, *a, **k)
    # REPLACE, not add to: the server also encodes its default "<|endoftext|>", which V4.1's vocabulary splits into plain
    # pieces ("<", "|", ...) - every DSML tag would then end the reply
    eos = tokenizer.encode(E.eos_token, parse_special=True)
    assert len(eos) == 1, eos
    self.stop_ids = set(eos)
    self.effort_end = False      # #458's tail-turn trick is the server's default template text; V4.1's effort lives in the first system turn
    # V4.1's own markers are GGUF type 4 (<｜User｜>, <｜Assistant｜>, <｜deepseek_image｜>, ｜DSML｜, ...), which the
    # server's literal list (type-3 control tokens only) leaves out.  Written inside a message's text - an agent
    # reading serve_ds41.py or a chat template - they opened a turn, or <｜deepseek_image｜> made DeepSeek's encoder
    # refuse the request with a 400 that every later turn of the conversation repeated.  Now they are text there too;
    # only the ones the encoder writes are markers.
    types = getattr(tokenizer, "token_types", None)
    if types:
        user_defined = [t for t, i in getattr(tokenizer, "special_tokens", {}).items() if types[i] == 4]
        self.literals = F.literal_tags(list(getattr(tokenizer, "control_tokens", ())) + user_defined)


S.Service.__init__ = _svc_init_ds41

# ---- the #123 thinking budget must close V4.1's thinking token
# serve/server.py holds plain-text markers ("</think>"); both wrap-up texts go through tok.encode(..., parse_special=True),
# so putting the model's own end-of-thinking token there is what actually closes the thinking and lets it answer.
# Nothing after it: DeepSeek's encoder writes the answer right after </think> when the client sends the turn back, and
# the default "\n\n" there (the reply's content drops it) made every turn after a budget wrap-up re-read the whole
# conversation (the engine held "</think>\n\n", the next turn "</think>" and the answer).
S.REASONING_CLOSE = "\n" + E.thinking_end_token
S.REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n" + E.thinking_end_token

# ---- tool calls: DSML
CALLS_OPEN = f"<{E.dsml_token}{E.tool_calls_block_name}>"           # <｜DSML｜ calls>
CALLS_CLOSE = f"</{E.dsml_token}{E.tool_calls_block_name}>"
INVOKE_OPEN = f"<{E.dsml_token}{E.tool_call_tag_name}"               # <｜DSML｜ invoke   (name="X">)
INVOKE_CLOSE = f"</{E.dsml_token}{E.tool_call_tag_name}>"
P_OPEN = f"<{E.dsml_token}{E.tool_parameter_tag_name}"
P_CLOSE = f"</{E.dsml_token}{E.tool_parameter_tag_name}>"
_P_RE = re.compile(r' name="(.*?)" string="(true|false)">(.*?)' + re.escape(P_CLOSE), re.DOTALL)


def unescape(value: str) -> str:
    """A value may arrive with its own closing tag escaped as &lt;... (and &amp; chains before it); DeepSeek's encoder
    never escapes, so only that exact spelling is undone."""
    tail = P_CLOSE[1:]
    return re.sub(r"&((?:amp;)*)lt;" + re.escape(tail),
                  lambda m: ("&" + m.group(1)[:-4] + "lt;" if m.group(1) else "<") + tail, value)


def ds41_call_end(text: str) -> int:
    return text.find(INVOKE_CLOSE)


def ds41_parse_tool_call(body: str, schema=None):
    m = re.match(r'\s*name="(.*?)">', body, re.DOTALL)
    if not m:
        raise ValueError("malformed tool call: " + body[:80])
    args = {}
    for key, is_str, value in _P_RE.findall(body[m.end():]):
        value = unescape(value)
        if is_str == "true":
            args[key] = value
        else:
            try:
                args[key] = json.loads(value)
            except ValueError:
                args[key] = value
    return F.ToolCall(name=m.group(1), arguments=args)


# The served path TRANSLATES the DSML stream into the XML form the server's parser reads.
#
# `serve/frontend.py` detects a call with `_follower` / `_bare_opener` / `call_end` around the server's
# `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`, and a DSML `invoke` is not in that
# vocabulary.  Measured 2026-10-16 with the three hooks below merely overridden (CALL_START/CALL_END, call_end,
# parse_tool_call): the DSML call came out as plain content - `tool_calls: null`, the markup in `content`, the tool
# never ran - because `_follower` looks for `<function=`, `<parameter=` or `{` right after the opener.  Translating
# keeps every one of that parser's heuristics (code fences, a quoted call, the streamed body) working on the model's
# own syntax instead of teaching it a second one.
_DSML = f"<{E.dsml_token}"
_INVOKE_RE = re.compile(re.escape(INVOKE_OPEN) + r'\s*name="([^"]+)"\s*>')
_PARAM_RE = re.compile(re.escape(P_OPEN) + r'\s*name="([^"]+)"\s+string="(true|false)"\s*>(.*?)' + re.escape(P_CLOSE),
                       re.DOTALL)
_WRAP_RE = re.compile(re.escape(CALLS_OPEN) + r"|" + re.escape(CALLS_CLOSE))


def ds41_translate(buf: str) -> str:
    """DSML -> the XML call form: `<｜DSML｜ invoke name="X">` -> `<tool_call><function=X>`,
    `<｜DSML｜ parameter name="K" string="b">V</｜DSML｜ parameter>` -> `<parameter=K>V</parameter>`,
    `</｜DSML｜ invoke>` -> `</function></tool_call>`; the optional `<｜DSML｜ calls>` wrapper is dropped."""
    buf = _WRAP_RE.sub("", buf)
    buf = _PARAM_RE.sub(lambda m: "<parameter=" + m.group(1) + ">" + m.group(3) + "</parameter>", buf)
    buf = _INVOKE_RE.sub(lambda m: "<tool_call>\n<function=" + m.group(1) + ">\n", buf)
    return buf.replace(INVOKE_CLOSE, "</function>\n</tool_call>")


def ds41_hold(buf: str) -> int:
    """Bytes at the END of `buf` that must not be emitted yet: a `<｜DSML｜` that is still being written (the
    tokenizer splits it across deltas), or an element whose closing tag has not arrived."""
    p = buf.rfind(_DSML)
    if p < 0:
        return 0
    tail = buf[p:]
    if ">" not in tail:
        return len(buf) - p
    m = re.match(re.escape(_DSML) + r"\s*(\w+)", tail)
    # the tag names carry the DSML leading space (E.tool_parameter_tag_name == " parameter"), the regex has eaten it
    if m and m.group(1) == E.tool_parameter_tag_name.strip() and P_CLOSE not in tail:
        return len(buf) - p                     # a parameter element whose own close has not arrived
    return 0


_op_init, _op_feed, _op_finish = F.OutputParser.__init__, F.OutputParser.feed, F.OutputParser.finish


def _op_init_ds41(self, *a, **k):
    _op_init(self, *a, **k)
    self.stream_tools = False          # the streaming scanner reads the <function=...> body
    self.dsml_hold = ""


def _strip_wrappers(self, delta: str, final: bool) -> str:
    """Translate the DSML stream for the server's parser, holding back a partial tag or element."""
    buf = self.dsml_hold + delta
    # `_hold` covers a suffix that is a PREFIX of one of these tags - an opener split across deltas ("<｜DSML", "｜ in")
    # and, just as important, a closing tag split the same way (`</｜DSML` + `｜ invok` + `e>`): ds41_hold only sees a
    # complete `<｜DSML｜` token, so the closers have to be held here (measured: without them the close leaked out raw)
    tags = (_DSML, INVOKE_CLOSE, P_CLOSE, CALLS_OPEN, CALLS_CLOSE)
    keep = 0 if final else max(self._hold(buf, tags), ds41_hold(buf))
    self.dsml_hold = buf[len(buf) - keep:] if keep else ""
    return ds41_translate(buf[:len(buf) - keep] if keep else buf)


def _op_feed_ds41(self, delta: str):
    return _op_feed(self, _strip_wrappers(self, delta, False))


def _op_finish_ds41(self, reason=None):
    # reason (how the turn ended) is Strata's newer signature: passed through unchanged
    out = _op_feed(self, _strip_wrappers(self, "", True)) if self.dsml_hold else []
    return out + _op_finish(self, reason)


F.OutputParser.__init__ = _op_init_ds41
F.OutputParser.feed = _op_feed_ds41
F.OutputParser.finish = _op_finish_ds41
S.OutputParser = F.OutputParser

# ---- images (opt-in: a "vision" section in the config and --vision <gguf> in the engine's args)
# The engine encodes the images itself (ds41_generate --vision: the ViT on the second card, or on card 0 for the
# request).  So the server's encoder process is replaced by the preprocessing alone: DeepSeek's image_processor
# (tools/v41/vision/ds41_image.py, bit-identical patches) writes one SVP1 record per image, and the server's existing
# path does the rest - the FIFO, the combined file, the GENI line.  Each <｜deepseek_image｜> the template writes
# becomes its image's whole span: llm_h * (llm_w + 1) + 2 image tokens.
sys.path.insert(0, str(V41 / "vision"))
import hashlib as _hashlib   # noqa: E402
import shutil as _shutil   # noqa: E402
import threading as _threading   # noqa: E402

import ds41_image as DI   # noqa: E402


class Ds41ImageInput(S.Vision):
    """Stands in for S.Vision (the strata-vision process): no process, the patch grid computed here."""

    def __init__(self, cfg: dict, log=None, env=None, lazy: bool = False):
        self.dir = self.work_dir()
        self.lock = _threading.Lock()
        self.cache = {}
        self.proc, self.stopped = None, False
        self.max_tokens = int(cfg.get("max_tokens") or DI.MAX_TOKENS)

    def alive(self) -> bool:
        return True

    def unload(self):
        pass

    def restart(self):
        pass

    def close(self):
        pass

    def encode(self, source, keep=()):
        """-> (SVP1 record file, image tokens) for one image; cached by the image's bytes."""
        data = self.normalize(source if isinstance(source, bytes) else self.load(source))
        digest = _hashlib.sha256(data).digest()
        name = digest[:16].hex()
        with self.lock:
            if name in self.cache:
                self.cache[name] = self.cache.pop(name)
                return self.cache[name]
            try:
                patches, vh, vw, lh, lw = DI.load_patches(data, self.max_tokens)
            except Exception as e:  # noqa: BLE001 - an unreadable picture is the client's 400, not a crash
                raise ValueError(f"the image could not be read ({e})") from None
            key = int.from_bytes(digest[16:24], "little") or 1
            out = self.dir / f"{name}.svp"
            out.write_bytes(DI.svp1_record(patches, vh, vw, lh, lw, key))
            self.cache[name] = (out, DI.num_image_tokens(lh, lw))
            held = {str(p) for p in keep} | {str(out)}
            for old in list(self.cache):
                if len(self.cache) <= 64:
                    break
                if str(self.cache[old][0]) not in held:
                    self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[name]

    def shutdown(self):
        _shutil.rmtree(self.dir, ignore_errors=True)


S.Vision = Ds41ImageInput
IMAGE_TOKEN = 129264   # <｜deepseek_image｜> (config.json image_token_id; vocab.json agrees)


def ds41_expand_image_tokens(self, ids, encoded):
    """Every <｜deepseek_image｜> the template wrote -> that image's span of image tokens (prompt order).  A user's
    literal placeholder never gets here: DeepSeek's encoder refuses it in a message's text."""
    out, k = [], 0
    for t in ids:
        if t != IMAGE_TOKEN:
            out.append(t)
            continue
        if k == len(encoded):
            raise ValueError("the prompt and its images do not match")
        out += [IMAGE_TOKEN] * encoded[k][1]
        k += 1
    if k != len(encoded):
        raise ValueError("the prompt and its images do not match")
    return out


S.Service.expand_image_tokens = ds41_expand_image_tokens

# ---- full text log (llama.cpp-style): every request's thinking / answer / tool calls as they are generated, so a run
# can be watched (`tail -f`) and checked afterwards (loops, what it was thinking).  STRATA_TEXT_LOG=<path> (default
# /tmp/strata-ds41-text.log), STRATA_TEXT_LOG=0 = off.
import threading as _th
import time as _time

_TEXT_LOG = os.environ.get("STRATA_TEXT_LOG", "/tmp/strata-ds41-text.log")
_text_lock = _th.Lock()
_text_state = {"kind": None}
_KIND_LABEL = {"reasoning": "thinking", "content": "answer", "tool_start": "tool call", "tool_call": "tool call"}

def _text_write(s: str):
    if not _TEXT_LOG or _TEXT_LOG == "0":
        return
    try:
        with _text_lock, open(_TEXT_LOG, "a", encoding="utf-8") as f:
            f.write(s)
    except OSError:
        pass


_svc_note, _svc_run = S.Service._note, S.Service.run


def _note_ds41(self, n, evs, st=None, rate=None):
    out = []
    for ev in evs:
        if not ev.text:
            continue
        label = _KIND_LABEL.get(ev.kind, ev.kind)
        if label != _text_state["kind"]:
            _text_state["kind"] = label
            out.append(f"\n--- {label} ---\n")
        out.append(ev.text)
    if out:
        _text_write("".join(out))
    return _svc_note(self, n, evs, st, rate)


def _run_ds41(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
    t0 = _time.time()
    _text_state["kind"] = None
    _text_write(f"\n\n===== {_time.strftime('%Y-%m-%d %H:%M:%S')}  request: prompt {len(ids):,} tokens, max {max_new}, "
                f"thinking {'on' if thinking else 'off'}, tools {len(tools or [])} =====\n")
    end = "(stream closed)"
    try:
        # force is the request's "ignore the cached prompt state" flag: Strata's own server has it, the engine wrapper
        # only forwards it (a V4.1 prompt that cannot be extended is re-prefilled by the engine either way)
        for item in _svc_run(self, ids, thinking, tools, max_new, sampling, cancel, force=force):
            if item[0] == "done" and isinstance(item[1], dict):
                d = item[1]
                end = f"finish {d.get('finish')}, {d.get('completion_tokens')} tokens"
            yield item
    finally:
        _text_write(f"\n===== end: {end}, {_time.time() - t0:.1f} s =====\n")


S.Service._note = _note_ds41
S.Service.run = _run_ds41

if __name__ == "__main__":
    for i, v in enumerate(sys.argv[:-1]):   # paths are the caller's; the server runs from Strata's directory
        if v in ("--config", "--tokenizer", "--mcp-config"):
            sys.argv[i + 1] = os.path.abspath(sys.argv[i + 1])
    os.chdir(str(STRATA))
    # the config's "env" reaches the engine only (child_env): its STRATA_TEXT_LOG (local/v41-text.log) is read here too,
    # relative to this repository, unless the server's own environment sets one
    if "STRATA_TEXT_LOG" not in os.environ:
        cfgs = [sys.argv[i + 1] for i, v in enumerate(sys.argv[:-1]) if v == "--config"]
        try:
            tl = (json.load(open(cfgs[-1])).get("env") or {}).get("STRATA_TEXT_LOG") if cfgs else None
        except (OSError, ValueError):
            tl = None
        if tl:
            _TEXT_LOG = tl if tl == "0" or os.path.isabs(tl) else str(STRATA / tl)
    sys.exit(S.main())
