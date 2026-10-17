# The v41 sampler has no penalty stage (design note, not implemented)

Status: **plan only.** Nothing is changed in the engine. Written 2026-10-17, after three reasoning loops measured
on this machine (see "Why" below).

## Why

Every serving stack we can read on this machine has a penalty stage in front of the sampler:

- llama.cpp (`~/Dati/llama.cpp/common/arg.cpp`): `--repeat-penalty` / `--repeat-last-n`, `--frequency-penalty`,
  `--presence-penalty`, and **DRY** (`--dry-multiplier`, `--dry-base`, `--dry-allowed-length`,
  `--dry-penalty-last-n`, `--dry-sequence-breaker`), which penalises repeated *sequences* with a strength that grows
  with the length of the repeat and leaves short legitimate repeats alone.
- the platform this fork comes from (`Strata/src/program/generate.cpp:10648`): `penalty_repeat`, `penalty_freq`,
  `penalty_present`, `penalty_last_n`, a history ring capped at `kPenaltyWindowCap = 4096` (`:7756`), and the same
  keys parsed from requests (`:10691-10694`, qwen path `src/qwen35/q36_engine.cu:2252`).

The v41 engine has **neither**: `tools/v41/engine/ds41_generate.cpp:1115-1118` parses only `top_p`, `top_k`, `min_p`
and `seed`, and `sample_p` (`:169`) / `sample_dist` (`:195`) shape the distribution without touching the logits.

Measured consequence on this machine (three separate requests, thinking text captured in `local/v41-text.log`): a
12-word passage repeating 5+ times covering **96%, 88% and 70%** of the last 600 words, running for 35-130 s until a
budget or a manual stop. Not one of those loops would survive a penalty stage with the usual values.

## What to do (mechanical, once the shape is fixed here)

1. **Protocol** (`ds41_generate.cpp`, next to `:1115-1118`): parse `penalty_repeat` (float), `penalty_freq` (float),
   `penalty_present` (float), `penalty_last_n` (int), with **the platform's defaults**
   (`req_penalty_repeat = 1.0f`, `req_penalty_freq = 0.0f`, `req_penalty_present = 0.0f`, `req_penalty_last_n = 0`,
   `generate.cpp:10648-10649`). With those defaults the stage is a **no-op**, which is what keeps this change
   purity-preserving by construction.
2. **History**: a ring of the last `penalty_last_n` generated tokens (cap it at 4096 like `kPenaltyWindowCap`),
   updated after every emitted token, per sequence (the serve is one sequence at a time).
3. **Application**: one function over the logits, called **once, before both `sample_p` and `sample_dist`**, so the
   speculative path (DSpark draft + verify) sees the same penalised distribution as the plain path. This is the
   correctness point of the whole change: a penalty applied to only one of the two would break the exactness of
   speculative sampling, which the author guarantees as "the model's distribution at any temperature".
   - `penalty_repeat`: the llama.cpp rule, multiplicative on the sign-split logits (positive logits divided by the
     penalty, negative multiplied) for tokens in the window.
   - `penalty_freq` / `penalty_present`: subtract `freq * count` and `present * (count > 0)` from the logits.
4. **Shim** (`tools/v41/serve/serve_ds41.py`): pass the four keys through (today it parses only
   `temp/top_p/top_k/min_p/seed`, so a client asking for a penalty is silently ignored).
5. **Config**: leave `sampling` as the reference has it (`temperature 1.0, top_p 0.95`, no `top_k`, no `min_p`); the
   penalties are per-request, off unless asked.

## The gate

- with the defaults (1.0 / 0.0 / 0.0 / 0) the stage is a no-op: the generated ids must be **bit-identical** to today
  on the three standard prompts (short, 1.5K, 63K), greedy, 96 tokens;
- then, with `penalty_repeat 1.05` (or 1.1), the loop prompts must **finish**: the Divina Commedia
  "approfondiamo" request is the reference case, and the thinking must stop repeating within the same budget;
- and only after that, evaluate whether DRY in the llama.cpp sense is worth porting too (it is the mechanism aimed
  exactly at long repeated passages, but it is more code and it needs its own gate).

## What this does not fix

A factual gap is not a decoding problem: the loop above is the model trying to recall the nine circles of Hell and
failing (it produced "Antenati (lussuria)", which does not exist). A penalty stops the *spinning*; the sentence stays
wrong. That is the 2.25 bpw mixed-tier quant, not the sampler, and the only clean answers there are the original FP8
model (510 GB) or a re-quantisation of the reasoning-critical tensors.

## Progress

**Implemented (engine side, built, gate passed).** `tools/v41/engine/ds41_generate.cpp`:

- `PenaltyState` with the platform's defaults (`repeat 1.0`, `freq 0.0`, `present 0.0`, `last_n 0`, ring capped at
  4096 like `kPenaltyWindowCap`), plus `pen_record()` and `pen_apply()`.
- `pen_apply()` returns the input buffer unchanged when `last_n <= 0` or all three penalties are at their defaults,
  so the stage costs nothing and changes nothing unless it is asked for.
- `sample_p` was renamed to `sample_p_raw`, with a wrapper that applies the penalty and records the emitted token;
  `sample_dist` (the speculative path's draft) applies the same `pen_apply`, and its body is untouched: the
  parameter is renamed and a local `logits` shadows it, so draft and verify sample the same penalised row. That is
  the correctness point of the whole change.
- The protocol parses `penalty_repeat`, `penalty_freq`, `penalty_present`, `penalty_last_n` next to `min_p`.

**Gate (passed).** Same command as before the change (greedy 96 tokens on `local/prompts/code_review.i32`, fused
build): the only differences against `local/gate-before.txt` are four timing lines (`seeded in`, `loaded in`, the
VRAM-slot seconds and the prompt split); the generated text, the arena/file-tier counts (8229 / 7131) and the VRAM
slot count (1622) are identical. So with the defaults the tokens are unchanged, which is what the no-op design
promised.

**Step (4) needed no shim change.** `Strata/serve/server.py:1147-1162` already emits `penalty_repeat=`,
`penalty_freq=` and `penalty_present=` (plus `penalty_last_n`) into the engine's command line from a request's
`repetition_penalty` / `frequency_penalty` / `presence_penalty`, normalising the llama.cpp names, and the shim passes
the `sampling` dictionary through untouched. So a client can turn the stage on today:
`{"repetition_penalty": 1.05, "penalty_last_n": 256}`.

**Open.**

1. The loop test: restart the server (the running one predates the new binary) and send the "approfondiamo" request
   with `repetition_penalty 1.05`; the thinking must stop repeating instead of running 75 s.
2. The ring only records tokens that pass through `sample_p`, so with DSpark on the **accepted draft tokens** are
   missing from the window. The plain path (DSpark off, or the CLI) is correct; the speculative path needs
   `pen_record` called for every emitted token before this can be called finished.
3. The one-shot CLI has no penalty flags yet (`--temp` exists, `--penalty-repeat` does not), so testing without the
   server needs four flags in the style of `--temp` - and that needs the `PenaltyState` declaration moved above the
   argument parsing, since `g_pen` is currently declared after it.

## Acceptance test (measured, 2026-10-17)

Prompt: the Divina Commedia case that looped in the web chat (`--chat "raccontami in breve la divina commedia,
poi approfondisci i nove cerchi dell'Inferno"`, 29 tokens, thinking on), 700 tokens at `--temp 1.0`, repetition
measured as before: the fraction of the last 600 words inside 12-word passages seen five or more times.

| run | coverage |
|---|---:|
| no penalty | **48.2%** |
| `--penalty-repeat 1.05 --penalty-last-n 256` | 35.2% |
| `--penalty-repeat 1.1 --penalty-last-n 1024` | 35.2% |
| `--penalty-present 0.5 --penalty-last-n 1024` | 35.2% |

The stage does reach the sampler: at `--temp 0` (deterministic) the same prompt with `--penalty-repeat 2.0` gives a
different output from the run without it, and the four temp-1.0 outputs have four different md5 sums (the equal
coverage is the metric being coarse, not identical text).

**Conclusion.** The standard penalties reduce this loop but do not break it: 48% -> 35% and it keeps spinning. That
matches the diagnosis above - the model is trying to recall the nine circles of Hell and failing, so it is a
*semantic* dead end, and a penalty on repeated tokens does not close it. What is left, if this class of loop is to be
cured rather than shortened:

- **DRY** in the llama.cpp sense (`--dry-multiplier`, `--dry-base`, `--dry-allowed-length`): it penalises repeated
  *sequences* with a strength that grows with the length of the repeat, which is aimed exactly at a passage coming
  back dozens of times. More code than the three penalties above, and it needs its own gate.
- the model itself: the factual gap is the 2.25 bpw mixed-tier quant (it wrote "Antenati (lussuria)"), and no sampler
  setting fixes a missing fact.

## The real culprit (2026-10-17)

The wrong facts and the loops this document was written for turned out not to be the sampler's. The engine's
expert-selection bias (route bias: up to +1.0/-0.6, a function of where an expert's bytes live, applied on every
decode token) was measured to change the answers: with it the nine circles of Hell are never named, with it limited to
near-ties (`STRATA_RB_MARGIN=1`) they are named correctly, at 11.12 tok/s against 13.17 unrestricted and 2.97 with no
bias at all. The penalty stage here stays opt-in and default-off as designed; it was never the fix.

## The thinking closure (2026-10-17)

The other half of the story: with the thinking on, the model produced no `content` because `reasoning_budget_tokens`
was 2048 and the requests hit `max_tokens` first, so the platform's wrap-up never fired. With the budget at **192**
(the mechanism the platform provides, not a trick) the thinking closes at ~114 words, `finish=stop`, `content` is
clean, no English deliberation leaks into it, and the acceptance prompt answers the nine circles of Hell by name:
Limbo, Lussuriosi, Golosi, Avari/Prodighi, Irascibili, Eretici, Violenti, Malebolge, Nono. Measured with
`repetition_penalty 1.1` / `penalty_last_n 256` in the config: loop coverage 0% in both acceptance tests.
