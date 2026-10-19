# third_party/llama.cpp - the ggml this engine is built and measured with

Only `ggml/` (and llama.cpp's `LICENSE`, MIT) is kept here: the rest of llama.cpp is not needed. This is ggml 0.26.0
as it was on the machine where every number in `docs/ds4/` was measured, with this repository's ggml changes already
applied (`tools/v41/ggml/*.patch`: the fused DeepSeek-V4.1 ops `GGML_OP_DSV41_FAKE_QUANT` / `GGML_OP_DSV41_HC_MIXES`,
the FP4 KV cache, the deterministic MMQ, the rank argsort, the upstream fixes in `UPSTREAM_FIXES.md`). It was copied in
without git history and does not match one llama.cpp commit (a diff against 3cf03257, the commit
`third_party/ggml/VERSION.txt` names, differs in ~370 files), so it is kept whole instead of as a commit plus patches.
Do not re-apply the patches to it.

Build it on its own (the layout `STRATA_GGML_PREBUILT` expects: `build-hip/ggml/src/libggml*.a`):

    cmake -S third_party/llama.cpp/ggml -B third_party/llama.cpp/build-hip/ggml ...   (see the README, "Build")
