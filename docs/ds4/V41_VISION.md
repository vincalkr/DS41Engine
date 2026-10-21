# V4.1 image input

Images need the vision GGUF. With it in `models/vision-1017`, `./run.sh` starts the vision configuration
(`local/ds41-local-vision.json`, two cards; `--gpu 0` or `--gpu 1`: `local/ds41-local-1gpu-vision.json`); without
it, the text-only ones (`local/ds41-local.json`, `local/ds41-local-1gpu.json`). Text runs as fast either way (see
"Text-only speed": 21.01 / 19.60 tok/s with the vision configuration, 20.39 / 19.77 without).

```sh
VISION=1 local/fetch-model.sh                          # or: python3 tools/v41/vision/make_vision_gguf.py
./run.sh                                               # two cards: the encoder on card 1
./run.sh --gpu 0                                       # one card: the encoder on card 0, for the request
python3 tools/v41/vision/check_images.py picture.jpg   # ask the running server about a picture
```

OpenAI `image_url` parts (data: URLs, http(s) URLs, local paths) and Anthropic `image` blocks go through the server's
existing image handling. Measured on two RX 7900 XTX, ROCm 7.14, 2026-10-21.

## The weights: 1017, not September

The 1017 checkpoint keeps the vision tower in BF16: shard 1 of `deepseek-ai/DeepSeek-V4.1-Flash-1017` holds exactly
the 263 ViT + aligner tensors (0.90 GiB), shard 2 the three span delimiters (`image_start`, `image_newline`,
`image_end`). antirez's `DeepSeek-V4.1-Flash-Vision.gguf` is the September tower (`general.source.revision`
df42c109; three tensors checked byte-equal against that revision's shard 1).

- 1017 against September: 0 of 266 tensors equal; per tensor cosine 0.995-0.99999, relative L2 difference 0.3-10%
  (aligner.w1 9%, blocks.0.attn.wo 9.3%, blocks.0.attn.wqkv 3.4%). The tower and the aligner were retrained.
- At the output, on DeepSeek's example images (the same ViT code for both weight sets, float32 numpy): the image
  rows' cosine September vs 1017 is 0.888 mean / 0.575 5th percentile / 0.377 min on `corn.jpeg`, 0.914 / 0.719 /
  0.481 on `carrots.jpeg`. Two different 1017 rows of the same picture have cosine ~0.46-0.47, so the worst
  September rows are as far from the right row as an unrelated patch. The delimiters barely moved (0.999).

`tools/v41/vision/make_vision_gguf.py` copies the 1017 bytes into a GGUF with antirez's tensor names and metadata keys
(HTTP range reads: the two headers, shard 1's data, 30 KB of shard 2): `models/vision-1017/
DeepSeek-V4.1-Flash-1017-Vision.gguf`, 970,555,584 bytes, every tensor byte-identical to the checkpoint.

## What the model expects (DeepSeek's inference code, identical in September and 1017)

- Preprocessing (`image_processor.py`): RGB; upscaled to 544 x 544 pixels' area if smaller; rounded to multiples of
  14; shrunk until `llm_h * (llm_w + 1) + 2 <= 1024`; padded to that size with gray 127 (bicubic, centred);
  `(x / 255 - 0.5) / 0.5` rounded to bfloat16; patches `[vit_h * vit_w, 3 * 14 * 14]`. `llm = ceil(vit / 3)`.
- ViT (`vision.py`): patch embedding 588 -> 1024; 32 pre-RMSNorm blocks, attention 16 x 64 with a 2-D RoPE over the
  patch grid (row and column angles, rotate-half), full bidirectional; SwiGLU MLP 1024 -> 2 x 2816 -> 1024; final
  RMSNorm. Aligner: the grid padded to multiples of 3, 3 x 3 unfold (9216), Linear -> exact GELU -> Linear (5120).
- The span (`model.py` merge_image_embeddings): `[image_start] + ([row] * llm_w + [image_newline]) * llm_h +
  [image_end]`, every position carrying token 129264 (`<｜deepseek_image｜>`) with its embedding replaced. Plain 1-D
  positions.
- Image positions route with `exp_probs_b_vl` instead of `exp_probs_b` (the Q2 GGUF has it for all 40 layers).
- Engram: an image position is DEAD in the n-gram history (no n-gram reaches across it) and its Engram gate is 0.
- antirez/ds4 implements the same four points (`ds41_image_at`, `ds41_text_mask`); its V4.1 vision runs on Metal only.

## How it is built here

- Server (`tools/v41/serve/serve_ds41.py`): DeepSeek's encoder writes one placeholder per image block; each
  placeholder becomes its image's span of 129264 tokens. The patches come from `tools/v41/vision/ds41_image.py`, a
  torch-free port of `load_image` (bit-identical patches on the five reference images below), written as one `SVP1`
  record per image (cached by the image bytes). The server's existing image path does the rest: the request FIFO,
  one combined file, the `GENI` line.
- Engine (`ds41_generate --vision F`): `GENI <max_new> [k=v ..] <file> <ids>`. The encoder (`ds41_vision.cpp`, ggml
  on HIP) turns each record into its span rows; up to 8 spans are kept by image (`STRATA_VISION_CACHE`). The rows
  replace the embeddings at their positions in any pass that feeds them (`Ds41Dense::set_vision`); only a span that
  holds image rows gets the VL routing bias and the Engram gate mask in its graph, and the Engram history marks those
  positions DEAD. A prompt with images runs as the layer-major batch.
- Where the encoder runs: with `--peer-device 1` on card 1 (its own ggml backend there), loaded at the first image
  and kept (the peer store is built at the first decode from that card's free memory, so it never sees the encoder);
  with one card on card 0 through the dense half's backend, after the prompt has closed the VRAM expert cache, its
  weights and compute buffer freed again before the prefill. A second ggml context on card 0, freed after the
  encode, crashed the engine (segfault in `~ggml_backend_cuda_context`, 2026-10-21), hence the shared backend.
  Before encoding it checks the card's free memory (weights 0.97 GB if not loaded + ~100 MB + 24 KB a patch +
  256 MiB for hipBLAS) and refuses the image with an error instead: a failed allocation inside a kernel launch would
  end the engine.
- Prompt reuse: positions also carry the image's key, so a turn that resends the same picture reuses the prefix, and
  a different picture with the same grid does not. A conversation with images is never parked.

## Encoder accuracy and speed

`build-hip-fused/test_ds41_vision` against DeepSeek's `vision.py` itself (torch 2.14 CPU, float32, weights from the
same GGUF), card 1 with the 22 GiB peer store in place:

| image | patches | span rows | min / mean row cosine | rel L2 | encode |
| --- | ---: | ---: | --- | ---: | ---: |
| red square 200 x 150 | 34 x 45 | 194 | 1.000000 / 1.000000 | 1.1e-5 | 169 ms |
| corn.jpeg (DeepSeek's example) | 33 x 47 | 189 | 1.000000 / 1.000000 | 5.9e-6 | 167 ms |
| text, 300 x 1200 | 86 x 22 | 263 | 0.999994 / 1.000000 | 1.5e-4 | 208 ms |
| carrots.jpeg (DeepSeek's example) | 51 x 74 | 444 | 0.999997 / 1.000000 | 9.5e-5 | 732 ms |
| blobs 2000 x 1500 (the 1024-token cap) | 81 x 108 | 1001 | 0.999999 / 1.000000 | 3.6e-5 | 3375 ms |

Matrices F16 on the card, F32 accumulation (hipBLAS converts each F16 weight to F32 for the product; the aligner's
two big matrices go in blocks of 1280 rows to keep that temporary small). The attention is exact F32: scores in
blocks of query rows (64 MiB each, `DS41_VISION_KQ_MIB`), each block written into its columns in place. Two faster
variants were measured and are not used:

| variant | corn.jpeg min cosine | blobs min cosine | blobs encode |
| --- | ---: | ---: | ---: |
| exact (default) | 1.000000 | 0.999999 | 3.1-3.4 s (in the server 3.5-4.9 s) |
| `DS41_VISION_ATTN=flash` (ggml flash-attn, F16 Q/K/V) | 0.986548 | 0.624423 | 1.25 s |
| `DS41_VISION_PREC=f16` (F16 matmuls, F16 accumulation on RDNA3) | 0.858170 | - | - |

This ViT does not take F16 attention inputs (rel L2 2.9e-2 on corn, 7.3e-2 on blobs).

Card 1 memory: the encoder's weights 0.90 GiB + compute buffer 119-312 MiB (189 to 1001 rows). In the server, card 1
had 1,836 MiB free before the first image and 750 MiB with the weights resident; the 1001-row image ran there.

## Through the server

`./run.sh` (`local/ds41-local-vision.json`), temperature 0, thinking off, each request after a text request
(the peer store built), the question "Describe this image in two or three sentences.":

| image | prompt tokens | encode | prompt | request | answer (start) |
| --- | ---: | ---: | ---: | ---: | --- |
| red square (first image: weights load 0.7 s) | 207 | 2.47 s | 5.84 s | 15.2 s | "a solid red rectangle centered on a plain white background" |
| carrots.jpeg | 457 | 1.80 s | 6.82 s | 16.6 s | "a small pile of fresh orange carrots resting against a plain white background" |
| corn.jpeg | 202 | 0.22 s | 3.84 s | 8.9 s | "three ears of corn ... One ear is partially shucked, revealing bright yellow kernels" |
| blobs (1024-token cap) | 1014 | 3.46 s | 8.72 s | 21.4 s | "colorful, overlapping oval shapes set against a solid black background" |
| text 300 x 1200, "transcribe" | 284 | 0.26 s | 5.13 s | 15.5 s | all 12 lines exact ("Riga 1: ciao mondo 0" ... "Riga 12: ciao mondo 77") |

- Two images in one request (red square + corn, from the cache): both described, in order; 399 tokens, 10.5 s.
- A follow-up turn resending the same picture: 220 of 232 prompt tokens reused, 1.7 s.
- A different picture with the same grid (a blue circle, 200 x 150): reuse stops at the first image position
  (lcp 15), re-encoded, "The image shows a blue circle."

No reference answer exists for the language model here (ds4's V4.1 vision is Metal-only, the FP8 model does not fit
torch on this machine): the check is the encoder's parity above and the answers.

## Text-only speed

`local/bench-v41/serve-bench.py` (3 short answers, 3 answers after the 22K document; temperature 0, thinking off),
power-tune on:

A/B of the engine binary, the default configuration (no `--vision`): a fresh server per block, the blocks
alternated, 5 requests each; power-tune switched on once and checked on at the start and the end of every block
(an earlier A/B was discarded: run.sh's power-tune off from the previous server raced the next server's on).

| block | engine | short answers, tok/s | after 25.5K tokens, tok/s |
| --- | --- | --- | --- |
| 1 | before (e346e660) | 20.29 (17.50, 20.50, 21.30, 20.34, 21.81) | 19.90 (19.46, 20.14, 19.83, 19.52, 20.53) |
| 2 | with images | 21.17 (18.23, 21.18, 21.92, 22.29, 22.25) | 19.50 (20.14, 19.79, 18.01, 20.20, 19.37) |
| 3 | before | 20.48 (19.45, 21.54, 21.18, 21.35, 18.86) | 19.63 (19.60, 18.98, 19.77, 19.93, 19.89) |
| 4 | with images | 20.72 (19.49, 20.30, 22.15, 20.26, 21.40) | 20.05 (19.65, 19.61, 20.77, 20.43, 19.78) |
| | before / with images | 20.39 / 20.95 | 19.77 / 19.78 |

The same within the run-to-run spread: without `--vision` the new code paths build no extra graph node, allocate
nothing and only test an empty vector per pass.

`local/ds41-local-vision.json` (`--vision`), the same conditions, after four image requests (the encoder's weights
resident on card 1): short answers 21.01 tok/s (18.76, 20.91, 21.62, 21.94, 21.83), after 25.5K tokens 19.60
(19.48, 19.09, 19.79, 19.87, 19.76). Card 1's peer store is fixed (22 GiB) before the first image, so the resident
encoder takes nothing from the decode.

## One card

`./run.sh --gpu 0` (`local/ds41-local-1gpu-vision.json`), power-tune on: carrots, blobs and corn described as with
two cards (encode 1.8 / 4.9 / 0.23 s, each request loading the weights in 0.66-0.69 s and freeing them). The VRAM
expert cache re-seeded after each image request at 1668, 1665 and 1646 experts (15.96 / 15.94 / 15.76 GiB free): the
dense backend's hipBLAS pool keeps up to ~200 MB of the encoder's temporaries. Decode of the same 300-token answer
before the images 16.23 / 20.55 tok/s (first / second request), after them 17.57 / 19.68 - within the spread, but
the ~1.3% fewer cached experts after images are real; nothing changes before the first image.

## Files

- `tools/v41/vision/make_vision_gguf.py`, `ds41_image.py`, `check_images.py`
- `tools/v41/engine/ds41_vision.{hpp,cpp}`, `test_ds41_vision.cpp`; `ds41_dense`, `ds41_engram`, `ds41_generate`
  (`--vision`, `--vision-device`, `GENI`)
- `tools/v41/serve/serve_ds41.py` (images), `serve/server.py` (`Service.expand_image_tokens`)
- `local/ds41-local-vision.json`, `local/ds41-local-1gpu-vision.json`
