# ds41e

DeepSeek-V4.1-Flash (552B-parameter MoE, 384 experts per layer) on one RX 7900 XTX: the experts are spread over VRAM,
pinned RAM and NVMe, the routing is exact (the model's own expert choice), and an OpenAI- and Anthropic-compatible
server runs on `127.0.0.1:8150`. Linux, ROCm/HIP.

Measured on this PC (RX 7900 XTX, Ryzen 9 9900X, 122 GiB RAM, PCIe 5.0 NVMe), through the server:

| | |
|---|---:|
| decode, short chat prompts | 16-19 tok/s |
| decode, agent session at 25K context | 11-14 tok/s |
| prompt reading, 25K tokens | ~370 tok/s |

Details and every measurement: [docs/ds4/V41_EXACT_FAST.md](docs/ds4/V41_EXACT_FAST.md).

## Start

```sh
./run-v41-gsq.sh          # server on http://127.0.0.1:8150 (web chat at /, API at /v1), config: local/ds41-local.json
```

Python needs pillow, psutil, regex and jinja2 (`python3 -m venv .venv && .venv/bin/pip install pillow psutil regex jinja2`).
Never expose the server beyond `127.0.0.1` without `--api-key`.

## Model files

`local/fetch-model.sh` downloads them into `models/` (about 355 GB: the GSQ quant from
[IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored)
and the two Engram tables), checking every file. Keep them in `models/`, inside this folder.

## Build

ggml once, from `third_party/llama.cpp` (a llama.cpp checkout, not tracked here, with the patches in
`tools/v41/ggml/`), then the engine:

```sh
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-hip -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=OFF -DGPU_TARGETS=gfx1100
cmake --build third_party/llama.cpp/build-hip --target ggml -j

cmake -S . -B build-hip-fused -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_GGML_PREBUILT=$PWD/third_party/llama.cpp/build-hip \
  -DSTRATA_DSV41_HC_MIX=ON -DSTRATA_DSV41_FQ_OP=ON
cmake --build build-hip-fused -j
```

Server tests (no GPU or model needed): `.venv/bin/python -m unittest discover -s serve -p "test_*.py" -t .`

## Layout

| | |
|---|---|
| `tools/v41/engine/` | the engine (`ds41_generate`) |
| `src/`, `include/` | the parts of Strata the engine links (expert cache, kernels) |
| `serve/`, `tools/v41/serve/` | the server and web app |
| `local/` | configurations, ranks, benchmark scripts and their results |
| `docs/ds4/` | measurements and notes; [AMD_BRANCH_NOTES.md](docs/ds4/AMD_BRANCH_NOTES.md) is the longer earlier README |
| [HANDOVER.md](HANDOVER.md) | the state of the work and how it got here |

The folder must not be renamed to `Strata*`: another Strata install's setup moves the `models/` of such folders.

## Credits

Built on [Indras-Mirror/Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1) (now Xolotl) and
[Niko1221/Strata](https://github.com/Niko1221/Strata); their README is in [docs/upstream/](docs/upstream/README.md).
MIT, see [LICENSE](LICENSE).
