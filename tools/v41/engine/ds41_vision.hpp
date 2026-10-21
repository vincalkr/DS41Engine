// tools/v41/engine/ds41_vision.hpp - the DeepSeek-V4.1 image encoder (ViT + aligner) on a ggml backend.
//
// Math = DeepSeek's inference/vision.py (1017 repo): patch_embed Linear(3*14*14 -> 1024), 32 pre-RMSNorm blocks
// (attention with a 2-D RoPE over the patch grid, full bidirectional, 16 heads x 64; SwiGLU MLP 1024 -> 2x2816 ->
// 1024), a final RMSNorm, then the aligner: the patch grid padded to multiples of 3, 3x3 unfold (9216), Linear ->
// exact GELU -> Linear (5120).  The span the language model reads is model.py's merge_image_embeddings:
//     [image_start] + ([aligner row] * llm_w + [image_newline]) * llm_h + [image_end]
// Weights: the vision GGUF (tools/v41/vision/make_vision_gguf.py; BF16 as in the checkpoint), matrices held as F16
// on the device, vectors as F32.  The input is image_processor.load_image's patch grid (the server computes it).
//
// The encoder holds device memory only between load() and unload(): ds41_generate keeps it resident on a second
// card (--vision-device) or loads it for an image request and frees it again on the main card.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

struct Ds41VisionGrid {
    int vit_h = 0, vit_w = 0;   ///< the patch grid (image_processor: best_height / 14, best_width / 14)
    int llm_h = 0, llm_w = 0;   ///< the token grid after the 3x3 aligner (ceil(vit / 3))
    int patches() const { return vit_h * vit_w; }
    int span_rows() const { return llm_h * (llm_w + 1) + 2; }
};

class Ds41Vision {
public:
    Ds41Vision();
    ~Ds41Vision();
    Ds41Vision(const Ds41Vision &) = delete;
    Ds41Vision & operator=(const Ds41Vision &) = delete;

    /// Map the GGUF and check its geometry.  No device memory.
    bool open(const std::string & gguf, std::string & err);
    /// The weights onto `be` (F16 matrices, F32 vectors; ~0.90 GiB).  A no-op when already loaded there.
    bool load(ggml_backend_t be, std::string & err);
    /// Free the device weights and the compute buffer.
    void unload();
    bool loaded() const;
    /// One image: `patches` [g.patches(), 588] float32 in image_processor's order -> `span` [g.span_rows(), 5120].
    bool encode(const float * patches, const Ds41VisionGrid & g, std::vector<float> & span, std::string & err);
    int64_t n_embd() const;
    int image_token() const;            ///< the model's image token id (129264)
    size_t weight_bytes() const;        ///< device bytes of the loaded weights
    size_t compute_bytes() const;       ///< device bytes of the last encode's compute buffer
    double last_ms() const;             ///< wall time of the last encode (upload + compute + readback)

    struct Impl;
private:
    std::unique_ptr<Impl> p_;
};
