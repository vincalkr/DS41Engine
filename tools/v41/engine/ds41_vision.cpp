// tools/v41/engine/ds41_vision.cpp - see ds41_vision.hpp.
#include "ds41_vision.hpp"

#include "ggml-alloc.h"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

namespace {

constexpr int64_t kDim = 1024, kHeads = 16, kHeadDim = 64, kFF = 2816, kLayers = 32, kPatch = 14, kRatio = 3;
constexpr int64_t kPatchIn = 3 * kPatch * kPatch;   // 588
constexpr int64_t kEmbd = 5120;
constexpr int64_t kAlignIn = kDim * kRatio * kRatio;   // 9216
constexpr float kEps = 1e-6f, kTheta = 10000.0f;

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// DS41_VISION_PREC=f16 (opt-in): the matmuls in the backend's fast F16 mode (on RDNA3 hipBLAS accumulates and writes
// F16).  Default f32: F16 weights, F32 accumulation.
bool prec_f16() {
    static const bool v = std::getenv("DS41_VISION_PREC") && std::strcmp(std::getenv("DS41_VISION_PREC"), "f16") == 0;
    return v;
}

// DS41_VISION_ATTN=flash (opt-in): ggml's flash-attn, which reads Q/K/V as F16.  This ViT does not take that: on
// DeepSeek's corn.jpeg the aligner rows end at cosine >= 0.9865 / rel L2 2.9e-2 against vision.py in F32, while the
// default exact attention (F32 scores in blocks of query rows) ends at cosine 1.000000 / 2.1e-5 (2026-10-21).
bool flash_attn() {
    static const bool v = std::getenv("DS41_VISION_ATTN") && std::strcmp(std::getenv("DS41_VISION_ATTN"), "flash") == 0;
    return v;
}

// the exact attention's score block (DS41_VISION_KQ_MIB, default 64 MiB): N x nq x 16 F32
int64_t kq_bytes() {
    static const int64_t v = (int64_t) (std::getenv("DS41_VISION_KQ_MIB") ? std::atof(std::getenv("DS41_VISION_KQ_MIB")) : 64.0) * 1048576;
    return std::max<int64_t>(v, 1048576);
}

// the largest matrices (aligner) are multiplied in row blocks: with F32 accumulation hipBLAS converts the F16 weight
// to F32 first (a temporary of the whole matrix: 189 MB for aligner.w1), so a block keeps that temporary small
constexpr int64_t kRowBlock = 1280;
// the per-token part of a block (output projection, MLP) runs in blocks of this many tokens
constexpr int64_t kTokBlock = 1024;

}  // namespace

struct Ds41Vision::Impl {
    std::unique_ptr<strata::GgufFile> file;
    int image_token = 129264;
    std::vector<float> start, newline, end;   // the span delimiters (host, F32)
    ggml_backend_t be = nullptr;              // not owned
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor *> w;
    ggml_gallocr_t allo = nullptr;
    size_t cbytes = 0;
    double ms = 0;

    ~Impl() { free_dev(); }
    void free_dev() {
        if (allo) { ggml_gallocr_free(allo); allo = nullptr; }
        if (wbuf) { ggml_backend_buffer_free(wbuf); wbuf = nullptr; }
        if (wctx) { ggml_free(wctx); wctx = nullptr; }
        w.clear();
        be = nullptr;
        cbytes = 0;
    }
    const strata::TensorInfo * ti(const std::string & n) const { return file->find(n); }
    ggml_tensor * W(const std::string & n) const { return w.at(n); }
    ggml_tensor * B(int64_t l, const char * s) const { return w.at("vision.blocks." + std::to_string(l) + "." + s); }
};

Ds41Vision::Ds41Vision() : p_(new Impl) {}
Ds41Vision::~Ds41Vision() = default;
bool Ds41Vision::loaded() const { return p_->wbuf != nullptr; }
int64_t Ds41Vision::n_embd() const { return kEmbd; }
int Ds41Vision::image_token() const { return p_->image_token; }
size_t Ds41Vision::weight_bytes() const { return p_->wbuf ? ggml_backend_buffer_get_size(p_->wbuf) : 0; }
size_t Ds41Vision::compute_bytes() const { return p_->cbytes; }
double Ds41Vision::last_ms() const { return p_->ms; }

static void bf16_to_f32(const uint8_t * src, float * dst, int64_t n) {
    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(src), dst, n);
}

bool Ds41Vision::open(const std::string & path, std::string & err) {
    Impl & im = *p_;
    try {
        im.file.reset(new strata::GgufFile(path));
    } catch (const std::exception & e) {
        err = std::string("vision: ") + e.what();
        return false;
    }
    auto num = [&](const char * k, double dflt) { const strata::MetaValue * v = im.file->get(k); return v && v->is_num() ? v->num() : dflt; };
    const struct { const char * k; double want; } geo[] = {
        { "deepseek4-vision.block_count", kLayers }, { "deepseek4-vision.embedding_length", kDim },
        { "deepseek4-vision.feed_forward_length", kFF }, { "deepseek4-vision.attention.head_count", kHeads },
        { "deepseek4-vision.projection_length", kEmbd }, { "deepseek4-vision.patch_size", kPatch },
        { "deepseek4-vision.downsample_ratio", kRatio },
    };
    for (const auto & g : geo)
        if (num(g.k, -1) != g.want) {
            err = "vision: " + path + ": " + g.k + " is not " + std::to_string((int) g.want) + " (not a DeepSeek-V4.1 vision GGUF)";
            return false;
        }
    im.image_token = (int) num("deepseek4-vision.image.token_id", 129264);
    // every tensor the graph reads, BF16 / F32 / F16, with the checkpoint's shapes
    auto need = [&](const std::string & n, std::vector<uint64_t> shape) {
        const strata::TensorInfo * t = im.ti(n);
        if (!t) { err = "vision: " + path + ": missing " + n; return false; }
        if (t->shape != shape || (t->type != 30 && t->type != 0 && t->type != 1)) {
            err = "vision: " + path + ": " + n + " has an unexpected shape or type";
            return false;
        }
        return true;
    };
    if (!need("vision.patch_embed.proj.weight", { (uint64_t) kPatchIn, (uint64_t) kDim }) ||
        !need("vision.patch_embed.proj.bias", { (uint64_t) kDim }) || !need("vision.norm.weight", { (uint64_t) kDim }) ||
        !need("aligner.w1.weight", { (uint64_t) kAlignIn, (uint64_t) kEmbd }) || !need("aligner.w1.bias", { (uint64_t) kEmbd }) ||
        !need("aligner.w2.weight", { (uint64_t) kEmbd, (uint64_t) kEmbd }) || !need("aligner.w2.bias", { (uint64_t) kEmbd }) ||
        !need("image_start", { (uint64_t) kEmbd }) || !need("image_newline", { (uint64_t) kEmbd }) ||
        !need("image_end", { (uint64_t) kEmbd }))
        return false;
    for (int64_t l = 0; l < kLayers; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        if (!need(p + "norm1.weight", { (uint64_t) kDim }) || !need(p + "norm2.weight", { (uint64_t) kDim }) ||
            !need(p + "attn.wqkv.weight", { (uint64_t) kDim, (uint64_t) (3 * kDim) }) ||
            !need(p + "attn.wqkv.bias", { (uint64_t) (3 * kDim) }) ||
            !need(p + "attn.wo.weight", { (uint64_t) kDim, (uint64_t) kDim }) || !need(p + "attn.wo.bias", { (uint64_t) kDim }) ||
            !need(p + "mlp.w1.weight", { (uint64_t) kDim, (uint64_t) (2 * kFF) }) ||
            !need(p + "mlp.w2.weight", { (uint64_t) kFF, (uint64_t) kDim }))
            return false;
    }
    auto host = [&](const char * n, std::vector<float> & v) {
        const strata::TensorInfo * t = im.ti(n);
        v.resize(kEmbd);
        if (t->type == 30) bf16_to_f32(im.file->tensor_data(*t), v.data(), kEmbd);
        else if (t->type == 0) std::memcpy(v.data(), im.file->tensor_data(*t), kEmbd * 4);
        else for (int64_t i = 0; i < kEmbd; ++i) v[(size_t) i] = ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(im.file->tensor_data(*t))[i]);
    };
    host("image_start", im.start);
    host("image_newline", im.newline);
    host("image_end", im.end);
    return true;
}

bool Ds41Vision::load(ggml_backend_t be, std::string & err) {
    Impl & im = *p_;
    if (!im.file) { err = "vision: not opened"; return false; }
    if (im.wbuf && im.be == be) return true;
    im.free_dev();
    const double t0 = now_ms();
    // the device tensors: every matrix F16, every vector F32 (as ggml's mul_mat / add want them)
    std::vector<std::pair<std::string, const strata::TensorInfo *>> list;
    for (const auto & t : im.file->tensors())
        if (t.name.rfind("vision.", 0) == 0 || t.name.rfind("aligner.", 0) == 0) list.emplace_back(t.name, &t);
    ggml_init_params ip = { ggml_tensor_overhead() * (list.size() + 8), nullptr, true };
    im.wctx = ggml_init(ip);
    for (const auto & [name, t] : list) {
        ggml_tensor * x = t->shape.size() == 2
            ? ggml_new_tensor_2d(im.wctx, GGML_TYPE_F16, (int64_t) t->shape[0], (int64_t) t->shape[1])
            : ggml_new_tensor_1d(im.wctx, GGML_TYPE_F32, (int64_t) t->shape[0]);
        ggml_set_name(x, name.c_str());
        im.w[name] = x;
    }
    im.wbuf = ggml_backend_alloc_ctx_tensors(im.wctx, be);
    if (!im.wbuf) { err = "vision: cannot allocate the weights on the device"; im.free_dev(); return false; }
    im.be = be;
    // convert (BF16 -> F32 -> F16) in parallel, upload one tensor at a time
    auto convert = [&](const strata::TensorInfo * t, ggml_tensor * dst, std::vector<uint8_t> & out, std::vector<float> & f32) {
        const int64_t n = (int64_t) t->elements();
        f32.resize((size_t) n);
        const uint8_t * src = im.file->tensor_data(*t);
        if (t->type == 30) bf16_to_f32(src, f32.data(), n);
        else if (t->type == 0) std::memcpy(f32.data(), src, (size_t) n * 4);
        else for (int64_t i = 0; i < n; ++i) f32[(size_t) i] = ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(src)[i]);
        if (t->name == "aligner.w1.weight") {
            // the 3x3 unfold's feature order: F.unfold gives c*9 + ky*3 + kx, the graph's reshape c + 1024*kx + 3072*ky
            // (encode, "aligner"), so the input columns are permuted once here
            std::vector<float> row((size_t) kAlignIn);
            for (int64_t o = 0; o < kEmbd; ++o) {
                float * r = f32.data() + (size_t) (o * kAlignIn);
                for (int64_t c = 0; c < kDim; ++c)
                    for (int64_t ky = 0; ky < kRatio; ++ky)
                        for (int64_t kx = 0; kx < kRatio; ++kx)
                            row[(size_t) (c + kDim * kx + kDim * kRatio * ky)] = r[(size_t) (c * 9 + ky * 3 + kx)];
                std::memcpy(r, row.data(), (size_t) kAlignIn * 4);
            }
        }
        if (dst->type == GGML_TYPE_F16) {
            out.resize((size_t) n * 2);
            ggml_fp32_to_fp16_row(f32.data(), reinterpret_cast<ggml_fp16_t *>(out.data()), n);
        } else {
            out.resize((size_t) n * 4);
            std::memcpy(out.data(), f32.data(), (size_t) n * 4);
        }
    };
    const unsigned nth = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::vector<uint8_t>> outs(list.size());
    std::vector<std::thread> th;
    for (unsigned k = 0; k < nth; ++k)
        th.emplace_back([&, k]() {
            std::vector<float> f32;
            for (size_t i = k; i < list.size(); i += nth) convert(list[i].second, im.w[list[i].first], outs[i], f32);
        });
    for (auto & t : th) t.join();
    for (size_t i = 0; i < list.size(); ++i) ggml_backend_tensor_set(im.w[list[i].first], outs[i].data(), 0, outs[i].size());
    ggml_backend_synchronize(be);
    std::fprintf(stderr, "vision: %.2f GiB of weights on %s in %.0f ms\n", ggml_backend_buffer_get_size(im.wbuf) / 1073741824.0,
                 ggml_backend_name(be), now_ms() - t0);
    return true;
}

void Ds41Vision::unload() { p_->free_dev(); }

bool Ds41Vision::encode(const float * patches, const Ds41VisionGrid & g, std::vector<float> & span, std::string & err) {
    Impl & im = *p_;
    if (!im.wbuf) { err = "vision: weights not loaded"; return false; }
    const int64_t vh = g.vit_h, vw = g.vit_w, N = (int64_t) vh * vw;
    const int64_t lh = (vh + kRatio - 1) / kRatio, lw = (vw + kRatio - 1) / kRatio;
    if (vh < 1 || vw < 1 || lh != g.llm_h || lw != g.llm_w || g.span_rows() > 4096) { err = "vision: bad patch grid"; return false; }
    const double t0 = now_ms();
    const bool f16 = prec_f16();

    const int64_t nq = std::max<int64_t>(1, std::min<int64_t>(N, kq_bytes() / (N * kHeads * 4)));
    const int64_t nodes = 2048 + kLayers * (64 + 12 * ((N + nq - 1) / nq) + 24 * ((N + kTokBlock - 1) / kTokBlock));
    std::vector<uint8_t> meta(ggml_tensor_overhead() * (size_t) nodes + ggml_graph_overhead_custom((size_t) nodes, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context * gc = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gc, (size_t) nodes, false);
    auto mm = [&](ggml_tensor * wt, ggml_tensor * x) {
        ggml_tensor * y = ggml_mul_mat(gc, wt, x);
        if (!f16) ggml_prec_set_acc(y, GGML_PREC_F32);
        return y;
    };
    auto rms = [&](ggml_tensor * x, ggml_tensor * wt) { return ggml_mul(gc, ggml_rms_norm(gc, x, kEps), wt); };
    auto mm_blocks = [&](ggml_tensor * wt, ggml_tensor * x) {   // mm in blocks of kRowBlock output rows
        ggml_tensor * y = nullptr;
        for (int64_t r0 = 0; r0 < wt->ne[1]; r0 += kRowBlock) {
            const int64_t m = std::min(kRowBlock, wt->ne[1] - r0);
            ggml_tensor * part = mm(ggml_view_2d(gc, wt, wt->ne[0], m, wt->nb[1], (size_t) r0 * wt->nb[1]), x);
            y = y ? ggml_concat(gc, y, part, 0) : part;
        }
        return y;
    };

    ggml_tensor * in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, kPatchIn, N);
    ggml_set_input(in);
    // the 2-D RoPE tables (vision.py get_vision_cos_sin): per patch 32 angles = row * inv_freq[16] | col * inv_freq[16]
    const int64_t RD = kHeadDim / 2;   // 32
    ggml_tensor * cs = ggml_new_tensor_3d(gc, GGML_TYPE_F32, RD, 1, N);
    ggml_tensor * sn = ggml_new_tensor_3d(gc, GGML_TYPE_F32, RD, 1, N);
    ggml_set_input(cs);
    ggml_set_input(sn);

    ggml_tensor * x = ggml_add(gc, mm(im.W("vision.patch_embed.proj.weight"), in), im.W("vision.patch_embed.proj.bias"));
    auto rope = [&](ggml_tensor * t) {   // t [64, 16, N]: cat(x1 cos - x2 sin, x2 cos + x1 sin) over the two halves
        ggml_tensor * x1 = ggml_view_3d(gc, t, RD, kHeads, N, t->nb[1], t->nb[2], 0);
        ggml_tensor * x2 = ggml_view_3d(gc, t, RD, kHeads, N, t->nb[1], t->nb[2], (size_t) RD * ggml_element_size(t));
        ggml_tensor * a = ggml_sub(gc, ggml_mul(gc, x1, cs), ggml_mul(gc, x2, sn));
        ggml_tensor * b = ggml_add(gc, ggml_mul(gc, x2, cs), ggml_mul(gc, x1, sn));
        return ggml_concat(gc, a, b, 0);
    };
    for (int64_t l = 0; l < kLayers; ++l) {
        ggml_tensor * h = rms(x, im.B(l, "norm1.weight"));
        ggml_tensor * qkv = ggml_add(gc, mm(im.B(l, "attn.wqkv.weight"), h), im.B(l, "attn.wqkv.bias"));   // [3072, N]
        auto part = [&](int k) { return ggml_view_3d(gc, qkv, kHeadDim, kHeads, N, (size_t) kHeadDim * 4, qkv->nb[1], (size_t) k * kDim * 4); };
        ggml_tensor * o = nullptr;
        if (flash_attn()) {   // opt-in: one flash-attn (F16 Q/K/V inside; see the note at flash_attn())
            ggml_tensor * q = ggml_permute(gc, rope(part(0)), 0, 2, 1, 3);                                  // [64, N, 16]
            ggml_tensor * k = ggml_cast(gc, ggml_permute(gc, rope(part(1)), 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor * v = ggml_cast(gc, ggml_permute(gc, part(2), 0, 2, 1, 3), GGML_TYPE_F16);
            o = ggml_flash_attn_ext(gc, q, k, v, nullptr, 1.0f / std::sqrt((float) kHeadDim), 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
        } else {
            // exact: F32 scores, one block of query rows at a time (the scores of a block, N x nq x 16 floats, stay
            // under kq_bytes()); each block's output is written into its columns of o (SET in place, no copies)
            ggml_tensor * q = ggml_cont(gc, ggml_permute(gc, ggml_scale(gc, rope(part(0)), 1.0f / std::sqrt((float) kHeadDim)), 0, 2, 1, 3));   // [64, N, 16]
            ggml_tensor * k = ggml_cont(gc, ggml_permute(gc, rope(part(1)), 0, 2, 1, 3));                   // [64, N, 16]
            ggml_tensor * vt = ggml_cont(gc, ggml_permute(gc, part(2), 1, 2, 0, 3));                        // [N, 64, 16]
            o = ggml_scale(gc, x, 0.0f);                                                                     // [1024, N]
            for (int64_t c0 = 0; c0 < N; c0 += nq) {
                const int64_t m = std::min(nq, N - c0);
                ggml_tensor * qc = ggml_cont(gc, ggml_view_3d(gc, q, kHeadDim, m, kHeads, q->nb[1], q->nb[2], (size_t) c0 * q->nb[1]));
                ggml_tensor * kq = ggml_mul_mat(gc, k, qc);                                                  // [N, m, 16]
                ggml_prec_set_acc(kq, GGML_PREC_F32);
                kq = ggml_soft_max_inplace(gc, kq);
                ggml_tensor * oc = ggml_mul_mat(gc, vt, kq);                                                 // [64, m, 16]
                ggml_prec_set_acc(oc, GGML_PREC_F32);
                oc = ggml_reshape_2d(gc, ggml_cont(gc, ggml_permute(gc, oc, 0, 2, 1, 3)), kDim, m);         // [1024, m]
                o = ggml_set_2d_inplace(gc, o, oc, o->nb[1], (size_t) c0 * o->nb[1]);
            }
        }
        o = ggml_reshape_2d(gc, o, kDim, N);                                                                 // [1024, N]
        // the rest of the block is per token: output projection, residual, MLP in blocks of kTokBlock tokens,
        // written back into x's columns (the MLP's 2 x 2816 hidden rows are the largest activation otherwise)
        for (int64_t c0 = 0; c0 < N; c0 += kTokBlock) {
            const int64_t m = std::min(kTokBlock, N - c0);
            ggml_tensor * xc = ggml_view_2d(gc, x, kDim, m, x->nb[1], (size_t) c0 * x->nb[1]);
            ggml_tensor * oc = ggml_view_2d(gc, o, kDim, m, o->nb[1], (size_t) c0 * o->nb[1]);
            ggml_tensor * y = ggml_add(gc, xc, ggml_add(gc, mm(im.B(l, "attn.wo.weight"), oc), im.B(l, "attn.wo.bias")));
            y = ggml_add(gc, y, mm(im.B(l, "mlp.w2.weight"), ggml_swiglu(gc, mm(im.B(l, "mlp.w1.weight"), rms(y, im.B(l, "norm2.weight"))))));
            x = ggml_set_2d_inplace(gc, x, y, x->nb[1], (size_t) c0 * x->nb[1]);
        }
    }
    x = rms(x, im.W("vision.norm.weight"));
    // aligner: [1024, vw, vh] padded to multiples of 3, the 3x3 blocks gathered as [1024 * 3 (c, kx), 3 (ky)] per block
    ggml_tensor * gx = ggml_reshape_3d(gc, x, kDim, vw, vh);
    const int64_t pw = (kRatio - vw % kRatio) % kRatio, ph = (kRatio - vh % kRatio) % kRatio;
    if (pw || ph) gx = ggml_pad(gc, gx, 0, (int) pw, (int) ph, 0);
    gx = ggml_reshape_4d(gc, gx, kDim * kRatio, lw, kRatio, lh);
    gx = ggml_cont(gc, ggml_permute(gc, gx, 0, 2, 1, 3));                                                    // [3072, 3, lw, lh]
    gx = ggml_reshape_2d(gc, gx, kAlignIn, lw * lh);
    ggml_tensor * y = ggml_gelu_erf(gc, ggml_add(gc, mm_blocks(im.W("aligner.w1.weight"), gx), im.W("aligner.w1.bias")));
    y = ggml_add(gc, mm_blocks(im.W("aligner.w2.weight"), y), im.W("aligner.w2.bias"));                            // [5120, lw * lh]
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    if (!im.allo) im.allo = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.be));
    if (!ggml_gallocr_alloc_graph(im.allo, gf)) {
        ggml_free(gc);
        err = "vision: cannot allocate the compute buffer";
        return false;
    }
    im.cbytes = ggml_gallocr_get_buffer_size(im.allo, 0);
    ggml_backend_tensor_set(in, patches, 0, (size_t) (N * kPatchIn) * 4);
    {
        std::vector<float> c((size_t) (N * RD)), s((size_t) (N * RD));
        float inv[16];
        for (int i = 0; i < 16; ++i) inv[i] = 1.0f / std::pow(kTheta, (float) (2 * i) / (float) RD);
        for (int64_t r = 0; r < vh; ++r)
            for (int64_t q = 0; q < vw; ++q) {
                float * cr = c.data() + (size_t) ((r * vw + q) * RD), * sr = s.data() + (size_t) ((r * vw + q) * RD);
                for (int i = 0; i < 16; ++i) {
                    const float a = (float) r * inv[i], b = (float) q * inv[i];
                    cr[i] = std::cos(a); sr[i] = std::sin(a);
                    cr[16 + i] = std::cos(b); sr[16 + i] = std::sin(b);
                }
            }
        ggml_backend_tensor_set(cs, c.data(), 0, c.size() * 4);
        ggml_backend_tensor_set(sn, s.data(), 0, s.size() * 4);
    }
    const bool ok = ggml_backend_graph_compute(im.be, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> rows;
    if (ok) {
        rows.resize((size_t) (kEmbd * lw * lh));
        ggml_backend_tensor_get(y, rows.data(), 0, rows.size() * 4);
    }
    ggml_free(gc);
    // the compute buffer goes back after every image (only the weights stay on the card between images)
    ggml_gallocr_free(im.allo);
    im.allo = nullptr;
    if (!ok) { err = "vision: the encoder graph failed"; return false; }
    // the span (model.py merge_image_embeddings)
    span.resize((size_t) (g.span_rows() * kEmbd));
    float * o = span.data();
    auto put = [&](const float * r) { std::memcpy(o, r, kEmbd * 4); o += kEmbd; };
    put(im.start.data());
    for (int64_t r = 0; r < lh; ++r) {
        for (int64_t c = 0; c < lw; ++c) put(rows.data() + (size_t) ((r * lw + c) * kEmbd));
        put(im.newline.data());
    }
    put(im.end.data());
    im.ms = now_ms() - t0;
    return true;
}
