// fmt_cost.cpp - what an expert format costs on this box's real rows: ns per row-token of ggml-cpu's dispatched
// vec_dot (the CPU tier's fallback) against the AVX-512 rows kernel, per format, on rows taken from both packs.
//   fmt_cost <weights.bin> <cols> <type name>            (row slices of both packs in slices/)
// The numbers it produced are in docs/ds4/V41_EXACT_FAST.md.  Build (this box):
//   c++ -O3 -std=c++20 -march=native -Iinclude -Ithird_party/llama.cpp/ggml/include \
//       -Ithird_party/llama.cpp/ggml/src fmt_cost.cpp \
//       build-hip-fused/libstrata_kernels_cpu.a \
//       third_party/llama.cpp/build-hip/ggml/src/libggml.a third_party/llama.cpp/build-hip/ggml/src/libggml-base.a \
//       third_party/llama.cpp/build-hip/ggml/src/libggml-cpu.a -o fmt_cost -lpthread
#include "strata/kernels/cpu/iq_avx512.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <ggml.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::kernels::cpu;
extern "C" void ggml_vec_dot_iq1_m_q8_K_generic(int n, float* s, size_t nb, const void* vx, size_t bx,
                                                const void* vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float* s, size_t nb, const void* vx, size_t bx,
                                                 const void* vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float* s, size_t nb, const void* vx, size_t bx,
                                                  const void* vy, size_t by, int nrc);
namespace {
uint32_t g_state = 987654321u;
float frand() { g_state = g_state * 1103515245u + 12735u; return (float) ((g_state >> 16) & 255u) / 256.f - 0.5f; }
double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
volatile float sink;
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::printf("usage: %s weights.bin cols <type name>\n", argv[0]); return 2; }
    std::vector<uint8_t> W;
    {
        FILE* fp = std::fopen(argv[1], "rb");
        if (!fp) return 7;
        std::fseek(fp, 0, SEEK_END);
        const long sz = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        W.resize((size_t) sz);
        if (std::fread(W.data(), 1, (size_t) sz, fp) != (size_t) sz) return 9;
        std::fclose(fp);
    }
    const int cols = std::atoi(argv[2]);
    int type = -1;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t)
        if (ggml_get_type_traits((ggml_type) t)->type_name && !strcasecmp(ggml_type_name((ggml_type) t), argv[3]))
            type = t;
    if (type < 0) { std::printf("unknown type %s\n", argv[3]); return 3; }
    const int blck = ggml_blck_size((ggml_type) type);
    const size_t rowbytes = (size_t) (cols / blck) * ggml_type_size((ggml_type) type);
    const int rows = (int) std::min<size_t>(96, W.size() / rowbytes);
    if (rows < 8 || cols % QK_K) return 11;
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
    if (!dot) { std::printf("no cpu vec_dot for %s\n", argv[3]); return 12; }
    // ggml-cpu's generic reference: what the AVX-512 kernel is bit-exact against (the dispatched dot is not it)
    ggml_vec_dot_t gen = nullptr;
    if (type == GGML_TYPE_IQ1_M) gen = ggml_vec_dot_iq1_m_q8_K_generic;
    else if (type == GGML_TYPE_IQ2_XS) gen = ggml_vec_dot_iq2_xs_q8_K_generic;
    else if (type == GGML_TYPE_IQ2_XXS) gen = ggml_vec_dot_iq2_xxs_q8_K_generic;
    std::vector<std::vector<uint8_t>> act(8);
    for (auto& a : act) {
        a.resize((size_t) (cols / QK_K) * sizeof(block_q8_K), 0);
        std::vector<float> f((size_t) cols);
        for (auto& v : f) v = 4.f * frand();
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(f.data(), a.data(), cols);
    }
    std::printf("FMT %s COLS %d ROWBYTES %zu ROWS %d B/VAL %.4f\n", argv[3], cols, rowbytes, rows,
                (double) rowbytes / (double) cols);
    const size_t z = 0;
    for (const int nt : {1, 2, 4, 8}) {
        const void* ap[8];
        std::vector<float> outbuf((size_t) nt * (size_t) rows, 0.f);
        std::vector<float*> op((size_t) nt);
        for (int t = 0; t < nt; ++t) { ap[t] = act[(size_t) t].data(); op[(size_t) t] = outbuf.data() + (size_t) t * rows; }
        // one pass: the tier's fallback, one ggml vec_dot per row per token
        double fb = 1e30;
        for (int rep = 0; rep < 4; ++rep) {
            const double t0 = now_s();
            long passes = 0;
            do {
                for (int r = 0; r < rows; ++r) {
                    const uint8_t* p = W.data() + (size_t) r * rowbytes;
                    for (int t = 0; t < nt; ++t) {
                        float s = 0.f;
                        dot(cols, &s, z, p, z, ap[t], z, 1);
                        sink = s;
                        op[(size_t) t][r] = s;
                    }
                }
                ++passes;
            } while (now_s() - t0 < 0.15);
            fb = std::min(fb, (now_s() - t0) * 1e9 / ((double) rows * nt * (double) passes));
        }
        std::vector<float> ref = outbuf;   // the fallback's answers, for the parity line
        {
            double lo = 1e30, hi = -1e30, amax = 0.0;
            for (float v : ref) { lo = std::min(lo, (double) v); hi = std::max(hi, (double) v); amax = std::max(amax, std::fabs((double) v)); }
            std::printf("   values: min %.1f max %.1f maxabs %.1f  (float32 ulp at maxabs %.3g)\n", lo, hi, amax,
                        std::nextafter((float) amax, 2 * amax) - (float) amax);
        }
        std::vector<float> genr;
        if (gen) {
            genr.resize(outbuf.size());
            for (int r = 0; r < rows; ++r) {
                const uint8_t* p = W.data() + (size_t) r * rowbytes;
                for (int t = 0; t < nt; ++t) {
                    float s = 0.f;
                    gen(cols, &s, z, p, z, ap[t], z, 1);
                    genr[(size_t) t * rows + r] = s;
                }
            }
            double d = 0.0; int ex = 0;
            for (size_t i = 0; i < ref.size(); ++i) { d = std::max(d, std::fabs((double) ref[i] - genr[i])); ex += (ref[i] == genr[i]); }
            std::printf("   dispatched vs generic: exact %d/%d maxabs %.3g\n", ex, (int) ref.size(), d);
        }
        double kn = -1.0;
        int exact = 0, total = 0;
        double maxabs = 0.0;
        if (nt >= 2 && iq512_supported(type)) {
            std::fill(outbuf.begin(), outbuf.end(), 0.f);
            iq512_rows(type, W.data(), rowbytes, cols, ap, nt, op.data(), 0, rows);
            double gmax = 0.0; int gex = 0;
            for (size_t i = 0; i < outbuf.size(); ++i) {
                exact += (ref[i] == outbuf[i]);
                maxabs = std::max(maxabs, std::fabs((double) ref[i] - outbuf[i]));
                ++total;
                if (!genr.empty()) {
                    gex += (genr[i] == outbuf[i]);
                    gmax = std::max(gmax, std::fabs((double) genr[i] - outbuf[i]));
                }
            }
            if (!genr.empty())
                std::printf("   kernel vs generic:   exact %d/%d maxabs %.3g\n", gex, (int) outbuf.size(), gmax);
            kn = 1e30;
            for (int rep = 0; rep < 4; ++rep) {
                const double t0 = now_s();
                long passes = 0;
                do {
                    iq512_rows(type, W.data(), rowbytes, cols, ap, nt, op.data(), 0, rows);
                    ++passes;
                } while (now_s() - t0 < 0.15);
                kn = std::min(kn, (now_s() - t0) * 1e9 / ((double) rows * nt * (double) passes));
            }
            std::printf("   nt%-2d FALLBACK %7.2f  KERNEL %7.2f ns/row-token  %.2fx  exact %d/%d maxabs %.3g\n", nt, fb,
                        kn, fb / kn, exact, total, maxabs);
        } else {
            std::printf("   nt%-2d FALLBACK %7.2f  (no AVX-512 kernel for this type)\n", nt, fb);
        }
    }
    return 0;
}
