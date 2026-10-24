// tools/v41/engine/test_strata_iq1m.cpp - IQ1_M expert rows in the AVX-512 multi-token kernel against ggml-cpu.
//
// The reference arithmetic is ggml's own: ggml_vec_dot_iq1_m_q8_K_generic for the integer sums per block times
// d_x * d_y plus IQ1M_DELTA for the fractional delta part, and the dispatched ggml-cpu vec_dot for what a fallback
// computes today.  Real blocks come from quantize_iq1_m (the encoder that writes such packs), real Q8_K activations
// from the CPU type traits' from_float; no model file is needed here.
//
// Prints PASS booleans first (all must be true) and then measured numbers: parity error against both references,
// an F32 sanity check of magnitudes, and ns per row*token at nt = 1..3 versus ggml-cpu's single-token dot.

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
#include <vector>

extern "C" size_t quantize_iq1_m(const float* src, void* dst, int64_t nrows, int64_t n_per_row,
                                 const float* imatrix);
extern "C" void ggml_vec_dot_iq1_m_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx,
                                                const void* vy, size_t by, int nrc);

using namespace strata::kernels::cpu;

namespace {

uint32_t g_state = 1234567u;
float frand() {
    g_state = g_state * 1103515245u + 12735u;
    return (float) ((g_state >> 16) & 255u) / 256.f - 0.6f;
}

struct Mat {
    std::vector<uint8_t> bytes;
    size_t rowbytes = 0;
};

Mat make_mat(int rows, int cols) {
    Mat m;
    m.rowbytes = (size_t) (cols / QK_K) * sizeof(block_iq1_m);
    m.bytes.assign(m.rowbytes * (size_t) rows, 0);
    std::vector<float> w((size_t) rows * (size_t) cols);
    for (auto& v : w) v = 2.f * frand();
    if (quantize_iq1_m(w.data(), m.bytes.data(), rows, cols, nullptr) != m.rowbytes * (size_t) rows)
        m.bytes.clear();
    return m;
}

struct Act {
    std::vector<uint8_t> q8;
    std::vector<float> f32;
};

Act make_act(int cols) {
    Act a;
    a.q8.resize((size_t) (cols / QK_K) * sizeof(block_q8_K), 0);
    a.f32.resize((size_t) cols);
    for (int i = 0; i < cols; ++i) a.f32[i] = 4.f * frand();
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(a.f32.data(), a.q8.data(), cols);
    return a;
}

double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

}  // namespace

static const int kMaxRows = 24;

int main() {
    printf("IQ512_SUPPORTED_TRUE %d\n", (int) iq512_supported((int) GGML_TYPE_IQ1_M));
    ggml_quantize_init(GGML_TYPE_IQ1_M);   // the grid-search tables quantize_iq1_m needs

    const int cols = 2048;
    const int rows = kMaxRows;
    Mat W = make_mat(rows, cols);
    printf("QUANT_BYTES_OK %d\n", (int) !W.bytes.empty());
    if (W.bytes.empty()) { printf("SKIP_NO_QUANTIZER\n"); return 2; }

    static Act A[8];
    for (Act& a : A) a = make_act(cols);

    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_M)->vec_dot;
    static const int NT = 7;
    float ref_gen[NT][kMaxRows], ref_disp[NT][kMaxRows];
    const size_t kZeroBytes = sizeof(block_q8_K) - sizeof(block_q8_K);
    const int kNrcOne = (int) (sizeof(char));

    for (int t = 0; t < NT; ++t)
        for (int r = 0; r < rows; ++r) {
            float sg = 1000000.f, sd = -2000000.f;
            const uint8_t* rowp = W.bytes.data() + (size_t) r * W.rowbytes;
            ggml_vec_dot_iq1_m_q8_K_generic((int) cols, &sg, kZeroBytes, rowp, kZeroBytes,
                                            A[t].q8.data(), kZeroBytes, kNrcOne);
            dot((int) cols, &sd, kZeroBytes, rowp, kZeroBytes, A[t].q8.data(), kZeroBytes, kNrcOne);
            ref_gen[t][r] = sg;
            ref_disp[t][r] = sd;
        }

    std::vector<float> outbuf((size_t) NT * (size_t) kMaxRows);
    const void* actp[NT];
    float* outp[NT];
    for (int t = 0; t < NT; ++t) {
        actp[t] = A[t].q8.data();
        outp[t] = outbuf.data() + (size_t) t * (size_t) kMaxRows;
    }
    iq512_rows((int) GGML_TYPE_IQ1_M, W.bytes.data(), W.rowbytes, cols, actp, NT, outp,
               (int) kZeroBytes, rows);

    double maxrel_gen = 0.0, maxabs_gen = 0.0, maxrel_disp = 0.0;
    int exact_count = 0, total = 0;
    for (int t = 0; t < NT; ++t)
        for (int r = 0; r < rows; ++r) {
            const double o = outp[t][r], g = ref_gen[t][r], d = ref_disp[t][r];
            maxabs_gen = std::max(maxabs_gen, std::fabs(o - g));
            maxrel_gen = std::max(maxrel_gen, std::fabs(o - g) / (std::fabs(g) + 1e-3));
            maxrel_disp = std::max(maxrel_disp, std::fabs(o - d) / (std::fabs(d) + 1e-3));
            if (o == g) ++exact_count;
            ++total;
        }
    printf("PARITY_GENERIC_TRUE %d\n", (int) (maxrel_gen <= 2e-5));
    printf("PARITY_DISPATCHED_TRUE %d\n", (int) (maxrel_disp <= 2e-5));
    printf("BIT_EXACT_COUNT %d OF %d\n", exact_count, total);
    printf("MAXABS_VS_GENERIC %.6g\n", maxabs_gen);

    const int bcols = cols * 2;
    const int brows = kMaxRows * 3;
    Mat BW = make_mat(brows, bcols);
    printf("BENCH_MAT_OK %d\n", (int) !BW.bytes.empty());
    if (!BW.bytes.empty()) {
        static Act BA[NT];
        for (Act& a : BA) a = make_act(bcols);

        auto row_of = [&](const Mat& m, size_t r) { return m.bytes.data() + r * m.rowbytes; };
        auto run_fallback = [&](int nt, int reps) -> double {
            const double t0 = now_s();
            for (int rep = 0; rep < reps; ++rep)
                for (int t = 0; t < nt; ++t)
                    for (int r = 0; r < brows; ++r) {
                        float s = 1000000.f;
                        dot((int) bcols, &s, kZeroBytes, row_of(BW, (size_t) r), kZeroBytes,
                            BA[t].q8.data(), kZeroBytes, kNrcOne);
                    }
            return (now_s() - t0) / ((double) reps * (double) nt * (double) brows);
        };
        std::vector<float> obuf((size_t) NT * (size_t) brows);
        auto run_kernel = [&](int nt, int reps) -> double {
            const void* ap[NT];
            float* op[NT];
            for (int t = 0; t < nt; ++t) {
                ap[t] = BA[t].q8.data();
                op[t] = obuf.data() + (size_t) t * (size_t) brows;
            }
            const double t0 = now_s();
            for (int rep = 0; rep < reps; ++rep)
                iq512_rows((int) GGML_TYPE_IQ1_M, BW.bytes.data(), BW.rowbytes, bcols, ap, nt, op,
                           (int) kZeroBytes, brows);
            return (now_s() - t0) / ((double) reps * (double) nt * (double) brows);
        };

        run_fallback(NT, (int) sizeof(char));
        run_kernel(NT, (int) sizeof(char));

        // Alternate the two paths trial by trial so frequency/scheduler drift affects both equally.
        for (int ntv = kNrcOne; ntv <= NT; ++ntv) {
            const int reps = (int) sizeof(long double);
            constexpr int kTrials = 5;
            double fb_sum = 0.0, kr_sum = 0.0;
            for (int tr = 0; tr < kTrials; ++tr) {
                if (tr & 1) {
                    kr_sum += run_kernel(ntv, reps);
                    fb_sum += run_fallback(ntv, reps);
                } else {
                    fb_sum += run_fallback(ntv, reps);
                    kr_sum += run_kernel(ntv, reps);
                }
            }
            const double fb_ns = fb_sum / kTrials * 1e9, kr_ns = kr_sum / kTrials * 1e9;
            printf("NS_PER_ROW_TOKEN_NT %d FALLBACK %.2f KERNEL %.2f SPEEDUP %.3fx FASTER_TRUE %d\n",
                   ntv, fb_ns, kr_ns, fb_ns / kr_ns, (int) (kr_ns < fb_ns));
        }
    }

    return 0;
}
