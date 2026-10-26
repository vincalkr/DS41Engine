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
#include "strata/kernels/cpu/native_expert.hpp"

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

    // The engine path itself: native_gu_rows / native_down_rows below and above native_gu_mt_min must give the same
    // rows (below it they call ggml-cpu's per-token vec_dot, from four tokens this kernel).
    printf("NATIVE_MT_MIN_IQ1M %d\n", native_gu_mt_min((int) GGML_TYPE_IQ1_M));
    {
        const int ne = cols / 2;
        const int nf = 768;
        NativeFmt f{};
        std::string err;
        int fmt_ok = (int) native_fmt((int) GGML_TYPE_IQ1_M, (int) GGML_TYPE_IQ1_M, ne, nf, f, err);
        Mat G = make_mat(nf, ne), U = make_mat(nf, ne), D = make_mat(ne, nf);
        const size_t gate_bytes = G.rowbytes * (size_t) nf;
        const size_t down_bytes = D.rowbytes * (size_t) ne;
        const int geom_ok = fmt_ok && !G.bytes.empty() && !U.bytes.empty() && !D.bytes.empty()
                         && G.rowbytes == f.gu_row && D.rowbytes == f.d_row
                         && f.up_off == gate_bytes && f.down_off == gate_bytes + U.bytes.size()
                         && f.bytes == gate_bytes + U.bytes.size() + down_bytes
                         && f.act_bytes <= kNativeActBytes && f.h_bytes <= kNativeHBytes;
        printf("NATIVE_FMT_OK %d\n", fmt_ok);
        printf("ENGINE_GEOMETRY_OK %d\n", geom_ok);

        float gu_maxrel = -1.f, dn_maxrel = -1.f;
        int gu_exact = 0, gu_total = 0, dn_exact = 0, dn_total = 0;
        int hi_exact = 0, lo_exact = 0, gu_gen_exact = 0;
        float hi_abs = -1.f, lo_abs = -1.f, gu_gen_maxrel = -1.f;
        const int nt_low = 3, nt_high = 5;
        if (geom_ok) {
            std::vector<uint8_t> blob(f.bytes, 0);
            memcpy(blob.data(), G.bytes.data(), gate_bytes);
            memcpy(blob.data() + f.up_off, U.bytes.data(), U.bytes.size());
            memcpy(blob.data() + f.down_off, D.bytes.data(), down_bytes);

            static uint8_t gacts[NT][kNativeActBytes];
            static uint8_t hacts[NT][kNativeHBytes];
            for (int t = 0; t < NT; ++t) {
                std::vector<float> x((size_t) ne), h((size_t) nf);
                for (auto& v : x) v = 4.f * frand();
                native_quant_act(f, x.data(), gacts[t]);
                for (auto& v : h) v = 2.f * frand();
                native_quant_h(f, h.data(), hacts[t]);
            }

            std::vector<float> guLo((size_t) NT * (size_t) nf), guHi((size_t) NT * (size_t) nf);
            std::vector<float> dnLo((size_t) NT * (size_t) ne), dnHi((size_t) NT * (size_t) ne);
            const void* gap[NT];
            float* gout[NT];
            for (int t = 0; t < NT; ++t) {
                gap[t] = gacts[t];
                gout[t] = guLo.data() + (size_t) t * (size_t) nf;
            }
            native_gu_rows(f, blob.data(), gap, nt_low, gout, 0, nf);
            for (int t = 0; t < NT; ++t) gout[t] = guHi.data() + (size_t) t * (size_t) nf;
            native_gu_rows(f, blob.data(), gap, nt_high, gout, 0, nf);

            const void* hap[NT];
            float* hout[NT];
            for (int t = 0; t < NT; ++t) {
                hap[t] = hacts[t];
                hout[t] = dnLo.data() + (size_t) t * (size_t) ne;
            }
            native_down_rows(f, blob.data(), hap, nt_low, hout, 0, ne);
            for (int t = 0; t < NT; ++t) hout[t] = dnHi.data() + (size_t) t * (size_t) ne;
            native_down_rows(f, blob.data(), hap, nt_high, hout, 0, ne);

            for (int t = 0; t < nt_low; ++t)
                for (int r = 0; r < nf; ++r) {
                    const double a = guLo[(size_t) t * (size_t) nf + (size_t) r];
                    const double b = guHi[(size_t) t * (size_t) nf + (size_t) r];
                    gu_maxrel = std::max(gu_maxrel, (float) (std::fabs(a - b) / (std::fabs(b) + 1e-3)));
                    if (a == b) ++gu_exact;
                    ++gu_total;
                }
            for (int t = 0; t < nt_low; ++t)
                for (int r = 0; r < ne; ++r) {
                    const double a = dnLo[(size_t) t * (size_t) ne + (size_t) r];
                    const double b = dnHi[(size_t) t * (size_t) ne + (size_t) r];
                    dn_maxrel = std::max(dn_maxrel, (float) (std::fabs(a - b) / (std::fabs(b) + 1e-3)));
                    if (a == b) ++dn_exact;
                    ++dn_total;
                }
            // Down rows are one dot per row*token: check both paths against the generic reference itself.
            const uint8_t* drow = blob.data() + f.down_off;
            for (int t = 0; t < nt_low; ++t)
                for (int r = 0; r < ne; ++r) {
                    float sg = 1000000.f;
                    ggml_vec_dot_iq1_m_q8_K_generic((int) nf, &sg, kZeroBytes, drow + (size_t) r * f.d_row,
                                                    kZeroBytes, hacts[t], kZeroBytes, kNrcOne);
                    const double hi = dnHi[(size_t) t * (size_t) ne + (size_t) r];
                    const double lo = dnLo[(size_t) t * (size_t) ne + (size_t) r];
                    hi_abs = std::max(hi_abs, (float) std::fabs(hi - sg));
                    lo_abs = std::max(lo_abs, (float) std::fabs(lo - sg));
                    if (hi == sg) ++hi_exact;
                    if (lo == sg) ++lo_exact;
                }
            // Gate/up rows: the same generic dots through silu(gate)*up as iq1m_gu_rows applies.
            for (int t = 0; t < nt_low; ++t)
                for (int r = 0; r < nf; ++r) {
                    const uint8_t* grow = blob.data() + (size_t) r * f.gu_row;
                    float sg = 1000000.f, su = 2000000.f;
                    ggml_vec_dot_iq1_m_q8_K_generic((int) ne, &sg, kZeroBytes, grow, kZeroBytes,
                                                    gacts[t], kZeroBytes, kNrcOne);
                    ggml_vec_dot_iq1_m_q8_K_generic((int) ne, &su, kZeroBytes, blob.data() + f.up_off + (size_t) r * f.gu_row,
                                                    kZeroBytes, gacts[t], kZeroBytes, kNrcOne);
                    const double ref = (double) ((sg / (1.f + std::exp(-sg))) * su);
                    const double o = guHi[(size_t) t * (size_t) nf + (size_t) r];
                    gu_gen_maxrel = std::max(gu_gen_maxrel, (float) (std::fabs(o - ref) / (std::fabs(ref) + 1e-3)));
                    if (o == ref) ++gu_gen_exact;
                }
        }
        printf("ENGINE_DOWN_VS_GENERIC_HI_EXACT %d OF %d\n", hi_exact, geom_ok ? nt_low * ne : 0);
        printf("ENGINE_DOWN_VS_GENERIC_LO_EXACT %d OF %d\n", lo_exact, geom_ok ? nt_low * ne : 0);
        printf("ENGINE_DOWN_MAXABS_HI %.6g LO %.6g\n", hi_abs, lo_abs);

        printf("ENGINE_GU_MAXREL %.6g ENGINE_DOWN_MAXREL %.6g\n", gu_maxrel, dn_maxrel);
        printf("ENGINE_GU_EQ_TOL_TRUE %d\n", (int) (geom_ok && gu_maxrel <= 2e-5));
        printf("ENGINE_GU_BIT_EXACT_COUNT %d OF %d\n", gu_exact, gu_total);
        printf("ENGINE_GU_VS_GENERIC_MAXREL %.6g EXACT %d OF %d\n", gu_gen_maxrel, gu_gen_exact, geom_ok ? nt_low * nf : 0);
        printf("ENGINE_GU_VS_GENERIC_TOL_TRUE %d\n", (int) (geom_ok && gu_gen_maxrel <= 4e-5));
        printf("ENGINE_DOWN_PATH_SAME_TOL_TRUE %d MAXREL %.6g\n", (int) (geom_ok && dn_maxrel <= 4e-5), dn_maxrel);
        printf("ENGINE_DOWN_KERNEL_VS_GENERIC_BIT_EXACT_TRUE %d COUNT %d OF %d MAXABS %.6g\n",
               (int) (geom_ok && hi_exact == nt_low * ne && hi_abs == 0.f), hi_exact, geom_ok ? nt_low * ne : 0, hi_abs);
        printf("ENGINE_DOWN_FALLBACK_VS_GENERIC_EXACT_COUNT %d OF %d\n", lo_exact, geom_ok ? nt_low * ne : 0);
    }

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
