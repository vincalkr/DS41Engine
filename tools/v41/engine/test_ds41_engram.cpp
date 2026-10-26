// test_ds41_engram - the iq4_nl Engram-table gate: the module's own row dequant (ds41_dequant_iq4_nl_row, written in
// ds41_engram.cpp because the CPU bench builds that module without ggml) against ggml's type traits, which are the
// reference the pack's quantizer used.  Random rows over the whole nibble domain and every finite binary16 scale.
#include "ds41_engram.hpp"

#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace strata::v41;
namespace {
uint64_t rs = 0x2468ace13579bdf1ull;
uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t) (rs >> 16); }
}  // namespace

int main() {
    const int64_t dim = 256;                        // the Engram row width (engram_head_dim)
    std::vector<uint8_t> w((size_t) (dim / 32 * 18));
    std::vector<float> a((size_t) dim), b((size_t) dim);
    int exact = 0, total = 0, diff = 0;
    double maxabs = 0.0;
    for (int trial = 0; trial < 4000; ++trial) {
        for (int64_t q = 0; q < dim / 32; ++q) {
            uint8_t * blk = &w[(size_t) q * 18];
            // a finite, non-zero binary16 scale: exponent 1..30, any mantissa
            const uint32_t e = 1 + rnd() % 30;
            const uint16_t h = (uint16_t) ((rnd() & 1u) << 15 | e << 10 | (rnd() & 0x3ffu));
            blk[0] = (uint8_t) (h & 0xff);
            blk[1] = (uint8_t) (h >> 8);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) rnd();          // all 256 nibble pairs
        }
        ds41_dequant_iq4_nl_row(w.data(), dim, a.data());
        ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(w.data(), b.data(), (int) dim);
        for (int64_t i = 0; i < dim; ++i) {
            ++total;
            if (std::memcmp(&a[(size_t) i], &b[(size_t) i], 4) == 0) ++exact;
            else {
                ++diff;
                const double d = std::abs((double) a[(size_t) i] - (double) b[(size_t) i]);
                if (d > maxabs) maxabs = d;
            }
        }
    }
    // and the domain endpoints: d = 1.0, every nibble value, in both nibble halves
    for (int nib = 0; nib < 16; ++nib) {
        for (int64_t q = 0; q < dim / 32; ++q) {
            uint8_t * blk = &w[(size_t) q * 18];
            blk[0] = 0x00; blk[1] = 0x3c;                                     // 1.0h
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) (nib << 4 | nib);   // both nibbles the same value
        }
        ds41_dequant_iq4_nl_row(w.data(), dim, a.data());
        ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(w.data(), b.data(), (int) dim);
        for (int64_t i = 0; i < dim; ++i) { ++total; exact += (std::memcmp(&a[(size_t) i], &b[(size_t) i], 4) == 0); }
    }
    std::printf("iq4_nl row dequant: %d of %d values bit-identical to ggml's to_float (%d differ, max abs %.3g)\n",
                exact, total, diff, maxabs);
    if (exact != total) { std::printf("FAILED\n"); return 1; }
    std::printf("all passed\n");
    return 0;
}
