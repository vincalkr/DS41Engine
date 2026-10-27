// src/kernels/cuda/fork/fork_prom.cuh - prompt-batch ("PROM") grouped-expert kernels for the PTQ1_0 (143) and Q2_0
// (42) expert tiers (Strata-DS4 fork; docs/UPSTREAM_SYNC.md).  Fork code kept out of upstream's iq_kernels.cu (two
// added lines there: the #include and the hook in native_expert_grouped); included at a fixed point of that
// translation unit and uses its helpers (not a standalone header).
//
// WHY.  native_gu_kernel / native_down_kernel and the *_multi ones hold one OUTPUT ROW per warp and re-decode that
// row's weight words for every entry (token) of the group, so a 1024-token prompt's expert compute is dominated by
// the weight DECODE repeated per row-and-token (~2.3 s/1024 tokens, nsys).  Here each warp decodes its row ONCE
// into shared memory as int8 codes (natural element order, the -1 of the ternary/2-bit digit folded in) plus the
// stored block scale per 32 activations, then applies it to EVERY entry of the group with int8 dp4a against the
// q8_1 activations - the weight bytes are read (and decoded) once per (row, group) instead of once per
// (row, entry).  Arithmetic: the integer chain is exact, the only float difference to the old kernels is the order
// of the per-block scale multiplies (~1e-7 relative; the q8_1 activation rounding is unchanged).
#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstdio>

// One switch for the whole path: env STRATA_PROM=1 at first use, or native_expert_prom_set_enabled.
bool g_prom_enabled = [] { const char* v = std::getenv("STRATA_PROM"); return v != nullptr && v[0] == '1'; }();
// Test only (native_expert_prom_set_force): ignore the measured type/size gate below so the unit test can run the
// kernels on both tiers at every row count.  The engine never sets it.
bool g_prom_force = false;

// Rows per block in the PROM kernels: one per warp (8 warps).  Shared per row = K int8 codes + K/32 float scales.
constexpr int PROM_ROWS = 8;

// Codes of one 128-weight PTQ1_0 block in natural element order (trit digits {0,1,2} - 1 -> {-1,0,1}); `blk` is the
// 128-block, e in 0..127.  Trit extraction is the packer's inversion (ptq1_0.cpp / v41_formats._ptq1_byte), byte
// b of trit t holding ((uint8)(b * 3^t) * 3) >> 8.
__device__ __forceinline__ int8_t prom_ptq1_code(const strata_block_ptq1_0* blk, int e) {
    uint8_t b;
    int t;
    if (e < 80) { b = blk->qs[e & 15]; t = e >> 4; }
    else if (e < 120) { const int k = e - 80; b = blk->qs[16 + (k & 7)]; t = k >> 3; }
    else { const int k = e - 120; b = blk->qh[k & 1]; t = k >> 1; }
    uint32_t v = b;
#pragma unroll 4
    for (int i = 0; i < t; ++i) v = (v * 3u) & 0xFFu;
    return (int8_t) ((int) ((v * 3u) >> 8) - 1);
}

// Codes of one Q2_0 (64-weight) block: 2-bit fields 00..11 = -1,0,1,2, natural order (pack_q2_0).
__device__ __forceinline__ int8_t prom_q2_0_code(const block_q2_0* blk, int e) {
    return (int8_t) (((blk->qs[e >> 2] >> (2 * (e & 3))) & 3) - 1);
}

// Decode one weight row (K elements) into this warp's shared slot: s_w[e] = code, s_d[b] = stored scale of the
// block holding activation block b (32 activations).  `lane`-strided; K a multiple of 128 (PTQ1_0) / 64 (Q2_0).
template<int TY>
__device__ __forceinline__ void prom_decode_row(const uint8_t* __restrict__ row, int8_t* __restrict__ s_w,
                                                float* __restrict__ s_d, int K, int lane) {
    if constexpr (TY == 143) {
        const strata_block_ptq1_0* b128 = (const strata_block_ptq1_0*) row;
        for (int e = lane; e < K; e += 32) s_w[e] = prom_ptq1_code(b128 + (e >> 7), e & 127);
        for (int b = lane; b < (K >> 5); b += 32) s_d[b] = __half2float(b128[b >> 2].d);
    } else {
        const block_q2_0* b64 = (const block_q2_0*) row;
        for (int e = lane; e < K; e += 32) s_w[e] = prom_q2_0_code(b64 + (e >> 6), e & 63);
        for (int b = lane; b < (K >> 5); b += 32) s_d[b] = __half2float(b64[b >> 1].d);
    }
}

__device__ __forceinline__ float prom_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Gate/up (GU) or down (DWN).  Same grouped-launch contract as native_gu_kernel / native_down_kernel: group g's
// blob at grp_ptr[g], its entries grp_start[g]..grp_start[g+1), each entry's token row ent_tok[e] (GU) / its own
// index e (DWN, the SwiGLU output layout), outputs at entry index (GU) / ent_dst[e] (DWN).
template<int TY, bool GU>
__global__ void __launch_bounds__(256) native_prom_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent,
                                                          const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                          float* __restrict__ gate, float* __restrict__ up,
                                                          float* __restrict__ out) {
    extern __shared__ uint8_t prom_smem[];
    const int K = GU ? (int) L.n_embd : (int) L.n_ff;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_rows = GU ? (int) (2 * L.n_ff) : (int) L.n_embd;
    const int row = blockIdx.x * PROM_ROWS + warp;
    if (row >= n_rows) return;
    int8_t* s_w = (int8_t*) prom_smem + (size_t) warp * (size_t) K;
    float* s_d = (float*) (prom_smem + (size_t) PROM_ROWS * ((size_t) K + 15 & ~(size_t) 15)) + (size_t) warp * (size_t) (K >> 5);
    const bool is_up = GU && row >= L.n_ff;
    const int rr = is_up ? row - (int) L.n_ff : row;
    const size_t row_off = (GU ? (is_up ? L.up_off : 0) : L.down_off) + (size_t) rr * (GU ? L.gu_row : L.d_row);
    float* dst = GU ? (is_up ? up : gate) : out;
    const int xstride = GU ? (int) (L.n_embd / 32) : (int) (L.n_ff / 32);
    const int nblk = xstride;   // q8_1 activation blocks per row == K / 32
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        prom_decode_row<TY>((const uint8_t*) grp_ptr[g] + row_off, s_w, s_d, K, lane);
        __syncwarp();
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const block_q8_1* xb = xq + (size_t) (GU ? ent[e] : e) * xstride;
            float acc = 0.0f;
            for (int i = 0; i < nblk; i += 4) {
                const int b = i + (lane >> 3);
                const uint32_t cw = ((const uint32_t*) s_w)[32 * (i >> 2) + lane];
                const uint32_t aw = ((const uint32_t*) xb[b].qs)[lane & 7];
                const int dot = ggml_cuda_dp4a((int) cw, (int) aw, 0);
                acc += s_d[b] * __low2float(xb[b].ds) * (float) dot;
            }
            acc = prom_warp_sum(acc);
            if (lane == 0) {
                if (GU) dst[(size_t) e * L.n_ff + rr] = acc;
                else out[(size_t) ent[e] * L.n_embd + rr] = acc;
            }
        }
        __syncwarp();
    }
}

// Dynamic shared bytes of one PROM block (8 rows of K: codes + per-32 scales), 16-byte aligned rows.  Over 48 KiB the
// launch would need cudaFuncSetAttribute, so the launchers bail to the default kernels instead (K <= 5461).
inline size_t prom_shared(int64_t K) {
    const size_t kp = ((size_t) K + 15) & ~(size_t) 15;
    return (size_t) PROM_ROWS * (kp + (size_t) (K >> 5) * sizeof(float));
}
inline bool prom_fits(int64_t K) { return prom_shared(K) <= 48 * 1024; }

inline void prom_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// Gate/up half: true when the PROM kernel was launched for this layer's format (caller runs its own SwiGLU + q8_1
// pass either way), false to fall through to the default launch_gu.  `gy` is the caller's group-row count and
// `large_groups` says the call averages >= 64 entries per group.
//
// THE GATE is the measured envelope (test_ds41_prom on real PTQ1_0 blobs, 5 launches each, min): PTQ1_0 gate/up is
// 1.21x / 1.37x / 1.45x the native kernel at 64 / 512 / 1024 entries per group but 0.33x / 0.64x at 1 / 7 (the
// shared-memory weight tile costs more than the per-row decode it saves when the tile is used a few times), and Q2_0
// is 0.57-0.80x at every size (its native multi kernel already reuses the decode 4x with every lane busy, so the
// tile's LDS traffic is a straight loss).  So: PTQ1_0 only, large groups only.
inline bool prom_gu(const NativeExpertLayout& L, int64_t gy, bool large_groups, cudaStream_t s,
                    const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_tok, const block_q8_1* X, float* gate, float* up) {
    if (!g_prom_enabled || L.up_type >= 0) return false;                                  // mixed gate/up: not covered
    if (!g_prom_force && (L.gu_type != 143 || !large_groups)) return false;
    if ((L.gu_type != 143 && L.gu_type != 42) || (L.n_embd & 127) != 0 || !prom_fits(L.n_embd)) return false;
    const dim3 grid((unsigned) ((2 * L.n_ff + PROM_ROWS - 1) / PROM_ROWS), (unsigned) gy);
    if (L.gu_type == 143) native_prom_kernel<143, true><<<grid, 256, prom_shared(L.n_embd), s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, nullptr);
    else native_prom_kernel<42, true><<<grid, 256, prom_shared(L.n_embd), s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, nullptr);
    prom_check("native_expert_prom/gu");
    return true;
}

// Down half: same contract, `hq` is the SwiGLU output quantized to q8_1 (per entry index).
// Down half: as prom_gu.  PTQ1_0 down measures 1.02-1.05x the native kernel alone (Q2_0 0.9x), so the same gate.
inline bool prom_down(const NativeExpertLayout& L, int64_t gy, bool large_groups, cudaStream_t s,
                      const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                      const int32_t* ent_dst, const block_q8_1* hq, float* out) {
    if (!g_prom_enabled) return false;
    if (!g_prom_force && (L.d_type != 143 || !large_groups)) return false;
    if ((L.d_type != 143 && L.d_type != 42) || (L.n_ff & 127) != 0 || !prom_fits(L.n_ff)) return false;
    const dim3 grid((unsigned) ((L.n_embd + PROM_ROWS - 1) / PROM_ROWS), (unsigned) gy);
    if (L.d_type == 143) native_prom_kernel<143, false><<<grid, 256, prom_shared(L.n_ff), s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, nullptr, nullptr, out);
    else native_prom_kernel<42, false><<<grid, 256, prom_shared(L.n_ff), s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, nullptr, nullptr, out);
    prom_check("native_expert_prom/down");
    return true;
}
