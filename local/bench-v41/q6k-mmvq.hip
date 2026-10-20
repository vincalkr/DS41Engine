// q6k-mmvq.hip - the decode dense half's Q6_K matrix-vector products in isolation, on one RX 7900 XTX (gfx1100).
//
// ggml's kernel (a copy of mul_mat_vec_q<Q6_K, ncols_dst 1> and quantize_q8_1 as ggml launches them on gfx1100:
// 2 waves a row, each lane one int of ql per Q6_K block) against q6k_mmvq below: 4 lanes a block, each lane 16 B of
// ql, 16 B of ql+32 and 16 B of qh (64 weights) with 128-bit loads on the standard Q6_K layout (blocks are 210 B, so
// most of these loads are only 2-byte aligned; gfx1100 takes them), R rows a wave sharing the activation registers,
// the activations as int8 + one float scale per 32 (the q8_1 values, so the integer dots are ggml's).
//
// Shapes: DeepSeek-V4.1-Flash's dense half at decode (K x rows [x channels], launches a token): q_a 5120x1280,
// q_b 1280x32768, kv 5120x512, out_a 4096x1024x8, out_b 8192x5120, gate/up_shexp 5120x2304 (2), down_shexp
// 2304x5120, head 5120x129280 (1). Random weights; each launch reads another copy of them (>= 768 MB rotated), so the
// 96 MB Infinity Cache does not hold them, as in decode, where 5.9 GB stream through per token.
//
//   q6k-mmvq [check|shapes|sweep|roof|token] [-n N] [-gap US] [-v FAMILIES] [-s SHAPES]
//     check   each shape, both kernels against a CPU reference (double) on the same q8 activations
//     shapes  us a launch and GB/s per shape (weights only), ggml vs the default q6k_mmvq configuration
//     sweep   every configuration per shape of the kernel families in -v (default ggml,rows,flat,fq,h,f)
//     roof    a plain coalesced read of each shape's bytes: the ceiling for that size
//     token   a decode token's 321 matvecs (+ their activation quantization) on 40 distinct layers + head, ms a token;
//             -gap US sleeps US between layers (the expert tier's time), so the clocks behave as in decode
//
// Build (from the repository root):
//   /opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1100 -O3 -std=gnu++17 -DGGML_USE_HIP \
//     -D__HIP_PLATFORM_AMD__=1 -DQ6K_MMVQ_SWEEP -Ithird_party/llama.cpp/ggml/src/ggml-cuda -Ithird_party/llama.cpp/ggml/src \
//     -Ithird_party/llama.cpp/ggml/include --hip-link -o build-hip-fused/q6k-mmvq local/bench-v41/q6k-mmvq.hip
#include "common.cuh"
#include "vecdotq.cuh"
#include "mmvq-q6k.cuh"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

// ------------------------------------------------------------------ ggml's kernels, as launched on gfx1100

// quantize_q8_1 (quantize.cu) for one column per channel, K a multiple of 512 (no padding)
static __global__ void __launch_bounds__(256, 1) base_quantize_q8_1(const float * __restrict__ x, block_q8_1 * __restrict__ y,
                                                                    const int K, const int64_t sx_ch) {
    const int i0 = blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= K) {
        return;
    }
    const int64_t i_cont = (int64_t) blockIdx.y*K + i0;
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;
    const float xi = x[blockIdx.y*sx_ch + i0];
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    y[ib].ds = make_half2(d, sum);
}

// mul_mat_vec_q<GGML_TYPE_Q6_K, 1, false> (mmvq.cu) with the RDNA3_0 table: nwarps 2, one row a block
static __global__ void __launch_bounds__(64, 1) base_mmvq_q6k(const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst) {
    constexpr int qk = QK_K, qi = QI6_K, vdr = VDR_Q6_K_Q8_1_MMVQ, nwarps = 2, warp_size = 32;
    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;
    const uint32_t channel = blockIdx.y;

    float tmp = 0.0f;
    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel*stride_channel_y;
    const int kbx_offset = channel*stride_channel_x + row0*stride_row_x;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
        tmp += vec_dot_q6_K_q8_1(vx, &y[kby], kbx_offset + kbx, kqs);
    }
    __shared__ float tmp_shared[nwarps-1][warp_size];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y-1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    for (int l = 0; l < nwarps-1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum<warp_size>(tmp);
    if (threadIdx.x == 0) {
        dst[channel*stride_channel_dst + row0] = tmp;
    }
}

// ggml's kernel with other launch parameters: NW waves a row (ggml: 2 on RDNA3), RPB rows a workgroup (ggml: 1)
template <int NW, int RPB>
static __global__ void __launch_bounds__(NW*32, 1) ggml_like_mmvq(const void * __restrict__ vx, const void * __restrict__ vy,
        float * __restrict__ dst, const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst) {
    constexpr int qk = QK_K, qi = QI6_K, vdr = VDR_Q6_K_Q8_1_MMVQ, warp_size = 32;
    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = RPB*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * NW*warp_size / qi;
    const uint32_t channel = blockIdx.y;
    float tmp[RPB];
#pragma unroll
    for (int i = 0; i < RPB; ++i) tmp[i] = 0.0f;
    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel*stride_channel_y;
    const int kbx_offset = channel*stride_channel_x + row0*stride_row_x;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int i = 0; i < RPB; ++i) tmp[i] += vec_dot_q6_K_q8_1(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
    }
    __shared__ float tmp_shared[NW > 1 ? NW-1 : 1][RPB][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < RPB; ++i) {
#pragma unroll
        for (int l = 0; l < NW-1; ++l) tmp[i] += tmp_shared[l][i][threadIdx.x];
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if (threadIdx.x == i) dst[channel*stride_channel_dst + row0 + i] = tmp[i];
    }
}

// q6k_mmvq with the R rows of a wave taken as one run of R*NB blocks, 8 a step (full lanes when R*NB % 8 == 0);
// the activations are loaded per step (the block column changes along the run). Rows must be contiguous.
template <int NB, int R, int W>
static __global__ void __launch_bounds__(W*32, 1) q6k_mmvq_flat(
        const uint8_t * __restrict__ vx, const int8_t * __restrict__ yq, const float * __restrict__ yd,
        float * __restrict__ dst, const int nrows, const int64_t ch_bytes, const int64_t dst_ch) {
    constexpr int K = NB*QK_K, NF = R*NB;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x*W + threadIdx.y)*R;
    if (row0 >= nrows) return;
    const int bl = lane >> 2, h = (lane >> 1) & 1, s = lane & 1;
    vx += blockIdx.y*ch_bytes + (int64_t) row0*NB*210 + 64*h + 16*s;
    yq += (int64_t) blockIdx.y*K + 128*h + 16*s;
    yd += (int64_t) blockIdx.y*(K/32) + 4*h;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int st = 0; st < (NF + 7)/8; ++st) {
        const int f = 8*st + bl;
        if (NF % 8 != 0 && f >= NF) break;
        const int r = f / NB, j = f - r*NB;
        const int4   a0 = *(const int4 *) (yq + 256*j +  0);
        const int4   a1 = *(const int4 *) (yq + 256*j + 32);
        const int4   a2 = *(const int4 *) (yq + 256*j + 64);
        const int4   a3 = *(const int4 *) (yq + 256*j + 96);
        const float4 d8 = *(const float4 *) (yd + 8*j);
        const uint8_t * b = vx + 210*f;
        const q6k_i32x4 la = *(const q6k_i32x4 *) (b);
        const q6k_i32x4 lb = *(const q6k_i32x4 *) (b + 32);
        const q6k_i32x4 qh = *(const q6k_i32x4 *) (b + 128 - 32*h);
        const q6k_i32x2 sc = *(const q6k_i32x2 *) (b + 192 - 56*h - 16*s);
        const float     dw = __half2float(*(const half *) (b + 208 - 64*h - 16*s));
        constexpr int ones = 0x01010101;
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0, z0 = 0, z1 = 0, z2 = 0, z3 = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            z0 = ggml_cuda_dp4a(a0[k], ones, z0); z1 = ggml_cuda_dp4a(a1[k], ones, z1);
            z2 = ggml_cuda_dp4a(a2[k], ones, z2); z3 = ggml_cuda_dp4a(a3[k], ones, z3);
        }
        s0 = -32*z0; s1 = -32*z1; s2 = -32*z2; s3 = -32*z3;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t l = la[k], m = lb[k], q = qh[k];
            s0 = ggml_cuda_dp4a(( l       & 0x0F0F0F0F) | ((q << 4) & 0x30303030), a0[k], s0);
            s1 = ggml_cuda_dp4a(( m       & 0x0F0F0F0F) | ((q << 2) & 0x30303030), a1[k], s1);
            s2 = ggml_cuda_dp4a(((l >> 4) & 0x0F0F0F0F) | ( q       & 0x30303030), a2[k], s2);
            s3 = ggml_cuda_dp4a(((m >> 4) & 0x0F0F0F0F) | ((q >> 2) & 0x30303030), a3[k], s3);
        }
        const int c0 = (int8_t) (sc.x >> (8*s)), c1 = (int8_t) (sc.x >> (16 + 8*s));
        const int c2 = (int8_t) (sc.y >> (8*s)), c3 = (int8_t) (sc.y >> (16 + 8*s));
        float v = d8.x * (float) (s0*c0);
        v = fmaf(d8.y, (float) (s1*c1), v);
        v = fmaf(d8.z, (float) (s2*c2), v);
        v = fmaf(d8.w, (float) (s3*c3), v);
        v *= dw;
        const int rlo = (8*st)/NB, rhi = min(R - 1, (8*st + 7)/NB);   // the rows the step's blocks belong to
#pragma unroll
        for (int rr = rlo; rr <= rhi; ++rr) {
            if (r == rr) acc[rr] += v;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = warp_reduce_sum<32>(acc[r]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) dst[blockIdx.y*dst_ch + row0 + r] = acc[r];
    }
}

// a plain read at a 2-byte offset: what unaligned 128-bit loads cost
static __global__ void __launch_bounds__(256, 1) read_bytes_off2(const uint8_t * __restrict__ x, const size_t n16, int * __restrict__ out) {
    int acc = 0;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i + 1 < n16; i += (size_t) gridDim.x*blockDim.x) {
        const q6k_i32x4 v = *(const q6k_i32x4 *) (x + 16*i + 2);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc = warp_reduce_sum<32>(acc);
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// an empty kernel launched with grid (7, 1000 + shape index) at the start of each shape: splits a trace by shape
static __global__ void shape_marker() {}

// the ceiling: a plain coalesced read of the same bytes (16 B a lane, grid-stride), one result a workgroup
static __global__ void __launch_bounds__(256, 1) read_bytes(const int4 * __restrict__ x, const size_t n16, int * __restrict__ out) {
    int acc = 0;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x*blockDim.x) {
        const int4 v = x[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    acc = warp_reduce_sum<32>(acc);
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ------------------------------------------------------------------ the variants that lost (kept to re-measure)
//   q6k_mmvq      the same loads, the activations from quantize_q8_1 in two arrays (int8 qs, float d): the integer
//                 dots of ggml, dp4a, R rows sharing the activation registers
//   q6k_mmvq_fq   q6k_mmvq with each workgroup quantizing the column into LDS exactly as quantize_q8_1 does
//   q6k_mmvq_h    f16 arithmetic: the column converted to f16, v_dot2_f32_f16

struct q6k_act {
    int8_t * qs;   // [nch][K]
    float  * d;    // [nch][K/32]
};

static __global__ void __launch_bounds__(256, 1) q6k_quantize_act(
        const float * __restrict__ x, int8_t * __restrict__ qs, float * __restrict__ d, const int K, const int64_t sx_ch) {
    const int i0 = blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= K) {
        return;
    }
    const float xi   = x[blockIdx.y*sx_ch + i0];
    const float amax = warp_reduce_max<32>(fabsf(xi));
    const float dq   = amax / 127.0f;
    const int8_t q   = amax == 0.0f ? 0 : roundf(xi / dq);   // as quantize_q8_1
    qs[(int64_t) blockIdx.y*K + i0] = q;
    if ((i0 & 31) == 0) {
        d[(int64_t) blockIdx.y*(K/32) + i0/32] = __half2float(__float2half(dq));
    }
}

// x [nch][K] floats, channel stride sx_ch floats; K a multiple of 32
static void q6k_quantize_act_cuda(const float * x, const q6k_act & act, const int K, const int nch, const int64_t sx_ch,
                                  cudaStream_t stream) {
    q6k_quantize_act<<<dim3((K + 255) / 256, nch), 256, 0, stream>>>(x, act.qs, act.d, K, sx_ch);
}

template <int NB, int R, int W>
static __global__ void __launch_bounds__(W*32, 1) q6k_mmvq(
        const uint8_t * __restrict__ vx, const int8_t * __restrict__ yq, const float * __restrict__ yd,
        float * __restrict__ dst, const int nrows, const int64_t row_bytes, const int64_t ch_bytes, const int64_t dst_ch) {
    constexpr int K = NB*QK_K;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x*W + threadIdx.y)*R;
    if (row0 >= nrows) {
        return;
    }
    const int bl = lane >> 2;         // block of the step
    const int h  = (lane >> 1) & 1;   // half of the block
    const int s  = lane & 1;          // l = 16s .. 16s+15 of the half
    vx += blockIdx.y*ch_bytes + (int64_t) row0*row_bytes + 64*h + 16*s;
    yq += (int64_t) blockIdx.y*K + 128*h + 16*s;
    yd += (int64_t) blockIdx.y*(K/32) + 4*h;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }

#pragma unroll
    for (int st = 0; st < (NB + 7)/8; ++st) {
        const int j = 8*st + bl;
        if (NB % 8 != 0 && j >= NB) {
            break;
        }
        // activations 256j + 128h + 32g + 16s + [0, 16) for g = 0..3, shared by the R rows
        const int4   a0 = *(const int4 *) (yq + 256*j +  0);
        const int4   a1 = *(const int4 *) (yq + 256*j + 32);
        const int4   a2 = *(const int4 *) (yq + 256*j + 64);
        const int4   a3 = *(const int4 *) (yq + 256*j + 96);
        const float4 d8 = *(const float4 *) (yd + 8*j);
        constexpr int ones = 0x01010101;
        int z0 = ggml_cuda_dp4a(a0.x, ones, 0), z1 = ggml_cuda_dp4a(a1.x, ones, 0);
        int z2 = ggml_cuda_dp4a(a2.x, ones, 0), z3 = ggml_cuda_dp4a(a3.x, ones, 0);
        z0 = ggml_cuda_dp4a(a0.y, ones, z0); z1 = ggml_cuda_dp4a(a1.y, ones, z1);
        z2 = ggml_cuda_dp4a(a2.y, ones, z2); z3 = ggml_cuda_dp4a(a3.y, ones, z3);
        z0 = ggml_cuda_dp4a(a0.z, ones, z0); z1 = ggml_cuda_dp4a(a1.z, ones, z1);
        z2 = ggml_cuda_dp4a(a2.z, ones, z2); z3 = ggml_cuda_dp4a(a3.z, ones, z3);
        z0 = ggml_cuda_dp4a(a0.w, ones, z0); z1 = ggml_cuda_dp4a(a1.w, ones, z1);
        z2 = ggml_cuda_dp4a(a2.w, ones, z2); z3 = ggml_cuda_dp4a(a3.w, ones, z3);
        z0 *= -32; z1 *= -32; z2 *= -32; z3 *= -32;

#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t * b = vx + r*row_bytes + 210*j;   // + 64h + 16s
            const q6k_i32x4 la = *(const q6k_i32x4 *) (b);
            const q6k_i32x4 lb = *(const q6k_i32x4 *) (b + 32);
            const q6k_i32x4 qh = *(const q6k_i32x4 *) (b + 128 - 32*h);
            const q6k_i32x2 sc = *(const q6k_i32x2 *) (b + 192 - 56*h - 16*s);
            const float     dw = __half2float(*(const half *) (b + 208 - 64*h - 16*s));
            int s0 = z0, s1 = z1, s2 = z2, s3 = z3;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t l = la[k], m = lb[k], q = qh[k];
                s0 = ggml_cuda_dp4a(( l       & 0x0F0F0F0F) | ((q << 4) & 0x30303030), a0[k], s0);
                s1 = ggml_cuda_dp4a(( m       & 0x0F0F0F0F) | ((q << 2) & 0x30303030), a1[k], s1);
                s2 = ggml_cuda_dp4a(((l >> 4) & 0x0F0F0F0F) | ( q       & 0x30303030), a2[k], s2);
                s3 = ggml_cuda_dp4a(((m >> 4) & 0x0F0F0F0F) | ((q >> 2) & 0x30303030), a3[k], s3);
            }
            // scales[8h + 2g + s]: byte s and 2+s of the half's first 4 (g = 0, 1) and last 4 (g = 2, 3)
            const int c0 = (int8_t) (sc.x >> (8*s)), c1 = (int8_t) (sc.x >> (16 + 8*s));
            const int c2 = (int8_t) (sc.y >> (8*s)), c3 = (int8_t) (sc.y >> (16 + 8*s));
            float f = d8.x * (float) (s0*c0);
            f = fmaf(d8.y, (float) (s1*c1), f);
            f = fmaf(d8.z, (float) (s2*c2), f);
            f = fmaf(d8.w, (float) (s3*c3), f);
            acc[r] = fmaf(dw, f, acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = warp_reduce_sum<32>(acc[r]);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            dst[blockIdx.y*dst_ch + row0 + r] = acc[r];
        }
    }
}

// q6k_mmvq with the activations quantized by each workgroup into LDS, exactly as quantize_q8_1 does it (the same q8
// values and half-rounded scales), so no quantize kernel runs before it. The workgroups loop over row groups
// (grid-stride), so the quantization is done once per workgroup, not once per R*W rows.
template <int NB, int R, int W>
static __global__ void __launch_bounds__(W*32, 1) q6k_mmvq_fq(
        const uint8_t * __restrict__ vx, const float * __restrict__ x, float * __restrict__ dst, const int nrows,
        const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch) {
    constexpr int K = NB*QK_K;
    __shared__ int4   s_q[K/16];    // int8 [K]
    __shared__ float4 s_d[K/128];   // float [K/32]

    const int lane = threadIdx.x;
    const int tid  = threadIdx.y*32 + lane;

    // quantize: 4 values a lane, 8 lanes a q8 block of 32
    x += blockIdx.y*sx_ch;
#pragma unroll
    for (int i0 = 0; i0 < K; i0 += W*128) {
        const int i = i0 + 4*tid;
        if (K % (W*128) != 0 && i >= K) {
            break;
        }
        const float4 v = *(const float4 *) (x + i);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 8));
        }
        const float dq = amax / 127.0f;
        int q = 0;
        if (amax != 0.0f) {
            q = (int) (uint8_t) (int8_t) roundf(v.x / dq)
              | (int) (uint8_t) (int8_t) roundf(v.y / dq) <<  8
              | (int) (uint8_t) (int8_t) roundf(v.z / dq) << 16
              | (int) (uint8_t) (int8_t) roundf(v.w / dq) << 24;
        }
        ((int *) s_q)[i/4] = q;
        if ((lane & 7) == 0) {
            ((float *) s_d)[i/32] = __half2float(__float2half(dq));
        }
    }
    __syncthreads();

    const int bl = lane >> 2;
    const int h  = (lane >> 1) & 1;
    const int s  = lane & 1;
    vx += blockIdx.y*ch_bytes + 64*h + 16*s;
    const int4   * yq = s_q + (128*h + 16*s)/16;
    const float4 * yd = s_d + h;

    for (int row0 = (blockIdx.x*W + threadIdx.y)*R; row0 < nrows; row0 += gridDim.x*W*R) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = 0.0f;
        }
        const uint8_t * xr = vx + (int64_t) row0*row_bytes;
#pragma unroll
        for (int st = 0; st < (NB + 7)/8; ++st) {
            const int j = 8*st + bl;
            if (NB % 8 != 0 && j >= NB) {
                break;
            }
            const int4   a0 = yq[16*j + 0];
            const int4   a1 = yq[16*j + 2];
            const int4   a2 = yq[16*j + 4];
            const int4   a3 = yq[16*j + 6];
            const float4 d8 = yd[2*j];
            constexpr int ones = 0x01010101;
            int z0 = 0, z1 = 0, z2 = 0, z3 = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                z0 = ggml_cuda_dp4a(a0[k], ones, z0); z1 = ggml_cuda_dp4a(a1[k], ones, z1);
                z2 = ggml_cuda_dp4a(a2[k], ones, z2); z3 = ggml_cuda_dp4a(a3[k], ones, z3);
            }
            z0 *= -32; z1 *= -32; z2 *= -32; z3 *= -32;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t * b = xr + r*row_bytes + 210*j;
                const q6k_i32x4 la = *(const q6k_i32x4 *) (b);
                const q6k_i32x4 lb = *(const q6k_i32x4 *) (b + 32);
                const q6k_i32x4 qh = *(const q6k_i32x4 *) (b + 128 - 32*h);
                const q6k_i32x2 sc = *(const q6k_i32x2 *) (b + 192 - 56*h - 16*s);
                const float     dw = __half2float(*(const half *) (b + 208 - 64*h - 16*s));
                int s0 = z0, s1 = z1, s2 = z2, s3 = z3;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t l = la[k], m = lb[k], q = qh[k];
                    s0 = ggml_cuda_dp4a(( l       & 0x0F0F0F0F) | ((q << 4) & 0x30303030), a0[k], s0);
                    s1 = ggml_cuda_dp4a(( m       & 0x0F0F0F0F) | ((q << 2) & 0x30303030), a1[k], s1);
                    s2 = ggml_cuda_dp4a(((l >> 4) & 0x0F0F0F0F) | ( q       & 0x30303030), a2[k], s2);
                    s3 = ggml_cuda_dp4a(((m >> 4) & 0x0F0F0F0F) | ((q >> 2) & 0x30303030), a3[k], s3);
                }
                const int c0 = (int8_t) (sc.x >> (8*s)), c1 = (int8_t) (sc.x >> (16 + 8*s));
                const int c2 = (int8_t) (sc.y >> (8*s)), c3 = (int8_t) (sc.y >> (16 + 8*s));
                float f = d8.x * (float) (s0*c0);
                f = fmaf(d8.y, (float) (s1*c1), f);
                f = fmaf(d8.z, (float) (s2*c2), f);
                f = fmaf(d8.w, (float) (s3*c3), f);
                acc[r] = fmaf(dw, f, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = warp_reduce_sum<32>(acc[r]);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                dst[blockIdx.y*dst_ch + row0 + r] = acc[r];
            }
        }
    }
}

template <int NB, int R, int W>
static void q6k_mmvq_launch(const void * vx, const q6k_act & act, float * dst, const int nrows, const int nch,
                            const int64_t row_bytes, const int64_t ch_bytes, const int64_t dst_ch, cudaStream_t stream) {
    const dim3 grid((nrows + R*W - 1) / (R*W), nch);
    q6k_mmvq<NB, R, W><<<grid, dim3(32, W), 0, stream>>>((const uint8_t *) vx, act.qs, act.d, dst, nrows, row_bytes,
                                                         ch_bytes, dst_ch);
}

#if defined(Q6K_MMVQ_SWEEP)
#define Q6K_MMVQ_RW(NB) \
    case 0x11: q6k_mmvq_launch<NB, 1, 1>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x12: q6k_mmvq_launch<NB, 1, 2>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x14: q6k_mmvq_launch<NB, 1, 4>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x21: q6k_mmvq_launch<NB, 2, 1>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x22: q6k_mmvq_launch<NB, 2, 2>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x24: q6k_mmvq_launch<NB, 2, 4>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x41: q6k_mmvq_launch<NB, 4, 1>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x42: q6k_mmvq_launch<NB, 4, 2>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x44: q6k_mmvq_launch<NB, 4, 4>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x81: q6k_mmvq_launch<NB, 8, 1>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x82: q6k_mmvq_launch<NB, 8, 2>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x84: q6k_mmvq_launch<NB, 8, 4>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true;
#else
#define Q6K_MMVQ_RW(NB) \
    case 0x12: q6k_mmvq_launch<NB, 1, 2>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x22: q6k_mmvq_launch<NB, 2, 2>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true; \
    case 0x42: q6k_mmvq_launch<NB, 4, 2>(vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream); return true;
#endif

template <int NB>
static bool q6k_mmvq_rw(const int R, const int W, const void * vx, const q6k_act & act, float * dst, const int nrows,
                        const int nch, const int64_t row_bytes, const int64_t ch_bytes, const int64_t dst_ch,
                        cudaStream_t stream) {
    switch (R*16 + W) {
        Q6K_MMVQ_RW(NB)
        default: return false;
    }
}

// dst[c][r] = sum_k x[c][r][k] * act[c][k] for the rows of a Q6_K matrix (row_bytes apart, channels ch_bytes apart);
// returns false (nothing launched) for a K or an R/W it was not compiled for. R/W 0: the configuration chosen per shape.
static bool q6k_mmvq_cuda(const void * vx, const q6k_act & act, float * dst, const int K, const int nrows, const int nch,
                          const int64_t row_bytes, const int64_t ch_bytes, const int64_t dst_ch, cudaStream_t stream,
                          int R = 0, int W = 0) {
    if (R == 0) {
        R = nrows >= 16384 ? 4 : nrows >= 2048 ? 2 : 1;
        W = 2;
    }
    if (nrows % R != 0) {
        return false;
    }
    switch (K / QK_K) {
        case  5: return q6k_mmvq_rw< 5>(R, W, vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream);
        case  9: return q6k_mmvq_rw< 9>(R, W, vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream);
        case 16: return q6k_mmvq_rw<16>(R, W, vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream);
        case 20: return q6k_mmvq_rw<20>(R, W, vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream);
        case 32: return q6k_mmvq_rw<32>(R, W, vx, act, dst, nrows, nch, row_bytes, ch_bytes, dst_ch, stream);
        default: return false;
    }
}

// q6k_mmvq_h: Q6_K x f32 column in f16 arithmetic, no activation quantization. Each step converts the lane's 64
// activations to f16 (round to nearest) once for the R rows; a 6-bit value q becomes the f16 1024 + q by putting
// 0x64 above its byte (v_perm), minus 1056 gives q - 32 exactly, and v_dot2_f32_f16 sums the products (exact in f32)
// in f32. Against ggml's q8_1 activations (8 bits per value, one scale per 32) this keeps 11 bits per activation.
typedef _Float16 q6k_h2 __attribute__((ext_vector_type(2)));

template <int NB, int R, int W>
static __global__ void __launch_bounds__(W*32, 1) q6k_mmvq_h(
        const uint8_t * __restrict__ vx, const float * __restrict__ x, float * __restrict__ dst, const int nrows,
        const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch) {
    const int lane = threadIdx.x;
    const int bl = lane >> 2;
    const int h  = (lane >> 1) & 1;
    const int s  = lane & 1;
    vx += blockIdx.y*ch_bytes + 64*h + 16*s;
    x  += blockIdx.y*sx_ch + 128*h + 16*s;
    const q6k_h2 off = { (_Float16) -1056.0f, (_Float16) -1056.0f };

    for (int row0 = (blockIdx.x*W + threadIdx.y)*R; row0 < nrows; row0 += gridDim.x*W*R) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = 0.0f;
        }
        const uint8_t * xr = vx + (int64_t) row0*row_bytes;
#pragma unroll
        for (int st = 0; st < (NB + 7)/8; ++st) {
            const int j = 8*st + bl;
            if (NB % 8 != 0 && j >= NB) {
                break;
            }
            // activations 256j + 128h + 32g + 16s + [0, 16), as f16 pairs
            q6k_h2 a[4][8];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float4 v = *(const float4 *) (x + 256*j + 32*g + 4*k);
                    a[g][2*k + 0] = q6k_h2 { (_Float16) v.x, (_Float16) v.y };
                    a[g][2*k + 1] = q6k_h2 { (_Float16) v.z, (_Float16) v.w };
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t * b = xr + r*row_bytes + 210*j;   // + 64h + 16s
                const q6k_i32x4 la = *(const q6k_i32x4 *) (b);
                const q6k_i32x4 lb = *(const q6k_i32x4 *) (b + 32);
                const q6k_i32x4 qh = *(const q6k_i32x4 *) (b + 128 - 32*h);
                const q6k_i32x2 sc = *(const q6k_i32x2 *) (b + 192 - 56*h - 16*s);
                const float     dw = __half2float(*(const half *) (b + 208 - 64*h - 16*s));
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t l = la[k], m = lb[k], q = qh[k];
                    const uint32_t q0 = ( l       & 0x0F0F0F0F) | ((q << 4) & 0x30303030);
                    const uint32_t q1 = ( m       & 0x0F0F0F0F) | ((q << 2) & 0x30303030);
                    const uint32_t q2 = ((l >> 4) & 0x0F0F0F0F) | ( q       & 0x30303030);
                    const uint32_t q3 = ((m >> 4) & 0x0F0F0F0F) | ((q >> 2) & 0x30303030);
#define Q6K_H_DOT(qq, sg, g) \
                    sg = __builtin_amdgcn_fdot2(__builtin_bit_cast(q6k_h2, __builtin_amdgcn_perm(0x64646464u, qq, 0x04010400u)) + off, a[g][2*k + 0], sg, false); \
                    sg = __builtin_amdgcn_fdot2(__builtin_bit_cast(q6k_h2, __builtin_amdgcn_perm(0x64646464u, qq, 0x04030402u)) + off, a[g][2*k + 1], sg, false)
                    Q6K_H_DOT(q0, s0, 0);
                    Q6K_H_DOT(q1, s1, 1);
                    Q6K_H_DOT(q2, s2, 2);
                    Q6K_H_DOT(q3, s3, 3);
#undef Q6K_H_DOT
                }
                const float c0 = (int8_t) (sc.x >> (8*s)), c1 = (int8_t) (sc.x >> (16 + 8*s));
                const float c2 = (int8_t) (sc.y >> (8*s)), c3 = (int8_t) (sc.y >> (16 + 8*s));
                acc[r] = fmaf(dw, fmaf(c3, s3, fmaf(c2, s2, fmaf(c1, s1, c0*s0))), acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = warp_reduce_sum<32>(acc[r]);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                dst[blockIdx.y*dst_ch + row0 + r] = acc[r];
            }
        }
    }
}

template <int NB, int R, int W>
static void q6k_mmvq_fq_launch(const void * vx, const float * x, float * dst, const int nrows, const int nch,
                               const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch,
                               const int max_wg, cudaStream_t stream) {
    int grid = (nrows + R*W - 1) / (R*W);
    if (max_wg > 0 && grid > max_wg) {
        grid = max_wg;
    }
    q6k_mmvq_fq<NB, R, W><<<dim3(grid, nch), dim3(32, W), 0, stream>>>((const uint8_t *) vx, x, dst, nrows, row_bytes,
                                                                       ch_bytes, sx_ch, dst_ch);
}

// dst[c][r] = sum_k x[c][r][k] * q8(y[c])[k] for the rows of a Q6_K matrix (rows row_bytes apart, channels ch_bytes
// apart; y f32, channels sx_ch floats apart, 16-byte aligned). The configuration per K and row count was measured on
// an RX 7900 XTX (local/bench-v41/q6k-mmvq.hip sweep). Returns false, launching nothing, for a K or a row count it
// has no configuration for.
static bool q6k_mmvq_fq_cuda(const void * vx, const float * y, float * dst, const int K, const int nrows, const int nch,
                             const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch,
                             cudaStream_t stream) {
    if (row_bytes % 2 != 0 || ch_bytes % 2 != 0 || nrows % 4 != 0) {
        return false;
    }
#define Q6K_FQ(NB, R, W, G) q6k_mmvq_fq_launch<NB, R, W>(vx, y, dst, nrows, nch, row_bytes, ch_bytes, sx_ch, dst_ch, G, stream); return true
    switch (K / QK_K) {
        case  5: Q6K_FQ( 5, 4, 4, 0);
        case  9: Q6K_FQ( 9, 2, 4, 0);
        case 16: Q6K_FQ(16, 2, 4, 0);
        case 20: if (nrows <= 1024) { Q6K_FQ(20, 1, 2, 0); } else if (nrows <= 8192) { Q6K_FQ(20, 2, 4, 0); } else { Q6K_FQ(20, 4, 4, 768); }
        case 32: Q6K_FQ(32, 2, 4, 0);
        default: return false;
    }
#undef Q6K_FQ
}

template <int NB, int R, int W>
static void q6k_mmvq_h_launch(const void * vx, const float * x, float * dst, const int nrows, const int nch,
                              const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch,
                              const int max_wg, cudaStream_t stream) {
    int grid = (nrows + R*W - 1) / (R*W);
    if (max_wg > 0 && grid > max_wg) {
        grid = max_wg;
    }
    q6k_mmvq_h<NB, R, W><<<dim3(grid, nch), dim3(32, W), 0, stream>>>((const uint8_t *) vx, x, dst, nrows, row_bytes,
                                                                      ch_bytes, sx_ch, dst_ch);
}

// the f16 kernel (q6k_mmvq_h), with the same contract as q6k_mmvq_fq_cuda
static bool q6k_mmvq_h_cuda(const void * vx, const float * y, float * dst, const int K, const int nrows, const int nch,
                            const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch,
                            cudaStream_t stream) {
    if (row_bytes % 2 != 0 || ch_bytes % 2 != 0 || nrows % 4 != 0) {
        return false;
    }
#define Q6K_H(NB, R, W, G) q6k_mmvq_h_launch<NB, R, W>(vx, y, dst, nrows, nch, row_bytes, ch_bytes, sx_ch, dst_ch, G, stream); return true
    switch (K / QK_K) {
        case  5: Q6K_H( 5, 4, 4, 0);
        case  9: Q6K_H( 9, 2, 4, 0);
        case 16: Q6K_H(16, 2, 4, 0);
        case 20: if (nrows <= 1024) { Q6K_H(20, 1, 2, 0); } else if (nrows <= 8192) { Q6K_H(20, 2, 4, 0); } else { Q6K_H(20, 4, 4, 768); }
        case 32: Q6K_H(32, 2, 4, 0);
        default: return false;
    }
#undef Q6K_H
}

// q6k_mmvq_f2: q6k_mmvq_f with the groups taken one after the other inside a step, so 16 activation registers are
// live instead of 64 (R rows of weights stay loaded across the four groups)
template <int NB, int R, int W>
static __global__ void __launch_bounds__(W*32, 1) q6k_mmvq_f2(
        const uint8_t * __restrict__ vx, const float * __restrict__ x, float * __restrict__ dst, const int nrows,
        const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch) {
    const int lane = threadIdx.x;
    const int bl = lane >> 2, h = (lane >> 1) & 1, s = lane & 1;
    vx += blockIdx.y*ch_bytes + 64*h + 16*s;
    x  += blockIdx.y*sx_ch + 128*h + 16*s;
    for (int row0 = (blockIdx.x*W + threadIdx.y)*R; row0 < nrows; row0 += gridDim.x*W*R) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.0f;
        if (NB >= 16) asm volatile("" ::: "memory");
        const uint8_t * xr = vx + (int64_t) row0*row_bytes;
#pragma unroll
        for (int st = 0; st < (NB + 7)/8; ++st) {
            const int j = 8*st + bl;
            if (NB % 8 != 0 && j >= NB) break;
            if (st > 0) asm volatile("" ::: "memory");
            q6k_i32x4 la[R], lb[R], qh[R];
            q6k_i32x2 sc[R];
            float dw[R], sg[R][4];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t * b = xr + r*row_bytes + 210*j;
                la[r] = *(const q6k_i32x4 *) (b);
                lb[r] = *(const q6k_i32x4 *) (b + 32);
                qh[r] = *(const q6k_i32x4 *) (b + 128 - 32*h);
                sc[r] = *(const q6k_i32x2 *) (b + 192 - 56*h - 16*s);
                dw[r] = __half2float(*(const half *) (b + 208 - 64*h - 16*s));
            }
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                float4 a[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) a[k] = *(const float4 *) (x + 256*j + 32*g + 4*k);
                const float z = -32.0f * (((a[0].x + a[1].x) + (a[2].x + a[3].x)) + ((a[0].y + a[1].y) + (a[2].y + a[3].y)) +
                                          ((a[0].z + a[1].z) + (a[2].z + a[3].z)) + ((a[0].w + a[1].w) + (a[2].w + a[3].w)));
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float t = z;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t l = la[r][k], m = lb[r][k], q = qh[r][k];
                        const uint32_t qq = g == 0 ? ( l       & 0x0F0F0F0F) | ((q << 4) & 0x30303030)
                                          : g == 1 ? ( m       & 0x0F0F0F0F) | ((q << 2) & 0x30303030)
                                          : g == 2 ? ((l >> 4) & 0x0F0F0F0F) | ( q       & 0x30303030)
                                          :          ((m >> 4) & 0x0F0F0F0F) | ((q >> 2) & 0x30303030);
                        t = fmaf((float) ( qq        & 0xFF), a[k].x, t);
                        t = fmaf((float) ((qq >>  8) & 0xFF), a[k].y, t);
                        t = fmaf((float) ((qq >> 16) & 0xFF), a[k].z, t);
                        t = fmaf((float) ( qq >> 24        ), a[k].w, t);
                    }
                    sg[r][g] = t;
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float c0 = (int8_t) (sc[r].x >> (8*s)), c1 = (int8_t) (sc[r].x >> (16 + 8*s));
                const float c2 = (int8_t) (sc[r].y >> (8*s)), c3 = (int8_t) (sc[r].y >> (16 + 8*s));
                acc[r] = fmaf(dw[r], fmaf(c3, sg[r][3], fmaf(c2, sg[r][2], fmaf(c1, sg[r][1], c0*sg[r][0]))), acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = warp_reduce_sum<32>(acc[r]);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) dst[blockIdx.y*dst_ch + row0 + r] = acc[r];
        }
    }
}

// q6k_mmvq_f4: the first f32 kernel, 4 lanes a block (128-bit loads, 64 weights a lane, 8 blocks a step), otherwise
// as q6k_mmvq_f in mmvq-q6k.cuh (which takes 8 lanes a block): 64 activation registers a step, slower on most shapes.
template <int NB, int R, int W>
static __global__ void __launch_bounds__(W*32, 1) q6k_mmvq_f4(
        const uint8_t * __restrict__ vx, const float * __restrict__ x, float * __restrict__ dst, const int nrows,
        const int64_t row_bytes, const int64_t ch_bytes, const int64_t sx_ch, const int64_t dst_ch) {
    const int lane = threadIdx.x;
    const int bl = lane >> 2;
    const int h  = (lane >> 1) & 1;
    const int s  = lane & 1;
    vx += blockIdx.y*ch_bytes + 64*h + 16*s;
    x  += blockIdx.y*sx_ch + 128*h + 16*s;

    for (int row0 = (blockIdx.x*W + threadIdx.y)*R; row0 < nrows; row0 += gridDim.x*W*R) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = 0.0f;
        }
        if (NB >= 16) {
            // the activations do not depend on the row: without this the compiler loads every step's 64 of them
            // once, before the row loop, and K = 4096 / 8192 run out of registers (measured)
            asm volatile("" ::: "memory");
        }
        const uint8_t * xr = vx + (int64_t) row0*row_bytes;
#pragma unroll
        for (int st = 0; st < (NB + 7)/8; ++st) {
            const int j = 8*st + bl;
            if (NB % 8 != 0 && j >= NB) {
                break;
            }
            if (NB >= 16 && st > 0) {
                // one step's loads in flight a wave: hoisting the later steps' 64 activation registers each made
                // the K = 4096 and 8192 kernels spill (measured), other waves cover the latency instead
                asm volatile("" ::: "memory");
            }
            // activations 256j + 128h + 32g + 16s + [0, 16), and -32 * their sum per g
            float4 a[4][4];
            float  z[4];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    a[g][k] = *(const float4 *) (x + 256*j + 32*g + 4*k);
                }
                const float4 t = make_float4((a[g][0].x + a[g][1].x) + (a[g][2].x + a[g][3].x),
                                             (a[g][0].y + a[g][1].y) + (a[g][2].y + a[g][3].y),
                                             (a[g][0].z + a[g][1].z) + (a[g][2].z + a[g][3].z),
                                             (a[g][0].w + a[g][1].w) + (a[g][2].w + a[g][3].w));
                z[g] = -32.0f * ((t.x + t.y) + (t.z + t.w));
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t * b = xr + r*row_bytes + 210*j;   // + 64h + 16s
                const q6k_i32x4 la = *(const q6k_i32x4 *) (b);
                const q6k_i32x4 lb = *(const q6k_i32x4 *) (b + 32);
                const q6k_i32x4 qh = *(const q6k_i32x4 *) (b + 128 - 32*h);
                const q6k_i32x2 sc = *(const q6k_i32x2 *) (b + 192 - 56*h - 16*s);
                const float     dw = __half2float(*(const half *) (b + 208 - 64*h - 16*s));
                float s0 = z[0], s1 = z[1], s2 = z[2], s3 = z[3];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t l = la[k], m = lb[k], q = qh[k];
                    const uint32_t q0 = ( l       & 0x0F0F0F0F) | ((q << 4) & 0x30303030);
                    const uint32_t q1 = ( m       & 0x0F0F0F0F) | ((q << 2) & 0x30303030);
                    const uint32_t q2 = ((l >> 4) & 0x0F0F0F0F) | ( q       & 0x30303030);
                    const uint32_t q3 = ((m >> 4) & 0x0F0F0F0F) | ((q >> 2) & 0x30303030);
#define Q6K_F4_FMA(qq, sg, g) \
                    sg = fmaf((float) ( qq        & 0xFF), a[g][k].x, sg); \
                    sg = fmaf((float) ((qq >>  8) & 0xFF), a[g][k].y, sg); \
                    sg = fmaf((float) ((qq >> 16) & 0xFF), a[g][k].z, sg); \
                    sg = fmaf((float) ( qq >> 24        ), a[g][k].w, sg)
                    Q6K_F4_FMA(q0, s0, 0);
                    Q6K_F4_FMA(q1, s1, 1);
                    Q6K_F4_FMA(q2, s2, 2);
                    Q6K_F4_FMA(q3, s3, 3);
#undef Q6K_F4_FMA
                }
                const float c0 = (int8_t) (sc.x >> (8*s)), c1 = (int8_t) (sc.x >> (16 + 8*s));
                const float c2 = (int8_t) (sc.y >> (8*s)), c3 = (int8_t) (sc.y >> (16 + 8*s));
                acc[r] = fmaf(dw, fmaf(c3, s3, fmaf(c2, s2, fmaf(c1, s1, c0*s0))), acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = warp_reduce_sum<32>(acc[r]);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                dst[blockIdx.y*dst_ch + row0 + r] = acc[r];
            }
        }
    }
}


// ------------------------------------------------------------------ host side

struct Shape { const char * name; int K, rows, ch, per_token; };
static const Shape SHAPES[] = {
    { "q_a",     5120,   1280, 1, 40 },
    { "q_b",     1280,  32768, 1, 40 },
    { "kv",      5120,    512, 1, 40 },
    { "out_a",   4096,   1024, 8, 40 },
    { "out_b",   8192,   5120, 1, 40 },
    { "gate_up", 5120,   2304, 1, 80 },
    { "down",    2304,   5120, 1, 40 },
    { "head",    5120, 129280, 1,  1 },
};

static size_t wbytes(const Shape & s) { return (size_t) s.rows * s.ch * (s.K / QK_K) * sizeof(block_q6_K); }

struct Buf {   // one shape's device buffers: ncopy weight copies, the activations in both formats, two outputs
    Shape s;
    int ncopy = 1;
    std::vector<uint8_t *> w;
    float * x = nullptr;
    block_q8_1 * y8 = nullptr;
    q6k_act act {};
    float * out_base = nullptr, * out_new = nullptr;
};

static void fill_q6k(std::vector<uint8_t> & h, size_t nblocks, std::mt19937 & rng) {
    h.resize(nblocks * sizeof(block_q6_K));
    for (size_t i = 0; i < h.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(&h[i], &r, std::min<size_t>(4, h.size() - i));
    }
    std::uniform_real_distribution<float> ud(0.0005f, 0.002f);
    for (size_t b = 0; b < nblocks; ++b) {
        block_q6_K * blk = (block_q6_K *) &h[b * sizeof(block_q6_K)];
        blk->d = __float2half(ud(rng));
    }
}

static Buf make_buf(const Shape & s, size_t rotate_bytes, std::mt19937 & rng, bool one_copy = false) {
    Buf b; b.s = s;
    const size_t wb = wbytes(s);
    b.ncopy = one_copy ? 1 : (int) std::max<size_t>(2, (rotate_bytes + wb - 1) / wb);
    std::vector<uint8_t> h;
    for (int c = 0; c < b.ncopy; ++c) {
        fill_q6k(h, wb / sizeof(block_q6_K), rng);
        uint8_t * d; CHECK(hipMalloc(&d, wb)); CHECK(hipMemcpy(d, h.data(), wb, hipMemcpyHostToDevice));
        b.w.push_back(d);
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hx((size_t) s.K * s.ch);
    for (auto & v : hx) v = nd(rng);
    CHECK(hipMalloc(&b.x, hx.size() * 4)); CHECK(hipMemcpy(b.x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&b.y8, (size_t) s.K * s.ch / QK8_1 * sizeof(block_q8_1)));
    CHECK(hipMalloc((void **) &b.act.qs, (size_t) s.K * s.ch));
    CHECK(hipMalloc((void **) &b.act.d,  (size_t) s.K * s.ch / 32 * 4));
    CHECK(hipMalloc(&b.out_base, (size_t) s.rows * s.ch * 4));
    CHECK(hipMalloc(&b.out_new,  (size_t) s.rows * s.ch * 4));
    return b;
}

static void free_buf(Buf & b) {
    for (auto p : b.w) CHECK(hipFree(p));
    CHECK(hipFree(b.x)); CHECK(hipFree(b.y8)); CHECK(hipFree((void *) b.act.qs)); CHECK(hipFree((void *) b.act.d));
    CHECK(hipFree(b.out_base)); CHECK(hipFree(b.out_new));
}

static void quant_base(const Buf & b, hipStream_t st) {
    base_quantize_q8_1<<<dim3((b.s.K + 255) / 256, b.s.ch), 256, 0, st>>>(b.x, b.y8, b.s.K, b.s.K);
}
static void run_base(const Buf & b, int copy, hipStream_t st) {
    const int nb = b.s.K / QK_K;
    base_mmvq_q6k<<<dim3(b.s.rows, b.s.ch), dim3(32, 2), 0, st>>>(b.w[copy], b.y8, b.out_base, b.s.K, nb,
                                                                  nb * b.s.rows, b.s.K / QK8_1, b.s.rows);
}
static void quant_new(const Buf & b, hipStream_t st) {
    q6k_quantize_act_cuda(b.x, b.act, b.s.K, b.s.ch, b.s.K, st);
}
static bool run_new(const Buf & b, int copy, hipStream_t st, int R = 0, int W = 0) {
    const int nb = b.s.K / QK_K;
    return q6k_mmvq_cuda(b.w[copy], b.act, b.out_new, b.s.K, b.s.rows, b.s.ch, (int64_t) nb * sizeof(block_q6_K),
                         (int64_t) nb * b.s.rows * sizeof(block_q6_K), b.s.rows, st, R, W);
}

template <int NW, int RPB>
static void run_ggml_like(const Buf & b, int copy, hipStream_t st) {
    const int nb = b.s.K / QK_K;
    ggml_like_mmvq<NW, RPB><<<dim3(b.s.rows / RPB, b.s.ch), dim3(32, NW), 0, st>>>(b.w[copy], b.y8, b.out_base, b.s.K, nb,
                                                                                  nb * b.s.rows, b.s.K / QK8_1, b.s.rows);
}
static bool run_ggml_like(const Buf & b, int copy, hipStream_t st, int NW, int RPB) {
    switch (NW*16 + RPB) {
        case 0x11: run_ggml_like<1, 1>(b, copy, st); return true;
        case 0x12: run_ggml_like<1, 2>(b, copy, st); return true;
        case 0x14: run_ggml_like<1, 4>(b, copy, st); return true;
        case 0x21: run_ggml_like<2, 1>(b, copy, st); return true;
        case 0x22: run_ggml_like<2, 2>(b, copy, st); return true;
        case 0x24: run_ggml_like<2, 4>(b, copy, st); return true;
        case 0x41: run_ggml_like<4, 1>(b, copy, st); return true;
        case 0x42: run_ggml_like<4, 2>(b, copy, st); return true;
        case 0x44: run_ggml_like<4, 4>(b, copy, st); return true;
        case 0x81: run_ggml_like<8, 1>(b, copy, st); return true;
        case 0x82: run_ggml_like<8, 2>(b, copy, st); return true;
        default: return false;
    }
}

template <int NB, int R, int W>
static void run_flat(const Buf & b, int copy, hipStream_t st) {
    q6k_mmvq_flat<NB, R, W><<<dim3((b.s.rows + R*W - 1) / (R*W), b.s.ch), dim3(32, W), 0, st>>>(b.w[copy], b.act.qs, b.act.d,
        b.out_new, b.s.rows, (int64_t) NB * b.s.rows * sizeof(block_q6_K), b.s.rows);
}
template <int NB>
static bool run_flat_nb(const Buf & b, int copy, hipStream_t st, int R, int W) {
    switch (R*16 + W) {
        case 0x11: run_flat<NB, 1, 1>(b, copy, st); return true;
        case 0x12: run_flat<NB, 1, 2>(b, copy, st); return true;
        case 0x14: run_flat<NB, 1, 4>(b, copy, st); return true;
        case 0x21: run_flat<NB, 2, 1>(b, copy, st); return true;
        case 0x22: run_flat<NB, 2, 2>(b, copy, st); return true;
        case 0x24: run_flat<NB, 2, 4>(b, copy, st); return true;
        case 0x41: run_flat<NB, 4, 1>(b, copy, st); return true;
        case 0x42: run_flat<NB, 4, 2>(b, copy, st); return true;
        case 0x44: run_flat<NB, 4, 4>(b, copy, st); return true;
        case 0x81: run_flat<NB, 8, 1>(b, copy, st); return true;
        case 0x82: run_flat<NB, 8, 2>(b, copy, st); return true;
        case 0x84: run_flat<NB, 8, 4>(b, copy, st); return true;
        default: return false;
    }
}
static bool run_flat(const Buf & b, int copy, hipStream_t st, int R, int W) {
    if (b.s.rows % R) return false;
    switch (b.s.K / QK_K) {
        case  5: return run_flat_nb< 5>(b, copy, st, R, W);
        case  9: return run_flat_nb< 9>(b, copy, st, R, W);
        case 16: return run_flat_nb<16>(b, copy, st, R, W);
        case 20: return run_flat_nb<20>(b, copy, st, R, W);
        case 32: return run_flat_nb<32>(b, copy, st, R, W);
        default: return false;
    }
}

// the fused kernel (q6k_mmvq_fq): f32 activations in, G = the most workgroups (0: one per R*W rows)
template <int NB, int R, int W>
static void run_fq(const Buf & b, int copy, hipStream_t st, int G) {
    int grid = (b.s.rows + R*W - 1) / (R*W);
    if (G > 0) grid = std::min(grid, G);
    q6k_mmvq_fq<NB, R, W><<<dim3(grid, b.s.ch), dim3(32, W), 0, st>>>(b.w[copy], b.x, b.out_new, b.s.rows,
        (int64_t) NB * sizeof(block_q6_K), (int64_t) NB * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows);
}
template <int NB>
static bool run_fq_nb(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    switch (R*16 + W) {
        case 0x12: run_fq<NB, 1, 2>(b, copy, st, G); return true;
        case 0x14: run_fq<NB, 1, 4>(b, copy, st, G); return true;
        case 0x18: run_fq<NB, 1, 8>(b, copy, st, G); return true;
        case 0x22: run_fq<NB, 2, 2>(b, copy, st, G); return true;
        case 0x24: run_fq<NB, 2, 4>(b, copy, st, G); return true;
        case 0x28: run_fq<NB, 2, 8>(b, copy, st, G); return true;
        case 0x42: run_fq<NB, 4, 2>(b, copy, st, G); return true;
        case 0x44: run_fq<NB, 4, 4>(b, copy, st, G); return true;
        case 0x48: run_fq<NB, 4, 8>(b, copy, st, G); return true;
        default: return false;
    }
}
static bool run_fq(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    if (b.s.rows % R) return false;
    switch (b.s.K / QK_K) {
        case  5: return run_fq_nb< 5>(b, copy, st, R, W, G);
        case  9: return run_fq_nb< 9>(b, copy, st, R, W, G);
        case 16: return run_fq_nb<16>(b, copy, st, R, W, G);
        case 20: return run_fq_nb<20>(b, copy, st, R, W, G);
        case 32: return run_fq_nb<32>(b, copy, st, R, W, G);
        default: return false;
    }
}

// the f32 / f16 kernels through the configuration tables ggml uses (q6k_mmvq_f_cuda, q6k_mmvq_h_cuda)
static bool run_f_table(const Buf & b, int copy, hipStream_t st) {
    const int nb = b.s.K / QK_K;
    return q6k_mmvq_f_cuda(b.w[copy], b.x, b.out_new, b.s.K, b.s.rows, b.s.ch, (int64_t) nb * sizeof(block_q6_K),
                           (int64_t) nb * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows, st);
}
static bool run_h_table(const Buf & b, int copy, hipStream_t st) {
    const int nb = b.s.K / QK_K;
    return q6k_mmvq_h_cuda(b.w[copy], b.x, b.out_new, b.s.K, b.s.rows, b.s.ch, (int64_t) nb * sizeof(block_q6_K),
                           (int64_t) nb * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows, st);
}

// the fused kernel through the configuration table ggml uses (q6k_mmvq_fq_cuda)
static bool run_fq_table(const Buf & b, int copy, hipStream_t st) {
    const int nb = b.s.K / QK_K;
    return q6k_mmvq_fq_cuda(b.w[copy], b.x, b.out_new, b.s.K, b.s.rows, b.s.ch, (int64_t) nb * sizeof(block_q6_K),
                            (int64_t) nb * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows, st);
}

// the f16 kernel (q6k_mmvq_h): f32 activations in, G = the most workgroups (0: one per R*W rows)
template <int NB, int R, int W>
static void run_h(const Buf & b, int copy, hipStream_t st, int G) {
    int grid = (b.s.rows + R*W - 1) / (R*W);
    if (G > 0) grid = std::min(grid, G);
    q6k_mmvq_h<NB, R, W><<<dim3(grid, b.s.ch), dim3(32, W), 0, st>>>(b.w[copy], b.x, b.out_new, b.s.rows,
        (int64_t) NB * sizeof(block_q6_K), (int64_t) NB * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows);
}
template <int NB>
static bool run_h_nb(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    switch (R*16 + W) {
        case 0x11: run_h<NB, 1, 1>(b, copy, st, G); return true;
        case 0x12: run_h<NB, 1, 2>(b, copy, st, G); return true;
        case 0x14: run_h<NB, 1, 4>(b, copy, st, G); return true;
        case 0x21: run_h<NB, 2, 1>(b, copy, st, G); return true;
        case 0x22: run_h<NB, 2, 2>(b, copy, st, G); return true;
        case 0x24: run_h<NB, 2, 4>(b, copy, st, G); return true;
        case 0x41: run_h<NB, 4, 1>(b, copy, st, G); return true;
        case 0x42: run_h<NB, 4, 2>(b, copy, st, G); return true;
        case 0x44: run_h<NB, 4, 4>(b, copy, st, G); return true;
        default: return false;
    }
}
static bool run_h(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    if (b.s.rows % R) return false;
    switch (b.s.K / QK_K) {
        case  5: return run_h_nb< 5>(b, copy, st, R, W, G);
        case  9: return run_h_nb< 9>(b, copy, st, R, W, G);
        case 16: return run_h_nb<16>(b, copy, st, R, W, G);
        case 20: return run_h_nb<20>(b, copy, st, R, W, G);
        case 32: return run_h_nb<32>(b, copy, st, R, W, G);
        default: return false;
    }
}

// the 4-lane f32 kernel (q6k_mmvq_f4): f32 activations in, G = the most workgroups (0: one per R*W rows)
template <int NB, int R, int W>
static void run_f(const Buf & b, int copy, hipStream_t st, int G) {
    int grid = (b.s.rows + R*W - 1) / (R*W);
    if (G > 0) grid = std::min(grid, G);
    q6k_mmvq_f4<NB, R, W><<<dim3(grid, b.s.ch), dim3(32, W), 0, st>>>(b.w[copy], b.x, b.out_new, b.s.rows,
        (int64_t) NB * sizeof(block_q6_K), (int64_t) NB * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows);
}
template <int NB>
static bool run_f_nb(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    switch (R*16 + W) {
        case 0x11: run_f<NB, 1, 1>(b, copy, st, G); return true;
        case 0x12: run_f<NB, 1, 2>(b, copy, st, G); return true;
        case 0x14: run_f<NB, 1, 4>(b, copy, st, G); return true;
        case 0x21: run_f<NB, 2, 1>(b, copy, st, G); return true;
        case 0x22: run_f<NB, 2, 2>(b, copy, st, G); return true;
        case 0x24: run_f<NB, 2, 4>(b, copy, st, G); return true;
        case 0x41: run_f<NB, 4, 1>(b, copy, st, G); return true;
        case 0x42: run_f<NB, 4, 2>(b, copy, st, G); return true;
        case 0x44: run_f<NB, 4, 4>(b, copy, st, G); return true;
        default: return false;
    }
}
static bool run_f(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    if (b.s.rows % R) return false;
    switch (b.s.K / QK_K) {
        case  5: return run_f_nb< 5>(b, copy, st, R, W, G);
        case  9: return run_f_nb< 9>(b, copy, st, R, W, G);
        case 16: return run_f_nb<16>(b, copy, st, R, W, G);
        case 20: return run_f_nb<20>(b, copy, st, R, W, G);
        case 32: return run_f_nb<32>(b, copy, st, R, W, G);
        default: return false;
    }
}

template <int NB, int R, int W>
static void run_f2(const Buf & b, int copy, hipStream_t st, int G) {
    int grid = (b.s.rows + R*W - 1) / (R*W);
    if (G > 0) grid = std::min(grid, G);
    q6k_mmvq_f2<NB, R, W><<<dim3(grid, b.s.ch), dim3(32, W), 0, st>>>(b.w[copy], b.x, b.out_new, b.s.rows,
        (int64_t) NB * sizeof(block_q6_K), (int64_t) NB * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows);
}
template <int NB>
static bool run_f2_nb(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    switch (R*16 + W) {
        case 0x11: run_f2<NB, 1, 1>(b, copy, st, G); return true;
        case 0x12: run_f2<NB, 1, 2>(b, copy, st, G); return true;
        case 0x21: run_f2<NB, 2, 1>(b, copy, st, G); return true;
        case 0x22: run_f2<NB, 2, 2>(b, copy, st, G); return true;
        case 0x41: run_f2<NB, 4, 1>(b, copy, st, G); return true;
        case 0x42: run_f2<NB, 4, 2>(b, copy, st, G); return true;
        case 0x44: run_f2<NB, 4, 4>(b, copy, st, G); return true;
        case 0x81: run_f2<NB, 8, 1>(b, copy, st, G); return true;
        case 0x82: run_f2<NB, 8, 2>(b, copy, st, G); return true;
        default: return false;
    }
}
static bool run_f2(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    if (b.s.rows % R) return false;
    switch (b.s.K / QK_K) {
        case  5: return run_f2_nb< 5>(b, copy, st, R, W, G);
        case  9: return run_f2_nb< 9>(b, copy, st, R, W, G);
        case 16: return run_f2_nb<16>(b, copy, st, R, W, G);
        case 20: return run_f2_nb<20>(b, copy, st, R, W, G);
        case 32: return run_f2_nb<32>(b, copy, st, R, W, G);
        default: return false;
    }
}

template <int NB, int R, int W>
static void run_f8(const Buf & b, int copy, hipStream_t st, int G) {
    int grid = (b.s.rows + R*W - 1) / (R*W);
    if (G > 0) grid = std::min(grid, G);
    q6k_mmvq_f<NB, R, W><<<dim3(grid, b.s.ch), dim3(32, W), 0, st>>>(b.w[copy], b.x, b.out_new, b.s.rows,
        (int64_t) NB * sizeof(block_q6_K), (int64_t) NB * b.s.rows * sizeof(block_q6_K), b.s.K, b.s.rows);
}
template <int NB>
static bool run_f8_nb(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    switch (R*16 + W) {
        case 0x11: run_f8<NB, 1, 1>(b, copy, st, G); return true;
        case 0x12: run_f8<NB, 1, 2>(b, copy, st, G); return true;
        case 0x14: run_f8<NB, 1, 4>(b, copy, st, G); return true;
        case 0x21: run_f8<NB, 2, 1>(b, copy, st, G); return true;
        case 0x22: run_f8<NB, 2, 2>(b, copy, st, G); return true;
        case 0x24: run_f8<NB, 2, 4>(b, copy, st, G); return true;
        case 0x41: run_f8<NB, 4, 1>(b, copy, st, G); return true;
        case 0x42: run_f8<NB, 4, 2>(b, copy, st, G); return true;
        case 0x44: run_f8<NB, 4, 4>(b, copy, st, G); return true;
        default: return false;
    }
}
static bool run_f8(const Buf & b, int copy, hipStream_t st, int R, int W, int G) {
    if (b.s.rows % R) return false;
    switch (b.s.K / QK_K) {
        case  5: return run_f8_nb< 5>(b, copy, st, R, W, G);
        case  9: return run_f8_nb< 9>(b, copy, st, R, W, G);
        case 16: return run_f8_nb<16>(b, copy, st, R, W, G);
        case 20: return run_f8_nb<20>(b, copy, st, R, W, G);
        case 32: return run_f8_nb<32>(b, copy, st, R, W, G);
        default: return false;
    }
}

// CPU reference: Q6_K dequantized as ggml's dequantize_row_q6_K, activations as q8 (qs * half(d)), double sums
static void reference(const Buf & b, std::vector<double> & ref, std::vector<double> & mag, bool exact_x = false) {
    const Shape & s = b.s;
    const int nb = s.K / QK_K;
    std::vector<uint8_t> hw(wbytes(s));
    CHECK(hipMemcpy(hw.data(), b.w[0], hw.size(), hipMemcpyDeviceToHost));
    std::vector<int8_t> qs((size_t) s.K * s.ch);
    std::vector<float> d((size_t) s.K * s.ch / 32);
    CHECK(hipMemcpy(qs.data(), b.act.qs, qs.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(d.data(), b.act.d, d.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> xf((size_t) s.K * s.ch);
    CHECK(hipMemcpy(xf.data(), b.x, xf.size() * 4, hipMemcpyDeviceToHost));
    ref.assign((size_t) s.rows * s.ch, 0.0); mag.assign(ref.size(), 0.0);
    const int nth = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t) th.emplace_back([&, t] {
        std::vector<float> wf(s.K);
        for (int64_t rr = t; rr < (int64_t) s.rows * s.ch; rr += nth) {
            const int c = rr / s.rows, r = rr % s.rows;
            const block_q6_K * x = (const block_q6_K *) &hw[((size_t) c * s.rows + r) * nb * sizeof(block_q6_K)];
            for (int i = 0; i < nb; ++i) {
                const float dd = __half2float(x[i].d);
                const uint8_t * ql = x[i].ql; const uint8_t * qh = x[i].qh; const int8_t * sc = x[i].scales;
                float * y = &wf[i * QK_K];
                for (int n = 0; n < QK_K; n += 128) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int8_t q1 = (int8_t) ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int8_t q3 = (int8_t) ((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int8_t q4 = (int8_t) ((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = dd * sc[is + 0] * q1;
                        y[l + 32] = dd * sc[is + 2] * q2;
                        y[l + 64] = dd * sc[is + 4] * q3;
                        y[l + 96] = dd * sc[is + 6] * q4;
                    }
                    y += 128; ql += 64; qh += 32; sc += 8;
                }
            }
            double acc = 0, m = 0;
            for (int k = 0; k < s.K; ++k) {
                const double a = exact_x ? (double) xf[(size_t) c * s.K + k] : (double) qs[(size_t) c * s.K + k] * d[((size_t) c * s.K + k) / 32];
                acc += (double) wf[k] * a; m += std::fabs((double) wf[k] * a);
            }
            ref[rr] = acc; mag[rr] = m;
        }
    });
    for (auto & x : th) x.join();
}

static double max_err(const float * dev, const std::vector<double> & ref, const std::vector<double> & mag) {
    std::vector<float> h(ref.size());
    CHECK(hipMemcpy(h.data(), dev, h.size() * 4, hipMemcpyDeviceToHost));
    double e = 0;
    for (size_t i = 0; i < h.size(); ++i) e = std::max(e, std::fabs(h[i] - ref[i]) / (mag[i] + 1e-30));
    return e;
}

static int cmd_check() {
    std::mt19937 rng(1234);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    int bad = 0;
    for (const Shape & s : SHAPES) {
        Buf b = make_buf(s, 0, rng, true);
        quant_base(b, st); quant_new(b, st);
        // the two activation formats hold the same q8 values
        std::vector<block_q8_1> y8((size_t) s.K * s.ch / QK8_1);
        std::vector<int8_t> qs((size_t) s.K * s.ch); std::vector<float> d(qs.size() / 32);
        CHECK(hipStreamSynchronize(st));
        CHECK(hipMemcpy(y8.data(), b.y8, y8.size() * sizeof(block_q8_1), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(qs.data(), b.act.qs, qs.size(), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(d.data(), b.act.d, d.size() * 4, hipMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < y8.size(); ++i) {
            if (std::memcmp(y8[i].qs, &qs[i * 32], 32) != 0 || __low2float(y8[i].ds) != d[i]) ++diff;
        }
        std::vector<double> ref, mag;
        reference(b, ref, mag);
        run_base(b, 0, st);
        CHECK(hipStreamSynchronize(st));
        const double eb = max_err(b.out_base, ref, mag);
        std::printf("%-8s K %5d rows %6d x%d  q8 blocks differing %zu  ggml max rel err %.2e", s.name, s.K, s.rows, s.ch, diff, eb);
        for (int R : { 1, 2, 4, 8 }) for (int W : { 1, 2, 4 }) {
            CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
            if (!run_new(b, 0, st, R, W)) continue;
            CHECK(hipStreamSynchronize(st));
            const double en = max_err(b.out_new, ref, mag);
            if (en > 1e-5 || !(en == en)) { std::printf("  [R%d W%d: %.2e BAD]", R, W, en); ++bad; }
        }
        for (int R : { 1, 2, 4, 8 }) for (int W : { 1, 2, 4 }) {
            CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
            if (!run_flat(b, 0, st, R, W)) continue;
            CHECK(hipStreamSynchronize(st));
            const double en = max_err(b.out_new, ref, mag);
            if (en > 1e-5 || !(en == en)) { std::printf("  [flat R%d W%d: %.2e BAD]", R, W, en); ++bad; }
        }
        for (int NW : { 1, 2, 4, 8 }) for (int RPB : { 1, 2, 4 }) {
            CHECK(hipMemsetAsync(b.out_base, 0xff, (size_t) s.rows * s.ch * 4, st));
            if (!run_ggml_like(b, 0, st, NW, RPB)) continue;
            CHECK(hipStreamSynchronize(st));
            const double en = max_err(b.out_base, ref, mag);
            if (en > 1e-5 || !(en == en)) { std::printf("  [ggml NW%d RPB%d: %.2e BAD]", NW, RPB, en); ++bad; }
        }
        for (int R : { 1, 2, 4 }) for (int W : { 2, 4, 8 }) for (int G : { 0, 384 }) {
            CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
            if (!run_fq(b, 0, st, R, W, G)) continue;
            CHECK(hipStreamSynchronize(st));
            const double en = max_err(b.out_new, ref, mag);
            if (en > 1e-5 || !(en == en)) { std::printf("  [fq R%d W%d G%d: %.2e BAD]", R, W, G, en); ++bad; }
        }
        {
            run_base(b, 0, st);
            CHECK(hipStreamSynchronize(st));
            std::vector<double> rx, mx;
            reference(b, rx, mx, true);
            const double eg = max_err(b.out_base, rx, mx);   // ggml's output from above
            double eh = 0;
            for (int R : { 1, 2, 4 }) for (int W : { 1, 2, 4 }) for (int G : { 0, 384 }) {
                CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
                if (!run_h(b, 0, st, R, W, G)) continue;
                CHECK(hipStreamSynchronize(st));
                const double en = max_err(b.out_new, rx, mx);
                eh = std::max(eh, en);
                if (en > 3e-4 || !(en == en)) { std::printf("  [h R%d W%d G%d: %.2e BAD]", R, W, G, en); ++bad; }
            }
            double ef = 0;
            for (int R : { 1, 2, 4 }) for (int W : { 1, 2, 4 }) for (int G : { 0, 384 }) {
                CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
                if (!run_f(b, 0, st, R, W, G)) continue;
                CHECK(hipStreamSynchronize(st));
                const double en = max_err(b.out_new, rx, mx);
                ef = std::max(ef, en);
                if (en > 1e-5 || !(en == en)) { std::printf("  [f R%d W%d G%d: %.2e BAD]", R, W, G, en); ++bad; }
            }
            double ef2 = 0;
            for (int R : { 1, 2, 4, 8 }) for (int W : { 1, 2 }) for (int G : { 0, 384 }) {
                CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
                if (!run_f2(b, 0, st, R, W, G)) continue;
                CHECK(hipStreamSynchronize(st));
                const double en = max_err(b.out_new, rx, mx);
                ef2 = std::max(ef2, en);
                if (en > 1e-5 || !(en == en)) { std::printf("  [f2 R%d W%d G%d: %.2e BAD]", R, W, G, en); ++bad; }
            }
            double ef8 = 0;
            for (int R : { 1, 2, 4 }) for (int W : { 1, 2, 4 }) for (int G : { 0, 384 }) {
                CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
                if (!run_f8(b, 0, st, R, W, G)) continue;
                CHECK(hipStreamSynchronize(st));
                const double en = max_err(b.out_new, rx, mx);
                ef8 = std::max(ef8, en);
                if (en > 1e-5 || !(en == en)) { std::printf("  [f8 R%d W%d G%d: %.2e BAD]", R, W, G, en); ++bad; }
            }
            std::printf("  vs exact f32 x: ggml %.2e, q6k_mmvq_h %.2e, q6k_mmvq_f %.2e, f2 %.2e, f8 %.2e", eg, eh, ef, ef2, ef8);
        }
        CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
        if (!run_fq_table(b, 0, st)) { std::printf("  [fq table: no configuration]"); ++bad; }
        CHECK(hipStreamSynchronize(st));
        std::printf("  q6k_mmvq_fq (table) %.2e", max_err(b.out_new, ref, mag));
        CHECK(hipMemsetAsync(b.out_new, 0xff, (size_t) s.rows * s.ch * 4, st));
        run_new(b, 0, st);
        CHECK(hipStreamSynchronize(st));
        std::printf("  q6k_mmvq (default) %.2e\n", max_err(b.out_new, ref, mag));
        if (diff || eb > 1e-5) ++bad;
        free_buf(b);
    }
    std::printf(bad ? "CHECK FAILED (%d)\n" : "check ok\n", bad);
    return bad ? 1 : 0;
}

// The clocks while a measurement runs: a thread reads the card's gpu_metrics (v1.3: current_gfxclk at byte 54,
// current_uclk at 58) every millisecond and averages them; samples with the core clock off (0, idle) are counted apart.
struct Clocks {
    std::string path;
    std::thread th;
    std::atomic<bool> on { false }, quit { false };
    std::atomic<long> n { 0 }, idle { 0 }, gfx { 0 }, mem { 0 };
    Clocks() {
        char bus[64] = {};
        if (hipDeviceGetPCIBusId(bus, sizeof bus, 0) == hipSuccess) {
            for (int c = 0; c < 16 && path.empty(); ++c) {
                const std::string dev = "/sys/class/drm/card" + std::to_string(c) + "/device";
                char real[512] = {};
                if (realpath(dev.c_str(), real) && strcasestr(real, bus)) path = dev + "/gpu_metrics";
            }
        }
        th = std::thread([this] {
            while (!quit) {
                if (on && !path.empty()) {
                    uint8_t b[128] = {};
                    if (FILE * f = std::fopen(path.c_str(), "rb")) {
                        const size_t got = std::fread(b, 1, sizeof b, f); std::fclose(f);
                        uint16_t g, m; std::memcpy(&g, b + 54, 2); std::memcpy(&m, b + 58, 2);
                        if (got >= 120 && b[2] == 1 && b[3] == 3) { if (g == 0) ++idle; else { ++n; gfx += g; mem += m; } }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    ~Clocks() { quit = true; th.join(); }
    std::string level() const {   // power_dpm_force_performance_level (auto, high, ...)
        if (path.empty()) return "?";
        const std::string f = path.substr(0, path.rfind('/')) + "/power_dpm_force_performance_level";
        char buf[32] = {};
        if (FILE * p = std::fopen(f.c_str(), "r")) { std::fgets(buf, sizeof buf, p); std::fclose(p); }
        buf[std::strcspn(buf, "\n")] = 0;
        return buf;
    }
    void start() { n = idle = gfx = mem = 0; on = true; }
    std::string stop() {
        on = false;
        char s[96];
        if (!n) return "clocks n/a";
        std::snprintf(s, sizeof s, "gfx %4ld mem %4ld MHz%s", gfx / n, mem / n, idle ? " (idle samples)" : "");
        return s;
    }
};
static Clocks * g_clk = nullptr;
static int g_gap_us = 0;   // sweep -gap: idle this long after each launch (the clocks then behave as in decode; time
                           // the kernels with a trace, the event times include the gaps)

// us a launch of fn(copy) over n launches, rotating the copies, after >= warm_ms of the same launches (so the clocks
// have settled); *clk = the clocks during the timed launches
template <typename F>
static double time_launches(hipStream_t st, int ncopy, int n, F fn, std::string * clk = nullptr, double warm_ms = 150) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < warm_ms; ++i) {
        fn(i % ncopy);
        if (i % 16 == 15) CHECK(hipStreamSynchronize(st));
    }
    hipEvent_t a, z; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&z));
    if (clk) g_clk->start();
    CHECK(hipEventRecord(a, st));
    for (int i = 0; i < n; ++i) {
        fn(i % ncopy);
        if (g_gap_us > 0) {
            CHECK(hipStreamSynchronize(st));
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < g_gap_us) {}
        }
    }
    CHECK(hipEventRecord(z, st));
    CHECK(hipEventSynchronize(z));
    if (clk) *clk = g_clk->stop();
    float ms; CHECK(hipEventElapsedTime(&ms, a, z));
    CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(z));
    return 1000.0 * ms / n;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static std::string g_fam, g_only;
static bool shape_on(const char * n) { return g_only.empty() || ("," + g_only + ",").find(std::string(",") + n + ",") != std::string::npos; }
static bool fam_on(const char * f) { return ("," + g_fam + ",").find(std::string(",") + f + ",") != std::string::npos; }

static int cmd_shapes(bool sweep, int n) {
    std::mt19937 rng(42);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    double tb_tok = 0, tn_tok = 0, tq_tok = 0, tf_tok = 0, th_tok = 0, tz_tok = 0;
    for (const Shape & s : SHAPES) {
        if (!shape_on(s.name)) continue;
        Buf b = make_buf(s, 768ull << 20, rng);
        shape_marker<<<dim3(7, 1000 + (int) (&s - SHAPES)), 32, 0, st>>>();
        quant_base(b, st); quant_new(b, st);
        const int nl = std::max(20, std::min(n, (int) (4e9 / wbytes(s))));
        const double gb = wbytes(s) / 1e9;
        std::vector<double> vb, vn;
        std::string cb, cn;
        for (int rep = 0; rep < 3; ++rep) {   // interleaved, the median of 3 each
            vb.push_back(time_launches(st, b.ncopy, nl, [&](int c) { run_base(b, c, st); }, &cb));
            vn.push_back(time_launches(st, b.ncopy, nl, [&](int c) { run_new(b, c, st); }, &cn));
        }
        std::vector<double> vq, vf, vh, vz; std::string cq, cf, ch, cz;
        for (int rep = 0; rep < 3; ++rep) {
            vq.push_back(time_launches(st, b.ncopy, nl, [&](int c) { quant_base(b, st); run_base(b, c, st); }, &cq));
            vf.push_back(time_launches(st, b.ncopy, nl, [&](int c) { run_f_table(b, c, st); }, &cf));
            vh.push_back(time_launches(st, b.ncopy, nl, [&](int c) { run_h_table(b, c, st); }, &ch));
            vz.push_back(time_launches(st, b.ncopy, nl, [&](int c) { run_fq_table(b, c, st); }, &cz));
        }
        const double tq = median(vq), tf = median(vf), th = median(vh), tz = median(vz);
        std::printf("%-8s one column, with the quantization: ggml q8_1+mmvq %7.2f us [%s] | f %7.2f us %+6.1f%% | h %7.2f us %+6.1f%% | fq %7.2f us %+6.1f%% [%s]\n",
                    s.name, tq, cq.c_str(), tf, 100.0 * (tf - tq) / tq, th, 100.0 * (th - tq) / tq, tz, 100.0 * (tz - tq) / tq, cf.c_str());
        tq_tok += s.per_token * tq; tf_tok += s.per_token * tf; th_tok += s.per_token * th; tz_tok += s.per_token * tz;
        const double tb = median(vb), tn = median(vn);
        const double tq8 = time_launches(st, 1, 400, [&](int) { quant_base(b, st); });
        const double tq6 = time_launches(st, 1, 400, [&](int) { quant_new(b, st); });
        std::printf("%-8s %6.2f MB x%2d  ggml %7.2f us %5.1f GB/s [%s] | q6k_mmvq %7.2f us %5.1f GB/s [%s] %+6.1f%% | quant %.2f / %.2f us\n",
                    s.name, gb * 1e3, s.per_token, tb, gb / tb * 1e6, cb.c_str(), tn, gb / tn * 1e6, cn.c_str(),
                    100.0 * (tn - tb) / tb, tq8, tq6);
        std::fflush(stdout);
        tb_tok += s.per_token * tb; tn_tok += s.per_token * tn;
        if (sweep) {
            auto line = [&](const char * what, int a, int c2, double t, const std::string & c) {
                std::printf("    %-9s %d %d %8.2f us %6.1f GB/s %+6.1f%% [%s]\n", what, a, c2, t, gb / t * 1e6, 100.0 * (t - tb) / tb, c.c_str());
            };
            if (fam_on("ggml")) for (int NW : { 1, 2, 4, 8 }) for (int RPB : { 1, 2, 4 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_ggml_like(b, cp, st, NW, RPB); }, &c);
                if (ok) line("ggml NW,RPB", NW, RPB, t, c);
            }
            if (fam_on("rows")) for (int R : { 1, 2, 4, 8 }) for (int W : { 1, 2, 4 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_new(b, cp, st, R, W); }, &c);
                if (ok) line("rows R,W", R, W, t, c);
            }
            if (fam_on("flat")) for (int R : { 1, 2, 4, 8 }) for (int W : { 1, 2, 4 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_flat(b, cp, st, R, W); }, &c);
                if (ok) line("flat R,W", R, W, t, c);
            }
            {
                std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { quant_base(b, st); run_base(b, cp, st); }, &c);
                line("ggml q8+mv", 2, 1, t, c);
                const double t2 = time_launches(st, b.ncopy, nl, [&](int cp) { quant_new(b, st); run_new(b, cp, st); }, &c);
                line("rows q8+mv", 0, 0, t2, c);
            }
            if (fam_on("fq")) for (int R : { 1, 2, 4 }) for (int W : { 2, 4, 8 }) for (int G : { 0, 384, 768 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_fq(b, cp, st, R, W, G); }, &c);
                if (ok) { char w[32]; std::snprintf(w, sizeof w, "fq G%d", G); line(w, R, W, t, c); }
            }
            if (fam_on("h")) for (int R : { 1, 2, 4 }) for (int W : { 1, 2, 4 }) for (int G : { 0, 768 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_h(b, cp, st, R, W, G); }, &c);
                if (ok) { char w[32]; std::snprintf(w, sizeof w, "h G%d", G); line(w, R, W, t, c); }
            }
            if (fam_on("f8")) for (int R : { 1, 2, 4 }) for (int W : { 1, 2, 4 }) for (int G : { 0, 768 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_f8(b, cp, st, R, W, G); }, &c);
                if (ok) { char w[32]; std::snprintf(w, sizeof w, "f8 G%d", G); line(w, R, W, t, c); }
            }
            if (fam_on("f2")) for (int R : { 1, 2, 4, 8 }) for (int W : { 1, 2, 4 }) for (int G : { 0, 384, 768 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_f2(b, cp, st, R, W, G); }, &c);
                if (ok) { char w[32]; std::snprintf(w, sizeof w, "f2 G%d", G); line(w, R, W, t, c); }
            }
            if (fam_on("f")) for (int R : { 1, 2, 4 }) for (int W : { 1, 2, 4 }) for (int G : { 0, 768 }) {
                bool ok = true; std::string c;
                const double t = time_launches(st, b.ncopy, nl, [&](int cp) { ok = run_f(b, cp, st, R, W, G); }, &c);
                if (ok) { char w[32]; std::snprintf(w, sizeof w, "f G%d", G); line(w, R, W, t, c); }
            }
            std::fflush(stdout);
        }
        free_buf(b);
    }
    std::printf("per token (matvecs only, launches back to back): ggml %.3f ms, q6k_mmvq %.3f ms (%+.1f%%)\n",
                tb_tok / 1e3, tn_tok / 1e3, 100.0 * (tn_tok - tb_tok) / tb_tok);
    std::printf("per token with the quantization (launches back to back): ggml q8_1+mmvq %.3f ms, q6k_mmvq_f %.3f ms (%+.1f%%), "
                "q6k_mmvq_h %.3f ms (%+.1f%%), q6k_mmvq_fq %.3f ms (%+.1f%%)\n", tq_tok / 1e3, tf_tok / 1e3,
                100.0 * (tf_tok - tq_tok) / tq_tok, th_tok / 1e3, 100.0 * (th_tok - tq_tok) / tq_tok, tz_tok / 1e3,
                100.0 * (tz_tok - tq_tok) / tq_tok);
    return 0;
}

static int cmd_roof(int n) {
    std::mt19937 rng(42);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    int * out; CHECK(hipMalloc((void **) &out, 1 << 20));
    for (const Shape & s : SHAPES) {
        Buf b = make_buf(s, 768ull << 20, rng);
        const int nl = std::max(20, std::min(n, (int) (4e9 / wbytes(s))));
        const double gb = wbytes(s) / 1e9;
        const size_t n16 = wbytes(s) / 16;
        std::printf("%-8s %6.2f MB", s.name, gb * 1e3);
        std::string c;
        for (int blocks : { 96*4, 96*8, 96*16, 96*32 }) {
            const double t = time_launches(st, b.ncopy, nl, [&](int cp) {
                read_bytes<<<blocks, 256, 0, st>>>((const int4 *) b.w[cp], n16, out); }, &c);
            std::printf("  %4d WGs %7.2f us %5.1f GB/s", blocks, t, gb / t * 1e6);
        }
        const double t2 = time_launches(st, b.ncopy, nl, [&](int cp) {
            read_bytes_off2<<<96*16, 256, 0, st>>>(b.w[cp], n16, out); }, &c);
        std::printf("  | +2 B offset, 1536 WGs %7.2f us %5.1f GB/s [%s]", t2, gb / t2 * 1e6, c.c_str());
        std::printf("\n");
        std::fflush(stdout);
        free_buf(b);
    }
    return 0;
}

static int cmd_token(int ntok, int gap_us) {
    std::mt19937 rng(7);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    const int L = 40;
    std::vector<std::vector<Buf>> layers(L);
    for (int l = 0; l < L; ++l)
        for (const Shape & s : SHAPES) {
            if (std::string(s.name) == "head") continue;
            layers[l].push_back(make_buf(s, 0, rng, true));
            if (std::string(s.name) == "gate_up") layers[l].push_back(make_buf(s, 0, rng, true));
        }
    const std::vector<Buf> head_only = { make_buf(SHAPES[7], 0, rng, true) };
    hipEvent_t a, z; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&z));
    static const char * names[] = { "ggml", "q6k_mmvq_f", "q6k_mmvq_h", "q6k_mmvq_fq" };
    for (int rep = 0; rep < 2; ++rep)
    for (int variant = 0; variant < 4; ++variant) {
        double gpu_ms = 0;
        for (int t = -3; t < ntok; ++t) {
            if (t == 0) g_clk->start();
            for (int l = 0; l <= L; ++l) {
                CHECK(hipEventRecord(a, st));
                const std::vector<Buf> & mats = l < L ? layers[l] : head_only;
                for (const Buf & b : mats) {
                    bool ok = true;
                    if (variant == 0) { quant_base(b, st); run_base(b, 0, st); }
                    else if (variant == 1) ok = run_f_table(b, 0, st);
                    else if (variant == 2) ok = run_h_table(b, 0, st);
                    else ok = run_fq_table(b, 0, st);
                    if (!ok) { std::fprintf(stderr, "no %s configuration for %s\n", names[variant], b.s.name); std::exit(1); }
                }
                CHECK(hipEventRecord(z, st));
                CHECK(hipEventSynchronize(z));
                float ms; CHECK(hipEventElapsedTime(&ms, a, z));
                if (t >= 0) gpu_ms += ms;
                if (gap_us > 0) {
                    const auto t0 = std::chrono::steady_clock::now();
                    while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < gap_us) {}
                }
            }
        }
        const std::string clk = g_clk->stop();
        std::printf("%-12s %.3f ms a token (GPU time of the 321 matvecs + quantization, gap %d us) [%s]\n",
                    names[variant], gpu_ms / ntok, gap_us, clk.c_str());
        std::fflush(stdout);
    }
    return 0;
}

int main(int argc, char ** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "shapes";
    int n = 2000, gap = 0;
    std::string fam = "ggml,rows,flat,fq,h,f", only;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "-n")) n = std::atoi(argv[i + 1]);
        if (!std::strcmp(argv[i], "-gap")) gap = std::atoi(argv[i + 1]);
        if (!std::strcmp(argv[i], "-v")) fam = argv[i + 1];
        if (!std::strcmp(argv[i], "-s")) only = argv[i + 1];
    }
    hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
    Clocks clk; g_clk = &clk;
    std::printf("%s, power_dpm_force_performance_level %s\n", clk.path.c_str(), clk.level().c_str());
    std::printf("device 0: %s (%s), %d CUs\n", p.name, p.gcnArchName, p.multiProcessorCount);
    g_fam = fam; g_only = only;
    if (cmd == "sweep") g_gap_us = gap;
    if (cmd == "check") return cmd_check();
    if (cmd == "shapes") return cmd_shapes(false, n);
    if (cmd == "sweep") return cmd_shapes(true, n);
    if (cmd == "roof") return cmd_roof(n);
    if (cmd == "token") return cmd_token(std::min(n, 50), gap);
    std::fprintf(stderr, "usage: q6k-mmvq [check|shapes|sweep|token] [-n N] [-gap US]\n");
    return 2;
}
