// src/kernels/cuda/fork/fork_gu_mixed.cuh - native_gu_mixed_kernel: gate and up in different formats.
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

// native_gu_kernel with gate and up in DIFFERENT formats (a pack that picks a format per tensor).  A
// warp's row is a gate row or an up row; with n_ff % GU_ROWS == 0 a whole block is one or the other, so the branch is
// block-uniform.  Same per-row arithmetic as native_gu_kernel<T> for each half.
template<int TG, int TU>
__global__ void __launch_bounds__(256) native_gu_mixed_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_tok,
                                                              const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                              float* __restrict__ gate, float* __restrict__ up) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = is_up ? L.up_off + (size_t) r * L.up_row : (size_t) r * L.gu_row;
    const int xb = (int) (L.n_embd / 32);
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const block_q8_1* x = xq + (size_t) ent_tok[e] * xb;
            const float s = is_up ? row_dot<TU>(wr, x, (int) (L.n_embd / Fmt<TU>::qk), lane)
                                  : row_dot<TG>(wr, x, (int) (L.n_embd / Fmt<TG>::qk), lane);
            if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
        }
    }
}
