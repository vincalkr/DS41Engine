// src/kernels/cuda/fork/fork_helpers.cuh - fork: byte-aligned int load and E8M0 -> float (MXFP4).
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

// byte-aligned (MXFP4's 17-byte blocks): llama.cpp's get_int_b1
__device__ __forceinline__ int get_int_b1(const void* x, const int& i32) {
    const uint8_t* x8 = (const uint8_t*) x;
    int x32 = x8[4 * i32 + 0] << 0;
    x32 |= x8[4 * i32 + 1] << 8;
    x32 |= x8[4 * i32 + 2] << 16;
    x32 |= x8[4 * i32 + 3] << 24;
    return x32;
}
// E8M0 (MXFP4's shared block exponent) -> float: 2^(e-127); e == 0 is the denormal 2^-127 (llama.cpp's
// ggml_cuda_e8m0_to_fp32)
__device__ __forceinline__ float e8m0_to_fp32(uint8_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : (uint32_t) e << 23);
}
