// src/kernels/cuda/fork/fork_fmt.cuh - fork formats in the grouped kernels: Fmt<10/11/39> and the STRATA_*_FMTS_FORK lists.
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

template<> struct Fmt<10> { static constexpr int qk = 256, ipb = QI2_K / VDR_Q2_K, step = VDR_Q2_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<11> { static constexpr int qk = 256, ipb = QI3_K / VDR_Q3_K, step = VDR_Q3_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q3_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<39> { static constexpr int qk = 32, ipb = QI_MXFP4 / VDR_MXFP4, step = VDR_MXFP4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_mxfp4_q8_1(v, y, kbx, iqs); } };

// PTQ1_0 (143): the whole 128-weight block per call (4 q8_1 sub-blocks), so ipb 1 and iqs always 0.
#ifndef STRATA_PTQ1_QUARTER   // the whole-block dot (llama-tq's) - default until the quarter split is measured
template<> struct Fmt<143> { static constexpr int qk = 128, ipb = 1, step = 0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_ptq1_0_q8_1(v, y, kbx, iqs); } };
#else                         // -DSTRATA_PTQ1_QUARTER: a quarter block per call (fork_vecdot.cuh), 4x the calls per row, 1/4 the chain (UNMEASURED)
template<> struct Fmt<143> { static constexpr int qk = 128, ipb = 4, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_ptq1_0_q8_1_quarter(v, y, kbx, iqs); } };
#endif
// Fork formats, kept out of upstream's lists above so those stay verbatim (fewer merge conflicts): Q2_K (10,
// DeepSeek-V4 down), Q3_K (11), MXFP4 (39), PTQ1_0 (143, ternary).  Each use site expands the _FORK list too.
#define STRATA_GU_FMTS_FORK(X) X(10) X(11) X(143)
#define STRATA_D_FMTS_FORK(X) X(10) X(11) X(39) X(143)
#define STRATA_MMVQ_FMTS_FORK(X) X(10) X(143)
