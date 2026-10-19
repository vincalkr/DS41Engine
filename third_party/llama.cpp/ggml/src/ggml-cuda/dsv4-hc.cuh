#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_fake_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv41_hc_mixes(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
