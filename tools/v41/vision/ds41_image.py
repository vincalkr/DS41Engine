"""ds41_image.py - DeepSeek-V4.1's image preprocessing without torch, and the SVP1 record the engine encodes.

A port of inference/image_processor.py (deepseek-ai/DeepSeek-V4.1-Flash-1017) to PIL +
numpy: the same resize plan (plan_image_grid / safe_resize / solve_resize_ratio), the same ImageOps.pad to the plan
with gray 127, the same normalization ((x / 255 - 0.5) / 0.5 in float32, then rounded to bfloat16) and the same patch
order ([n_vit_h * n_vit_w, 3 * 14 * 14], channel-major inside a patch).  On the five reference images of
the patches are bit-identical to load_image's.

SVP1 record (ds41_generate's GENI, VisionInput): int32 {0x31505653 'SVP1', vit_h, vit_w, llm_h, llm_w, 588},
uint64 key, float32 patches[vit_h * vit_w][588], little-endian.  The span the image takes in the prompt is
llm_h * (llm_w + 1) + 2 image tokens.
"""
import io
import math
import struct

import numpy as np
from PIL import Image, ImageOps

PATCH, RATIO, MAX_TOKENS, MIN_PIXELS = 14, 3, 1024, 295936   # config.json vision_* (1017)
SVP1 = 0x31505653


def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height, best_width):
    return math.ceil((best_height // PATCH) / RATIO), math.ceil((best_width // PATCH) / RATIO)


def solve_resize_ratio(height, width, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = PATCH * RATIO
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / PATCH) * PATCH, math.floor(width * beta / PATCH) * PATCH


def plan_image_grid(width, height, max_n_token=MAX_TOKENS, min_pixels=MIN_PIXELS):
    """-> (n_llm_h, n_llm_w, best_height, best_width), as image_processor.plan_image_grid (max_wh_ratio None)."""
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / PATCH) * PATCH
    best_height = math.ceil(height / PATCH) * PATCH
    n_llm_h, n_llm_w = llm_grid(best_height, best_width)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def _bf16(x):
    """float32 -> the nearest bfloat16 (round to nearest even), as float32 - torch's .to(torch.bfloat16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def load_patches(data: bytes, max_n_token=MAX_TOKENS):
    """-> (patches float32 [n_vit_h * n_vit_w, 588], n_vit_h, n_vit_w, n_llm_h, n_llm_w), as load_image."""
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, max_n_token)
    n_vit_h, n_vit_w = best_height // PATCH, best_width // PATCH
    image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    x = _bf16((x - np.float32(0.5)) / np.float32(0.5))
    patches = x.reshape(3, n_vit_h, PATCH, n_vit_w, PATCH).transpose(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * PATCH * PATCH)
    return np.ascontiguousarray(patches, dtype="<f4"), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def svp1_record(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, key: int) -> bytes:
    if not 0 < key < 1 << 64:
        raise ValueError("the image key must be a nonzero 64-bit value")
    head = struct.pack("<6iQ", SVP1, n_vit_h, n_vit_w, n_llm_h, n_llm_w, 3 * PATCH * PATCH, key)
    return head + np.ascontiguousarray(patches, dtype="<f4").tobytes()
