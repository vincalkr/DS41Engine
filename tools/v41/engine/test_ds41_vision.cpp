// tools/v41/engine/test_ds41_vision.cpp - the image encoder (Ds41Vision) against a reference span.
//
//   test_ds41_vision --vision models/vision-1017/DeepSeek-V4.1-Flash-1017-Vision.gguf [--device N] [--reps K]
//                    REFDIR [REFDIR ...]
//
// REFDIR holds what DeepSeek's own vision.py wrote for one image (float32, little-endian): patches.f32 [N, 588] (the
// image_processor.load_image patch grid), span.f32 [rows, 5120] (merge_image_embeddings) and meta.json
// ({"n_vit_h", "n_vit_w", "n_llm_h", "n_llm_w", ...}).  Prints per image the cosine of every aligner row against
// the reference (min / mean), the relative L2 error, the encode time and the device memory.  --device: the ggml GPU
// device index (default 0); -1 = the CPU backend.
#include "ds41_vision.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static bool read_f32(const std::string & p, std::vector<float> & v) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    v.resize((size_t) f.tellg() / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize) (v.size() * 4));
    return (bool) f;
}

static int json_int(const std::string & s, const char * key) {
    const size_t k = s.find(std::string("\"") + key + "\"");
    if (k == std::string::npos) return -1;
    return std::atoi(s.c_str() + s.find(':', k) + 1);
}

int main(int argc, char ** argv) {
    std::string gguf;
    int device = 0, reps = 1;
    std::vector<std::string> dirs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--vision" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else dirs.push_back(a);
    }
    if (gguf.empty() || dirs.empty()) { std::fprintf(stderr, "usage: see the file header\n"); return 2; }
    ggml_backend_t be = nullptr;
    if (device < 0) {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else {
        int seen = 0;
        for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(d);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (seen++ == device) { be = ggml_backend_dev_init(dev, nullptr); break; }
        }
    }
    if (!be) { std::fprintf(stderr, "no backend for device %d\n", device); return 1; }
    std::string err;
    Ds41Vision vis;
    if (!vis.open(gguf, err) || !vis.load(be, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    int bad = 0;
    for (const std::string & d : dirs) {
        std::ifstream mf(d + "/meta.json");
        const std::string meta((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
        Ds41VisionGrid g;
        g.vit_h = json_int(meta, "n_vit_h"); g.vit_w = json_int(meta, "n_vit_w");
        g.llm_h = json_int(meta, "n_llm_h"); g.llm_w = json_int(meta, "n_llm_w");
        std::vector<float> patches, ref, span;
        if (!read_f32(d + "/patches.f32", patches) || !read_f32(d + "/span.f32", ref) ||
            patches.size() != (size_t) g.patches() * 588 || ref.size() != (size_t) g.span_rows() * 5120) {
            std::fprintf(stderr, "%s: bad reference files\n", d.c_str());
            return 1;
        }
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            if (!vis.encode(patches.data(), g, span, err)) { std::fprintf(stderr, "%s: %s\n", d.c_str(), err.c_str()); return 1; }
            best = std::min(best, vis.last_ms());
        }
        // per row: cosine and relative error; the aligner rows (not the delimiters, which are copied) are the test
        double cmin = 1, csum = 0, e2 = 0, r2 = 0;
        int nimg = 0;
        for (int row = 0; row < g.span_rows(); ++row) {
            const bool delim = row == 0 || row == g.span_rows() - 1 || (row - 1) % (g.llm_w + 1) == g.llm_w;
            const float * a = span.data() + (size_t) row * 5120, * b = ref.data() + (size_t) row * 5120;
            double ab = 0, aa = 0, bb = 0, ee = 0;
            for (int i = 0; i < 5120; ++i) { ab += (double) a[i] * b[i]; aa += (double) a[i] * a[i]; bb += (double) b[i] * b[i]; ee += ((double) a[i] - b[i]) * ((double) a[i] - b[i]); }
            if (delim) { if (ee != 0) ++bad; continue; }
            const double c = ab / std::sqrt(aa * bb);
            cmin = std::min(cmin, c); csum += c; e2 += ee; r2 += bb; ++nimg;
        }
        std::printf("%s: %dx%d patches -> %d rows: cosine min %.6f mean %.6f, rel L2 %.2e | %.0f ms (best of %d), "
                    "weights %.2f GiB, compute buffer %.0f MiB\n", d.c_str(), g.vit_h, g.vit_w, g.span_rows(), cmin,
                    csum / nimg, std::sqrt(e2 / r2), best, reps, vis.weight_bytes() / 1073741824.0, vis.compute_bytes() / 1048576.0);
        if (cmin < 0.99) ++bad;
    }
    vis.unload();
    ggml_backend_free(be);
    return bad ? 1 : 0;
}
