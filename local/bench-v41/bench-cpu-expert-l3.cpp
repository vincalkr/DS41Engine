// local/bench-v41/bench-cpu-expert-l3.cpp - the CPU expert kernels reading from L3 against reading from DRAM.
//
// Answers one question before an L3 prefetch of predicted experts is built: is the CPU tier limited by the RAM (then
// experts already in L3 compute much faster) or by its own arithmetic (then they do not)?  Each "layer" computes E
// experts the way the engine's pool does (every thread a fixed slice of the gate/up rows, a barrier, then a slice of
// the down rows), with the engine's native kernels (native_gu_rows / native_down_rows) on real bytes of the pack:
//   cold: the E experts of each layer are new (a ring of NB blobs, far larger than L3)
//   hot:  the same E experts every layer (their bytes stay in L3; each thread keeps its own slice)
//
//   bench-cpu-expert-l3 SHARD GU_TYPE D_TYPE [THREADS=12] [E=2] [NB=96] [LAYERS=2000]
// ggml types: 29 IQ1_M, 16 IQ2_XXS, 17 IQ2_XS.  Build: local/bench-v41/bench-cpu-expert-l3.sh
#include "strata/kernels/cpu/native_expert.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <random>
#include <sched.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace strata::kernels::cpu;

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s SHARD GU_TYPE D_TYPE [THREADS] [E] [NB] [LAYERS]\n", argv[0]); return 2; }
    const int gu = std::atoi(argv[2]), dt = std::atoi(argv[3]);
    const int T = argc > 4 ? std::atoi(argv[4]) : 12, E = argc > 5 ? std::atoi(argv[5]) : 2;
    const int NB = argc > 6 ? std::atoi(argv[6]) : 96, L = argc > 7 ? std::atoi(argv[7]) : 2000;
    NativeFmt f;
    std::string err;
    if (!native_fmt(gu, dt, 5120, 2304, f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); return 1; }
    // real quantized bytes: NB blob-sized windows from the middle of a shard (expert data), aligned like the arena
    const size_t B = (f.bytes + 4095) & ~(size_t) 4095;
    uint8_t * pool = (uint8_t *) std::aligned_alloc(4096, B * (size_t) NB);
    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { std::perror(argv[1]); return 1; }
    for (int i = 0; i < NB; ++i)
        if (pread(fd, pool + B * (size_t) i, f.bytes, (off_t) (8ull << 30) + (off_t) B * i) != (ssize_t) f.bytes) { std::perror("pread"); return 1; }
    close(fd);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) f.n_embd), h((size_t) f.n_ff);
    for (auto & v : x) v = nd(rng);
    for (auto & v : h) v = 0.1f * nd(rng);
    std::vector<uint8_t> act(f.act_bytes), hq(f.h_bytes);
    native_quant_act(f, x.data(), act.data());
    native_quant_h(f, h.data(), hq.data());
    std::vector<std::vector<float>> ff((size_t) E, std::vector<float>((size_t) f.n_ff)), out((size_t) E, std::vector<float>((size_t) f.n_embd));

    std::atomic<int> bar{0};
    std::atomic<int> gen{0};
    auto barrier = [&](int & local_gen) {
        const int g = local_gen;
        if (bar.fetch_add(1) == T - 1) { bar.store(0); gen.store(g + 1, std::memory_order_release); }
        else while (gen.load(std::memory_order_acquire) == g) {}
        local_gen = g + 1;
    };
    auto run = [&](bool hot) {
        std::vector<std::thread> th;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < T; ++t)
            th.emplace_back([&, t, hot] {
                cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(t, &cs);
                pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
                int lg = 0;
                const int g0 = (int) (f.n_ff * t / T), g1 = (int) (f.n_ff * (t + 1) / T);
                const int d0 = (int) (f.n_embd * t / T), d1 = (int) (f.n_embd * (t + 1) / T);
                const void * a1[1] = { act.data() };
                const void * h1[1] = { hq.data() };
                for (int l = 0; l < L; ++l) {
                    for (int e = 0; e < E; ++e) {
                        const uint8_t * blob = pool + B * (size_t) (hot ? e : (l * E + e) % NB);
                        float * o[1] = { ff[(size_t) e].data() };
                        native_gu_rows(f, blob, a1, 1, o, g0, g1);
                    }
                    barrier(lg);
                    for (int e = 0; e < E; ++e) {
                        const uint8_t * blob = pool + B * (size_t) (hot ? e : (l * E + e) % NB);
                        float * o[1] = { out[(size_t) e].data() };
                        native_down_rows(f, blob, h1, 1, o, d0, d1);
                    }
                    barrier(lg);
                }
            });
        for (auto & x : th) x.join();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    run(true);   // warm-up
    const double cold = run(false), hot = run(true);
    const double n = (double) L * E;
    std::printf("gu %d d %d, %d threads, %d experts a layer, blob %.2f MB: cold %.1f us/expert (%.1f GB/s), hot %.1f us/expert (%.1f GB/s), hot/cold %.2fx\n",
                gu, dt, T, E, f.bytes / 1e6, cold / n * 1e6, n * f.bytes / cold / 1e9, hot / n * 1e6,
                n * f.bytes / hot / 1e9, cold / hot);
    return 0;
}
