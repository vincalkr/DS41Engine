// membw GIB HUGE(0/1) (gcc -O3 -mavx2 -fopenmp): multi-threaded streaming read bandwidth of an anonymous buffer, 4 KiB pages or THP.
#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static long anon_huge_kb(void) {
    FILE* f = fopen("/proc/self/smaps_rollup", "r"); char l[256]; long kb = -1;
    while (f && fgets(l, sizeof l, f)) if (sscanf(l, "AnonHugePages: %ld kB", &kb) == 1) break;
    if (f) fclose(f); return kb;
}
int main(int argc, char** argv) {
    const size_t n = (size_t) (argc > 1 ? atol(argv[1]) : 8) << 30;
    const int huge = argc > 2 ? atoi(argv[2]) : 0;
    char* raw = mmap(NULL, n + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char* a = (char*) (((uintptr_t) raw + (2 << 20) - 1) & ~(uintptr_t) ((2 << 20) - 1));
    madvise(a, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    double t0 = now();
    memset(a, 1, n);
    printf("pages: %s, AnonHugePages %ld MiB, fill %.1f s\n", huge ? "THP" : "4 KiB", anon_huge_kb() / 1024, now() - t0);
    const int ths[] = {6, 12, 24};
    for (int ti = 0; ti < 3; ++ti) {
        omp_set_num_threads(ths[ti]);
        double best = 0; long long sink = 0;
        for (int rep = 0; rep < 5; ++rep) {
            const double s = now();
            long long tot = 0;
#pragma omp parallel reduction(+ : tot)
            {
                const int t = omp_get_thread_num(), T = omp_get_num_threads();
                const size_t per = n / T / 64 * 64;
                const __m256i* p = (const __m256i*) (a + (size_t) t * per);
                __m256i x0 = _mm256_setzero_si256(), x1 = x0, x2 = x0, x3 = x0;
                for (size_t i = 0; i < per / 32; i += 4) {
                    x0 = _mm256_xor_si256(x0, _mm256_load_si256(p + i));
                    x1 = _mm256_xor_si256(x1, _mm256_load_si256(p + i + 1));
                    x2 = _mm256_xor_si256(x2, _mm256_load_si256(p + i + 2));
                    x3 = _mm256_xor_si256(x3, _mm256_load_si256(p + i + 3));
                }
                x0 = _mm256_xor_si256(_mm256_xor_si256(x0, x1), _mm256_xor_si256(x2, x3));
                tot += _mm256_extract_epi64(x0, 0);
            }
            const double gbs = (double) n / (now() - s) / 1e9;
            if (gbs > best) best = gbs;
            sink += tot;
        }
        printf("  %2d threads: %.1f GB/s (best of 5)%s\n", ths[ti], best, sink == 42 ? " " : "");
    }
    return 0;
}
