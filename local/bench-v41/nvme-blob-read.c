// nvme-blob-read FILE (gcc -O2 -pthread): O_DIRECT reads of 9.7 MiB "experts" at random 4 KiB-aligned offsets, the way the engine's file tier
// reads one blob (T threads, each preading its contiguous part), and with K blobs in flight at once.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int fd; static size_t blob = 10171392; static off_t fsize;
typedef struct { char* buf; off_t off; size_t len; } Job;
static void* rd(void* p) {
    Job* j = p; size_t got = 0;
    while (got < j->len) { ssize_t k = pread(fd, j->buf + got, j->len - got, j->off + got); if (k <= 0) { perror("pread"); exit(1); } got += k; }
    return NULL;
}
int main(int argc, char** argv) {
    fd = open(argv[1], O_RDONLY | O_DIRECT); if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st); fsize = st.st_size;
    srand(12345);
    const int Ts[] = {1, 8, 16, 32, 64}; const int Ks[] = {1, 2, 4};
    char* mem; posix_memalign((void**) &mem, 4096, blob * 4);
    for (int ki = 0; ki < 3; ++ki) for (int ti = 0; ti < 5; ++ti) {
        const int K = Ks[ki], T = Ts[ti]; const int rounds = 24 / K;
        double t0 = now();
        for (int r = 0; r < rounds; ++r) {
            pthread_t th[256]; Job jb[256]; int n = 0;
            for (int k = 0; k < K; ++k) {
                off_t base = ((off_t) ((double) rand() / RAND_MAX * (double) (fsize - (off_t) blob - 8192))) & ~(off_t) 4095;
                size_t part = (blob / T + 4095) & ~(size_t) 4095;
                for (int t = 0; t < T; ++t) {
                    size_t o = (size_t) t * part; if (o >= blob) break;
                    jb[n] = (Job){mem + (size_t) k * blob + o, base + (off_t) o, (o + part > blob ? blob - o : part)};
                    pthread_create(&th[n], NULL, rd, &jb[n]); ++n;
                }
            }
            for (int i = 0; i < n; ++i) pthread_join(th[i], NULL);
        }
        const double dt = now() - t0; const double nb = (double) rounds * K;
        printf("K=%d blobs in flight, T=%2d threads each: %.2f ms per round, %.2f GB/s\n", K, T, 1000 * dt / rounds, nb * blob / dt / 1e9);
    }
    return 0;
}
