// nvme-idle-read FILE GAP_MS KEEPALIVE(0 none, 1 a 4 KiB read every 0.5 ms, 2 a CPU tick every 0.5 ms) SPINNERS (gcc -O2 -pthread): one 9.7 MiB O_DIRECT blob read (8 threads, created per blob like the engine)
// every GAP_MS, at random offsets; KEEPALIVE=1 adds a 4 KiB read every 0.5 ms; SPINNERS busy-wait threads with
// _mm_pause (the CPU expert pool parked).  Prints the mean / median read latency over 60 blobs.
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int fd; static const size_t blob = 10171392; static off_t fsize; static atomic_int stop;
typedef struct { char* buf; off_t off; size_t len; } Job;
static void* rd(void* p) { Job* j = p; size_t got = 0; while (got < j->len) { ssize_t k = pread(fd, j->buf + got, j->len - got, j->off + got); if (k <= 0) exit(1); got += k; } return NULL; }
static void* keep(void* p) { (void) p; char* b; if (posix_memalign((void**) &b, 4096, 4096)) return NULL; unsigned s = 7;
    while (!stop) { off_t o = ((off_t) (rand_r(&s) / (double) RAND_MAX * (fsize - 8192))) & ~(off_t) 4095; if (pread(fd, b, 4096, o) < 0) break; usleep(500); } return NULL; }
static void* spin(void* p) { (void) p; while (!stop) _mm_pause(); return NULL; }
static void* tick(void* p) { (void) p; while (!stop) usleep(500); return NULL; }
static int cmp(const void* a, const void* b) { double x = *(double*) a, y = *(double*) b; return x < y ? -1 : x > y; }
int main(int argc, char** argv) {
    fd = open(argv[1], O_RDONLY | O_DIRECT); struct stat st; fstat(fd, &st); fsize = st.st_size;
    const double gap = atof(argv[2]) / 1000; const int ka = atoi(argv[3]), ns = atoi(argv[4]);
    pthread_t kt, sp[64];
    if (ka == 1) pthread_create(&kt, NULL, keep, NULL);
    if (ka == 2) pthread_create(&kt, NULL, tick, NULL);
    for (int i = 0; i < ns; ++i) pthread_create(&sp[i], NULL, spin, NULL);
    char* mem; if (posix_memalign((void**) &mem, 4096, blob)) return 1;
    srand(99); double lat[60];
    for (int r = 0; r < 60; ++r) {
        const double t0 = now();
        off_t base = ((off_t) ((double) rand() / RAND_MAX * (double) (fsize - (off_t) blob - 8192))) & ~(off_t) 4095;
        pthread_t th[8]; Job jb[8]; const size_t part = (blob / 8 + 4095) & ~(size_t) 4095;
        for (int t = 0; t < 8; ++t) { size_t o = t * part; jb[t] = (Job){mem + o, base + (off_t) o, o + part > blob ? blob - o : part}; pthread_create(&th[t], NULL, rd, &jb[t]); }
        for (int t = 0; t < 8; ++t) pthread_join(th[t], NULL);
        lat[r] = 1000 * (now() - t0);
        while (now() - t0 < gap) usleep(200);
    }
    stop = 1;
    double m = 0; for (int r = 0; r < 60; ++r) m += lat[r]; m /= 60; qsort(lat, 60, sizeof(double), cmp);
    printf("gap %4.0f ms, keepalive %d, %2d spinners: mean %.2f ms, median %.2f, p90 %.2f\n", gap * 1000, ka, ns, m, lat[30], lat[54]);
    return 0;
}
