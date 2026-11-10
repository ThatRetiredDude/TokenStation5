// How fast a drive reads the way an expert cache would: direct (O_DIRECT,
// no page cache) reads of one block size at random offsets in model files,
// from 1 to 16 threads at once, plus one long sequential read. Read-only.
//
//   cc -O2 -pthread -o ssd_read engine/tests/ssd_read.c
//   ./ssd_read ~/.local/share/tokenstation/models/*.gguf
//
// Each line: block size, threads, MB/s (10^6 bytes), reads/s, the median and
// 99th-percentile time of one read, and this process's CPU use. The block
// sizes are the experts of the models an expert cache would stream: one
// matrix of a Qwen3 30B A3B expert in Q4_K is 0.84 MiB (gate, up) to
// 1.23 MiB (down, Q6_K), a whole expert 2.73 MiB; one gpt-oss 120B expert is
// 12.6 MiB. TS5_SSD_SECONDS sets each test's length (default 4).
//
// It first prints the drive behind the first file (model, transport, the
// USB link's speed and driver, the queue settings) and whether an NVMe
// drive is present, and it won't start while a ts5 process is running
// (TS5_SSD_FORCE=1 runs anyway): the reads would slow a benchmark down.
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <sys/sysmacros.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAXF 64
#define MAXT 32
#define MAXLAT 200000

static int nf;
static int fds[MAXF];
static uint64_t sizes[MAXF];
static uint64_t total_size;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double cpu_seconds(void) {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 + u.ru_stime.tv_sec + u.ru_stime.tv_usec * 1e-6;
}

// xorshift64*, one per thread.
static uint64_t rnd(uint64_t *s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return *s * 2685821657736338717ull;
}

struct job {
    size_t block;
    int seq, id, nthreads;
    double deadline;
    uint64_t limit;          // stop after this many bytes over all threads
    uint64_t bytes, reads;
    float *lat;              // ms per read
    int nlat, maxlat;
    int err;
};

static uint64_t done_bytes;  // all threads, for the limit
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static void *worker(void *arg) {
    struct job *j = arg;
    void *buf;
    if (posix_memalign(&buf, 1 << 21, j->block)) { j->err = ENOMEM; return 0; }
    uint64_t s = 0x9e3779b97f4a7c15ull * (uint64_t)(j->id + 1);
    // Sequential: thread i reads blocks i, i + n, i + 2n, ... of the
    // largest file, so n threads keep n requests in flight in order.
    int big = 0;
    for (int i = 1; i < nf; i++) if (sizes[i] > sizes[big]) big = i;
    uint64_t next = (uint64_t)j->id * j->block;
    while (now() < j->deadline) {
        int f;
        uint64_t off;
        if (j->seq) {
            f = big;
            if (next + j->block > sizes[f]) break;
            off = next;
            next += (uint64_t)j->nthreads * j->block;
        } else {
            // A file in proportion to its size, then a 4 KiB-aligned offset.
            uint64_t at = rnd(&s) % total_size;
            for (f = 0; f < nf - 1 && at >= sizes[f]; f++) at -= sizes[f];
            if (sizes[f] < j->block + 4096) continue;
            off = (rnd(&s) % ((sizes[f] - j->block) / 4096)) * 4096;
        }
        const double t0 = now();
        size_t got = 0;
        while (got < j->block) {
            ssize_t r = pread(fds[f], (char *)buf + got, j->block - got, (off_t)(off + got));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { j->err = r < 0 ? errno : EIO; free(buf); return 0; }
            got += (size_t)r;
        }
        const double ms = (now() - t0) * 1e3;
        if (j->nlat < j->maxlat) j->lat[j->nlat++] = (float)ms;
        j->bytes += got;
        j->reads++;
        pthread_mutex_lock(&mu);
        done_bytes += got;
        const int stop = done_bytes >= j->limit;
        pthread_mutex_unlock(&mu);
        if (stop) break;
    }
    free(buf);
    return 0;
}

static int cmpf(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

static void run(const char *name, size_t block, int threads, int seq, double seconds, uint64_t limit) {
    struct job jobs[MAXT];
    pthread_t th[MAXT];
    float *lat = malloc(sizeof(float) * MAXLAT);
    const int per = MAXLAT / threads;
    done_bytes = 0;
    const double t0 = now(), c0 = cpu_seconds();
    for (int i = 0; i < threads; i++) {
        jobs[i] = (struct job){block, seq, i, threads, t0 + seconds, limit, 0, 0, lat + (size_t)i * per, 0, per, 0};
        pthread_create(&th[i], 0, worker, &jobs[i]);
    }
    uint64_t bytes = 0, reads = 0;
    int n = 0, err = 0;
    for (int i = 0; i < threads; i++) {
        pthread_join(th[i], 0);
        bytes += jobs[i].bytes;
        reads += jobs[i].reads;
        if (jobs[i].err) err = jobs[i].err;
        // Pack the latencies together for the percentiles.
        memmove(lat + n, jobs[i].lat, sizeof(float) * jobs[i].nlat);
        n += jobs[i].nlat;
    }
    const double wall = now() - t0, cpu = cpu_seconds() - c0;
    if (err) { printf("%-5s %8.2f MiB %2d threads: read error %s\n", name, block / 1048576.0, threads, strerror(err)); free(lat); return; }
    qsort(lat, n, sizeof(float), cmpf);
    const double p50 = n ? lat[n / 2] : 0, p99 = n ? lat[(int)(n * 0.99)] : 0;
    printf("%-5s %8.2f MiB %2d threads %8.1f MB/s %8.1f reads/s  p50 %7.2f ms  p99 %7.2f ms  cpu %5.1f%%  (%.2f GB in %.1f s)\n",
           name, block / 1048576.0, threads, bytes / 1e6 / wall, reads / wall, p50, p99, 100 * cpu / wall, bytes / 1e9, wall);
    printf("json {\"mode\":\"%s\",\"block\":%zu,\"threads\":%d,\"mbps\":%.1f,\"iops\":%.1f,\"p50_ms\":%.3f,\"p99_ms\":%.3f,\"cpu_pct\":%.1f,\"bytes\":%llu,\"seconds\":%.2f}\n",
           name, block, threads, bytes / 1e6 / wall, reads / wall, p50, p99, 100 * cpu / wall, (unsigned long long)bytes, wall);
    fflush(stdout);
    free(lat);
}

// The first line of a sysfs file, or "" if there is none.
static const char *line(const char *path) {
    static char buf[256];
    buf[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return buf;
    if (!fgets(buf, sizeof buf, f)) buf[0] = 0;
    fclose(f);
    buf[strcspn(buf, "\n")] = 0;
    return buf;
}

// The block device that holds `file`: its model, queue and the drivers and
// link speeds on its way to the CPU (the USB port's 480/5000/10000 Mb/s, uas
// or usb-storage).
static void describe(const char *file) {
    struct stat st;
    if (stat(file, &st)) return;
    char p[PATH_MAX], real[PATH_MAX], q[PATH_MAX + 64];
    snprintf(p, sizeof p, "/sys/dev/block/%u:%u", major(st.st_dev), minor(st.st_dev));
    if (!realpath(p, real)) { printf("drive: no sysfs entry for %s\n", p); return; }
    // A partition's parent directory is the whole disk.
    snprintf(q, sizeof q, "%s/partition", real);
    if (access(q, F_OK) == 0) { char *d = dirname(real); memmove(real, d, strlen(d) + 1); }
    char *name = strrchr(real, '/') + 1;
    printf("drive: %s", name);
    snprintf(q, sizeof q, "%s/device/vendor", real); printf("  %s", line(q));
    snprintf(q, sizeof q, "%s/device/model", real); printf(" %s", line(q));
    snprintf(q, sizeof q, "%s/size", real); printf("  %.1f GB", atof(line(q)) * 512 / 1e9);
    snprintf(q, sizeof q, "%s/queue/rotational", real); printf("  rotational %s", line(q));
    snprintf(q, sizeof q, "%s/queue/scheduler", real); printf("  scheduler %s", line(q));
    snprintf(q, sizeof q, "%s/queue/max_sectors_kb", real); printf("  max_sectors_kb %s", line(q));
    snprintf(q, sizeof q, "%s/queue/nr_requests", real); printf("  nr_requests %s\n", line(q));
    // Up the device path: every driver and link speed on the way.
    snprintf(q, sizeof q, "%s/device", real);
    char dev[PATH_MAX];
    if (realpath(q, dev)) {
        printf("  path %s\n", dev);
        while (strlen(dev) > 5) {
            char drv[PATH_MAX + 16], link[PATH_MAX];
            snprintf(drv, sizeof drv, "%s/driver", dev);
            if (realpath(drv, link)) printf("  %s: driver %s\n", strrchr(dev, '/') + 1, strrchr(link, '/') + 1);
            snprintf(drv, sizeof drv, "%s/speed", dev);
            const char *sp = line(drv);
            if (*sp) printf("  %s: link speed %s Mb/s\n", strrchr(dev, '/') + 1, sp);
            char *d = dirname(dev);
            memmove(dev, d, strlen(d) + 1);
        }
    }
    DIR *nv = opendir("/sys/class/nvme");
    int n = 0;
    if (nv) {
        for (struct dirent *e; (e = readdir(nv));) {
            if (e->d_name[0] == '.') continue;
            snprintf(q, sizeof q, "/sys/class/nvme/%s/model", e->d_name);
            printf("nvme: %s %s\n", e->d_name, line(q));
            n++;
        }
        closedir(nv);
    }
    if (!n) printf("nvme: none\n");
}

// Whether a process called ts5 is running.
static int ts5_running(void) {
    DIR *d = opendir("/proc");
    if (!d) return 0;
    int found = 0;
    for (struct dirent *e; !found && (e = readdir(d));) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char q[300];
        snprintf(q, sizeof q, "/proc/%s/comm", e->d_name);
        found = strcmp(line(q), "ts5") == 0;
    }
    closedir(d);
    return found;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s FILE...  (large files on the drive to test; read only)\n", argv[0]); return 2; }
    if (ts5_running() && !getenv("TS5_SSD_FORCE")) { fprintf(stderr, "ssd_read: a ts5 process is running; not starting (TS5_SSD_FORCE=1 to run anyway)\n"); return 3; }
    describe(argv[1]);
    for (int a = 1; a < argc && nf < MAXF; a++) {
        const int fd = open(argv[a], O_RDONLY | O_DIRECT);
        if (fd < 0) { fprintf(stderr, "%s: %s\n", argv[a], strerror(errno)); continue; }
        struct stat st;
        if (fstat(fd, &st) || st.st_size < (64 << 20)) { close(fd); continue; }  // skip small files
        fds[nf] = fd;
        sizes[nf] = (uint64_t)st.st_size;
        total_size += sizes[nf];
        nf++;
    }
    if (!nf) { fprintf(stderr, "no file of 64 MiB or more could be opened with O_DIRECT\n"); return 1; }
    const char *e = getenv("TS5_SSD_SECONDS");
    const double secs = e ? atof(e) : 4.0;
    printf("ssd_read: %d files, %.2f GB, %.1f s per test, O_DIRECT\n", nf, total_size / 1e9, secs);
    fflush(stdout);
    // Sequential first: what a model load sees.
    run("seq", 16u << 20, 1, 1, 2 * secs, 8ull << 30);
    run("seq", 16u << 20, 4, 1, 2 * secs, 8ull << 30);
    const size_t blocks[] = {128u << 10, 512u << 10, 864u << 10, 1260u << 10, 2796u << 10, 12900u << 10};
    const int threads[] = {1, 2, 4, 8, 16};
    for (size_t b = 0; b < sizeof blocks / sizeof *blocks; b++)
        for (size_t t = 0; t < sizeof threads / sizeof *threads; t++)
            run("rand", blocks[b], threads[t], 0, secs, 4ull << 30);
    return 0;
}
