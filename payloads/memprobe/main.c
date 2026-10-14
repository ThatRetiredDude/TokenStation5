/* ts5-memprobe: a read-only snapshot of where the PS5's memory is.
 *
 * Sent to the jailbreak's ELF loader (port 9021). It changes nothing: it
 * reads the process table, the VM counters, the direct and flexible memory
 * calls as a payload sees them, and every sysctl whose name mentions memory
 * or budgets, then writes the lot to stdout and to /data/ts5-memprobe.txt
 * (FTP: scripts/ts5-ps5.sh memprobe fetches it).
 *
 * Copyright (C) 2026 ThatRetiredDude. GPL-3.0-or-later (links the PS5
 * payload SDK). */

#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/user.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <ps5/kernel.h>

typedef struct app_info {
    uint32_t app_id;
    uint64_t unknown1;
    uint32_t app_type;
    char title_id[10];
    char unknown2[0x3c];
} app_info_t;

int sceKernelGetAppInfo(pid_t pid, app_info_t *info);
size_t sceKernelGetDirectMemorySize(void);
int sceKernelAvailableDirectMemorySize(off_t start, off_t end, size_t align, off_t *found, size_t *size);
int sceKernelAvailableFlexibleMemorySize(size_t *size);

static FILE *out;

static void say(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (out) {
        va_start(ap, fmt);
        vfprintf(out, fmt, ap);
        va_end(ap);
    }
}

#define MiB(x) ((double)(x) / (1024.0 * 1024.0))

static void processes(void) {
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PROC, 0};
    size_t len = 0;
    if (sysctl(mib, 4, NULL, &len, NULL, 0)) {
        say("processes: sysctl failed\n");
        return;
    }
    len += len / 4;
    char *buf = malloc(len);
    if (!buf || sysctl(mib, 4, buf, &len, NULL, 0)) {
        say("processes: sysctl failed\n");
        free(buf);
        return;
    }
    say("\n## processes (resident / virtual MiB)\n");
    say("%6s %6s %10s %9s %10s  %s\n", "pid", "ppid", "title", "rss", "vsz", "command");
    double total = 0;
    for (char *p = buf; p < buf + len;) {
        struct kinfo_proc *ki = (struct kinfo_proc *)p;
        if (ki->ki_structsize <= 0) break;
        p += ki->ki_structsize;
        app_info_t app;
        if (sceKernelGetAppInfo(ki->ki_pid, &app)) memset(&app, 0, sizeof(app));
        app.title_id[9] = 0;
        double rss = MiB((uint64_t)ki->ki_rssize * PAGE_SIZE);
        total += rss;
        say("%6d %6d %10s %9.1f %10.1f  %s\n", ki->ki_pid, ki->ki_ppid, app.title_id, rss, MiB(ki->ki_size),
            ki->ki_comm);
    }
    say("resident total: %.1f MiB (shared pages counted in every process)\n", total);
    free(buf);
}

static void counter(const char *name) {
    uint64_t v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, NULL, 0)) {
        say("%-32s n/a\n", name);
        return;
    }
    if (len == 4) v &= 0xffffffffu;
    say("%-32s %llu\n", name, (unsigned long long)v);
}

static void vm(void) {
    say("\n## physical memory (pages of %d bytes unless noted)\n", PAGE_SIZE);
    const char *names[] = {"hw.physmem", "hw.realmem", "hw.usermem", "hw.pagesize",
                           "vm.stats.vm.v_page_count", "vm.stats.vm.v_free_count", "vm.stats.vm.v_wire_count",
                           "vm.stats.vm.v_active_count", "vm.stats.vm.v_inactive_count",
                           "vm.stats.vm.v_laundry_count", "vm.stats.vm.v_cache_count", "vm.kmem_size",
                           "vm.kmem_size_max"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) counter(names[i]);
}

static void pools(void) {
    say("\n## direct and flexible memory, as this payload sees them\n");
    size_t pool = sceKernelGetDirectMemorySize();
    say("direct pool:          %zu (%.1f MiB)\n", pool, MiB(pool));
    off_t at = 0;
    size_t avail = 0;
    int r = sceKernelAvailableDirectMemorySize(0, (off_t)pool, 0x4000, &at, &avail);
    say("direct available:     %zu (%.1f MiB, largest run, rc 0x%x)\n", avail, MiB(avail), r);
    size_t flex = 0;
    r = sceKernelAvailableFlexibleMemorySize(&flex);
    say("flexible available:   %zu (%.1f MiB, rc 0x%x)\n", flex, MiB(flex), r);
}

/* Every sysctl, walked with the {0, 2} "next" query; prints the ones whose
 * name mentions memory or budgets, with their value when it is a number or
 * a string. */
static void sysctls(void) {
    say("\n## sysctls mentioning mem, dmem, budget or swap\n");
    int oid[CTL_MAXNAME + 2] = {0, 2, 1};
    size_t oidlen = 1;
    int count = 0;
    for (;;) {
        int next[CTL_MAXNAME];
        size_t nextlen = sizeof(next);
        oid[0] = 0;
        oid[1] = 2;
        if (sysctl(oid, (u_int)(oidlen + 2), next, &nextlen, NULL, 0)) break;
        oidlen = nextlen / sizeof(int);
        memcpy(oid + 2, next, nextlen);

        int q[CTL_MAXNAME + 2] = {0, 1};
        memcpy(q + 2, next, nextlen);
        char name[256];
        size_t namelen = sizeof(name);
        if (sysctl(q, (u_int)(oidlen + 2), name, &namelen, NULL, 0)) continue;
        if (!strstr(name, "mem") && !strstr(name, "budget") && !strstr(name, "swap")) continue;

        q[1] = 4;
        unsigned char fmt[64];
        size_t fmtlen = sizeof(fmt);
        if (sysctl(q, (u_int)(oidlen + 2), fmt, &fmtlen, NULL, 0)) continue;
        unsigned kind = *(unsigned *)fmt & CTLTYPE;
        if (kind == CTLTYPE_NODE) continue;

        unsigned char val[512];
        size_t vallen = sizeof(val);
        if (sysctl(next, (u_int)oidlen, val, &vallen, NULL, 0)) {
            say("%-48s (unreadable)\n", name);
        } else if (kind == CTLTYPE_STRING) {
            val[vallen < sizeof(val) ? vallen : sizeof(val) - 1] = 0;
            say("%-48s \"%s\"\n", name, (char *)val);
        } else if (vallen == 4) {
            say("%-48s %d\n", name, *(int *)val);
        } else if (vallen == 8) {
            say("%-48s %lld\n", name, *(long long *)val);
        } else {
            say("%-48s (%zu bytes)\n", name, vallen);
        }
        if (++count > 400) break;
    }
}

int main(void) {
    out = fopen("/data/ts5-memprobe.txt", "w");
    time_t now = time(NULL);
    say("# ts5-memprobe, firmware 0x%08x, %s", kernel_get_fw_version(), ctime(&now));
    pools();
    vm();
    processes();
    sysctls();
    say("\n# done\n");
    if (out) fclose(out);
    return 0;
}
