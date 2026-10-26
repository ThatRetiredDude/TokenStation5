#include "local_time.h"

#include <ctime>

#include "log.h"

extern "C" int sceKernelDlsym(int module, const char *name, void **addr);

namespace app {

namespace {

// libkernel's sceKernelConvertUtcToLocaltime: a UTC time moved into the
// console's time zone, daylight saving included. The title looks it up by name
// when a date is first shown, so it builds whether or not the SDK lists the
// call, and a console without it shows UTC.
struct Timezone {
    int minutes_west, dst;
};
using Convert = int (*)(const struct timespec *utc, struct timespec *local, Timezone *zone, int *dst);
constexpr int kLibKernel = 0x2001;

Convert converter() {
    static const Convert fn = [] {
        void *p = nullptr;
        if (sceKernelDlsym(kLibKernel, "sceKernelConvertUtcToLocaltime", &p) != 0 || !p) {
            ts5::log_line("ts5-title: the console didn't say its time zone; dates are in UTC");
            return (Convert) nullptr;
        }
        return (Convert)p;
    }();
    return fn;
}

}  // namespace

bool local_tm(int64_t utc, struct tm *out) {
    if (utc < 0 || !out) return false;
    time_t t = (time_t)utc;
    if (Convert fn = converter()) {
        struct timespec in{}, local{};
        in.tv_sec = t;
        Timezone zone{};
        int dst = 0;
        if (fn(&in, &local, &zone, &dst) == 0) t = local.tv_sec;
    }
    return gmtime_r(&t, out) != nullptr;
}

}  // namespace app
