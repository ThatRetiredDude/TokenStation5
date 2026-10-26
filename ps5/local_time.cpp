#include "local_time.h"

#include <ctime>

#include "log.h"

extern "C" int sceSystemServiceParamGetInt(int id, int *value);

namespace app {

namespace {

// The console's Date and Time settings, as the system service gives them to
// a title: its time zone in minutes from UTC, and whether summer time is on.
// libkernel's sceKernelConvertUtcToLocaltime would do this too, but
// sceKernelDlsym doesn't find it from a title (console run 2026-10-26).
constexpr int kParamTimeZone = 4, kParamSummerTime = 5;

// Seconds to add to UTC for the console's time, read when a date is first
// shown: a time zone changed while the title runs shows from its next start.
int64_t offset() {
    static const int64_t seconds = [] {
        int minutes = 0, summer = 0;
        if (sceSystemServiceParamGetInt(kParamTimeZone, &minutes) != 0) {
            ts5::log_line("ts5-title: the console didn't say its time zone; dates are in UTC");
            return (int64_t)0;
        }
        if (sceSystemServiceParamGetInt(kParamSummerTime, &summer) != 0) summer = 0;
        ts5::log_line("ts5-title: the console's time zone is %d minutes from UTC, summer time %s", minutes,
                      summer ? "on" : "off");
        return (int64_t)(minutes + (summer ? 60 : 0)) * 60;
    }();
    return seconds;
}

}  // namespace

bool local_tm(int64_t utc, struct tm *out) {
    if (utc < 0 || !out) return false;
    const time_t t = (time_t)(utc + offset());
    return gmtime_r(&t, out) != nullptr;
}

}  // namespace app
