// The console's own time of day, for dates on the TV.
#pragma once

#include <cstdint>
#include <ctime>

namespace app {

// The fields of `utc` (seconds since 1970) in the console's time zone, the one
// its Date and Time setting shows; UTC when the console doesn't say (the log
// says so once). False for a time before 1970.
bool local_tm(int64_t utc, struct tm *out);

}  // namespace app
