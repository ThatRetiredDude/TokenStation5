#!/usr/bin/env python3
# Writes a /proc/meminfo stand-in (TS5_MEMINFO) whose MemAvailable starts at
# this machine's own, climbs RISE MiB every 250 ms for SECONDS, then stays:
# memory a process that just ended is still giving back. Runs for TOTAL s.
#   meminfo_curve.py FILE RISE SECONDS TOTAL
import os, sys, time
path, rise, secs, total = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
base = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))
t0 = time.time()
while True:
    t = time.time() - t0
    kb = base + int(min(t, secs) / 0.25 * rise * 1024)
    with open(path + '.tmp', 'w') as f:
        f.write(f"MemTotal: {base * 2} kB\nMemAvailable: {kb} kB\n")
    os.replace(path + '.tmp', path)
    if t > total: break
    time.sleep(0.05)
