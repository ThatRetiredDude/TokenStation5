#!/usr/bin/env python3
# Writes a /proc/meminfo stand-in (TS5_MEMINFO) whose MemAvailable starts at
# this machine's own, climbs RISE MiB every 250 ms for SECONDS, then stays:
# memory a process that just ended is still giving back. Runs for TOTAL s.
# With PERIOD, the rise comes in steps every PERIOD s instead (as the kernel
# shows it, folding each CPU's count in about once a second), the first one
# at PERIOD / 2. With SECONDS "ck", it replays the PS5's own curve right
# after a full build (job CK, 16:11:22 on 2026-11-08): +572 MiB in 22 s;
# with "serve", its curve after a server of Qwen3 14B stopped (job CM part
# 5, 20:18:59): +98 MiB in 13 s, with up to 2.5 s of nothing between steps.
#   meminfo_curve.py FILE RISE SECONDS TOTAL [PERIOD]
import os, sys, time
path, total = sys.argv[1], float(sys.argv[4])
ck = sys.argv[3] in ('ck', 'serve')
rise, secs = float(sys.argv[2]), 0.0 if ck else float(sys.argv[3])
period = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
# (s after the build, MiB above the reading at its exit)
CK = [(0.50, 49), (1.26, 74), (1.51, 114), (2.26, 171), (2.51, 176), (3.51, 231), (4.27, 256),
      (4.52, 283), (5.02, 286), (5.27, 320), (5.52, 322), (6.28, 354), (6.53, 357), (7.28, 376),
      (7.53, 389), (8.29, 416), (9.29, 440), (10.04, 448), (12.05, 485), (14.06, 513),
      (16.07, 545), (18.08, 556), (20.08, 569), (22.09, 572)]
# Past 10 s the trace has a reading every 2 s; the steps still came about
# once a second, so each 2 s splits into two.
CK += [((a + c) / 2, (b + d) / 2) for (a, b), (c, d) in zip(CK[16:], CK[17:])]
SERVE = [(0.50, 1), (1.51, 5), (2.26, 10), (2.51, 32), (3.26, 39), (3.51, 41), (4.27, 43), (4.52, 46),
         (5.27, 48), (5.52, 49), (6.28, 67), (7.28, 69), (7.53, 72), (8.28, 80), (8.79, 81), (10.54, 82),
         (11.30, 96), (13.30, 98)]
if sys.argv[3] == 'serve': CK = SERVE
base = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))
t0 = time.time()
while True:
    t = time.time() - t0
    if ck:
        mib = max([m for s, m in CK if s <= t] or [0])
    elif period:
        steps = int((t + period / 2) / period) if t >= period / 2 else 0
        mib = min(steps * period, secs) / 0.25 * rise
    else:
        mib = min(t, secs) / 0.25 * rise
    kb = base + int(mib * 1024)
    with open(path + '.tmp', 'w') as f:
        f.write(f"MemTotal: {base * 2} kB\nMemAvailable: {kb} kB\n")
    os.replace(path + '.tmp', path)
    if t > total: break
    time.sleep(0.05)
