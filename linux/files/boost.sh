#!/bin/sh
# PS5 only: sets the fan curve (PS5_FAN) first, then the clock boost
# (PS5_BOOST), the way the console's own system brings the fan up before the
# clocks. Both are reapplied on every boot, because the ps5-gpufreq and
# ps5_fan drivers come up at their stock settings each time.
#
# The fan curve goes through the hw helper beside this script, which drives
# the kernel's ps5_fan hwmon over sysfs (no ps5_control needed) and falls
# back to ps5_control on an older kernel. The boost goes through ps5_control
# from ps5-linux-tools.
#
# PS5_FAN: auto (the stock curve) | cool (steeper, full speed from a 65 C
#   board reading) | early (aggressive: ramps from a 40 C board reading, full
#   by 55 C, for a box that runs long prompts back to back) | max (full
#   speed). Default auto, so the package leaves the stock curve in place
#   unless a machine opts in.
# Boost OFF is the default for models: with it on, the PS5's power chip picks
# the GPU clock and holds it under 2.23 GHz; off, the kernel's own
# ps5-gpufreq driver runs the GPU at the full 2.23 GHz, about 8% faster on
# prompts for every model (measured on Debian 13). PS5_BOOST=1 is the
# console's game clocks (CPU 3.5 GHz), cooler and a little slower.
# Does nothing (and never fails the service) when a tool is missing, so the
# package works on any Linux machine.
[ -r /etc/default/tokenstation ] && . /etc/default/tokenstation
here=$(dirname "$(readlink -f "$0")")

# Fan first. The hw helper sets the kernel's ps5_fan curve over sysfs and
# turns the console's fan servo on; it needs no ps5_control for the curve.
fan=${PS5_FAN:-auto}
if [ -x "$here/hw" ]; then
    "$here/hw" fan "$fan" >/dev/null 2>&1 || echo "tokenstation: fan '$fan' not applied" >&2
fi

# Boost needs ps5_control.
tool=${PS5_CONTROL:-$(command -v ps5_control 2>/dev/null || true)}
if [ -z "$tool" ] || [ ! -x "$tool" ]; then
    echo "tokenstation: ps5_control not found, clocks left alone" >&2
    exit 0
fi
case "${PS5_BOOST:-0}" in
1) boost=on ;;
*) boost=off ;;
esac
# Make sure the fan servo is on even when the hw helper wasn't beside us.
"$tool" --fan on >/dev/null 2>&1 || true
"$tool" --boost "$boost" || echo "tokenstation: ps5_control failed, clocks left alone" >&2
exit 0
