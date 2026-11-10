#!/bin/sh
# PS5 only: turns on the fan, then sets the clock boost as PS5_BOOST says,
# with ps5_control from ps5-linux-tools. The fan goes first, as the PS5's own
# system does.
# Boost OFF is the default for models: with it on, the PS5's power chip
# picks the GPU clock and holds it under 2.23 GHz; off, the kernel's own
# ps5-gpufreq driver runs the GPU at the full 2.23 GHz, about 8% faster on
# prompts for every model (measured on Debian 13). PS5_BOOST=1 is the
# console's game clocks (CPU 3.5 GHz), cooler and a little slower.
# Does nothing (and never fails the service) when the tool is missing, so the
# package works on any Linux machine.
tool=${PS5_CONTROL:-$(command -v ps5_control 2>/dev/null || true)}
if [ -z "$tool" ] || [ ! -x "$tool" ]; then
    echo "tokenstation: ps5_control not found, clocks left alone" >&2
    exit 0
fi
case "${PS5_BOOST:-0}" in
1) boost=on ;;
*) boost=off ;;
esac
"$tool" --fan on && "$tool" --boost "$boost" || echo "tokenstation: ps5_control failed, clocks left alone" >&2
exit 0
