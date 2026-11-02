#!/bin/sh
# PS5 only: turns on the fan, then the clock boost, with ps5_control from
# ps5-linux-tools. The fan goes first, as the PS5's own system does.
# Does nothing (and never fails the service) when PS5_BOOST=0 or the tool
# is missing, so the package works on any Linux machine.
[ "${PS5_BOOST:-1}" = 1 ] || exit 0
tool=${PS5_CONTROL:-$(command -v ps5_control 2>/dev/null || true)}
if [ -z "$tool" ] || [ ! -x "$tool" ]; then
    echo "tokenstation: ps5_control not found, clocks left alone" >&2
    exit 0
fi
"$tool" --fan on && "$tool" --boost on || echo "tokenstation: ps5_control failed, clocks left alone" >&2
exit 0
