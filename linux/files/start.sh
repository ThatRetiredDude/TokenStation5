#!/bin/sh
# Starts "ts5 serve" with the settings in /etc/default/tokenstation.
set -eu

models_dir=${MODELS_DIR:-/var/lib/tokenstation/models}
model=${MODEL:-}
if [ -z "$model" ]; then
    model=$(find "$models_dir" -maxdepth 1 -name '*.gguf' 2>/dev/null | LC_ALL=C sort | head -n 1)
fi
# No model yet: say so and stop cleanly (exit 0), so systemd doesn't restart
# it every few seconds. "hw model NAME.gguf" (Settings) or
# "sudo systemctl restart tokenstation" starts it once one is there.
if [ -z "$model" ]; then
    echo "tokenstation: no model yet. Put a .gguf in $models_dir (or set MODEL in /etc/default/tokenstation), then: sudo systemctl restart tokenstation" >&2
    exit 0
fi
if [ ! -f "$model" ]; then
    echo "tokenstation: MODEL=$model in /etc/default/tokenstation isn't there; the service stays off until it is (then: sudo systemctl restart tokenstation)" >&2
    exit 0
fi

ctx=${CONTEXT:-8192}
[ "$ctx" = auto ] && ctx=0

# The API keys other machines get in with ("ts5 keys", or ts5-tui's Access
# page). Whether they need one is in the file itself; without a file, and
# until "require" is turned on, everyone on the network gets in as before.
export TS5_KEYS="${KEYS:-/var/lib/tokenstation/access/keys.tsv}"

# EXTRA_ARGS is split on spaces on purpose.
# shellcheck disable=SC2086
exec /usr/bin/ts5 serve -m "$model" --host "${HOST:-0.0.0.0}" --port "${PORT:-8081}" -c "$ctx" ${EXTRA_ARGS:-}
