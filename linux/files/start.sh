#!/bin/sh
# Starts "ts5 serve" with the settings in /etc/default/tokenstation.
set -eu

models_dir=${MODELS_DIR:-/var/lib/tokenstation/models}
model=${MODEL:-}
if [ -z "$model" ]; then
    model=$(find "$models_dir" -maxdepth 1 -name '*.gguf' 2>/dev/null | LC_ALL=C sort | head -n 1)
fi
if [ -z "$model" ] || [ ! -f "$model" ]; then
    echo "tokenstation: no model. Put a .gguf in $models_dir or set MODEL in /etc/default/tokenstation." >&2
    exit 1
fi

ctx=${CONTEXT:-8192}
[ "$ctx" = auto ] && ctx=0

# EXTRA_ARGS is split on spaces on purpose.
# shellcheck disable=SC2086
exec /usr/bin/ts5 serve -m "$model" --host "${HOST:-0.0.0.0}" --port "${PORT:-8081}" -c "$ctx" ${EXTRA_ARGS:-}
