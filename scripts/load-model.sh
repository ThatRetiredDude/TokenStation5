#!/usr/bin/env bash
# Asks the running PS5LM Plus app to switch model, the way its library does:
# it polls /data/PS5LM/load once a second, restarts itself and loads the file
# (about 90 s for a model to reach the GPU).
#
#   scripts/load-model.sh Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
#   scripts/load-model.sh /data/PS5LM/models/some.gguf
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
. "$root/scripts/env.sh"
model=${1:?model file name or path on the console}
case $model in /*) ;; *) model="$PS5_MODEL_DIR/$model" ;; esac
printf '%s\n' "$model" | curl -sS -T - "ftp://$PS5_HOST:$PS5_FTP_PORT/data/PS5LM/load"
echo "load-model: asked for $model; waiting for the API"
for _ in $(seq 1 90); do
    sleep 5
    if curl -sf "http://$PS5_HOST:$PS5_API_PORT/health" > /dev/null; then
        echo "load-model: ready at http://$PS5_HOST:$PS5_API_PORT/v1"
        exit 0
    fi
done
echo "load-model: API not up after 7.5 minutes; check the TV" >&2
exit 1
