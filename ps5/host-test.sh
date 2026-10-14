#!/usr/bin/env bash
# Runs the PS5 title's own main (ps5/title.cpp) on Linux, with the console's
# functions stood in for by ps5/host_shim.cpp and a folder for /data, and
# walks it through what the PC does to it over FTP: pick a model by itself,
# serve, restart, a one-off check, quit.
#
#   ps5/host-test.sh MODEL.gguf
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model=$(realpath "${1:?a small GGUF model}")
work="$root/build/ps5-host"
data="$work/data"
port=8081
rm -rf "$data"
mkdir -p "$work/shaders" "$data"

for src in "$root"/engine/shaders/*.comp; do
    stem=$(basename "$src" .comp)
    glslangValidator -V --target-env vulkan1.2 -I"$root/engine/shaders" --vn "spv_$stem" \
        -o "$work/shaders/$stem.spv.h" "$src" > /dev/null
done
c++ -std=c++17 -O2 -Wall -pthread -DTS5_PS5=1 -DTS5_VULKAN=1 "-DTS5_DATA=\"$data\"" \
    -I "$root/engine/src" -I "$work/shaders" \
    "$root"/engine/src/{gguf,quants,model,cpu_backend,vk_backend,tokenizer,engine,server,cli}.cpp \
    "$root/ps5/title.cpp" "$root/ps5/host_shim.cpp" -ldl -o "$work/eboot.bin"

title_pid=
cleanup() { [ -n "$title_pid" ] && kill "$title_pid" 2> /dev/null || true; }
trap cleanup EXIT
fail() {
    echo "host-test: FAIL: $*" >&2
    for f in "$data"/ts5/ts5.prev.log "$data"/ts5/ts5.log "$data"/notifications.txt; do
        [ -f "$f" ] && { echo "--- $f" >&2; cat "$f" >&2; }
    done
    exit 1
}
launch() { (cd "$work" && exec ./eboot.bin) & title_pid=$!; }
wait_health() {
    for _ in $(seq 1 100); do
        curl -sf "http://127.0.0.1:$port/health" > /dev/null && return 0
        kill -0 "$title_pid" 2> /dev/null || fail "the title ended while starting"
        sleep 0.2
    done
    fail "no /health after 20 s"
}
wait_down() {
    for _ in $(seq 1 100); do
        curl -sf "http://127.0.0.1:$port/health" > /dev/null || return 0
        sleep 0.1
    done
    fail "the server did not go down"
}
notified() { grep -qF "$1" "$data/notifications.txt" || fail "no notification: $1"; }

echo "== no model: a notification, then the title leaves"
launch
wait "$title_pid" || fail "exit status $?"
title_pid=
notified "no model"

echo "== a model in /data/ts5/models: serves it with no args file"
cp "$model" "$data/ts5/models/"
launch
wait_health
notified "$(basename "$model") ready at http://"
reply=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}')
grep -q '"completion_tokens":4' <<< "$reply" || fail "chat completion: $reply"

echo "== restart: the title starts again, its log becomes ts5.prev.log"
: > "$data/ts5/restart"
wait_down
wait_health
grep -q "TokenStation5" "$data/ts5/ts5.prev.log" || fail "no previous log"

echo "== a one-off check, then back to serving"
echo "check -m $data/ts5/models/$(basename "$model") -n 6" > "$data/ts5/ts5.once"
: > "$data/ts5/restart"
wait_down
wait_health
[ ! -e "$data/ts5/ts5.once" ] || fail "ts5.once was not consumed"
grep -q "^ts5 check: .* PASS$" "$data/ts5/ts5.prev.log" || fail "check did not pass"
notified "GPU check passed"

echo "== the args file: another port"
echo "serve -m \"$data/ts5/models/$(basename "$model")\" --port 18081" > "$data/ts5/ts5.args"
: > "$data/ts5/restart"
wait_down
port=18081
wait_health

echo "== quit: the title leaves"
: > "$data/ts5/quit"
wait "$title_pid" || fail "exit status $?"
title_pid=
notified "closing"
echo "host-test: PASS"
