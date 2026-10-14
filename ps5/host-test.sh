#!/usr/bin/env bash
# Runs the PS5 title's own main (ps5/title.cpp) on Linux, with the console's
# functions stood in for by ps5/host_shim.cpp and folders for its own folder
# (/app0) and /data, and walks it through what the controller and the PC do
# to it: serve a model from the dashboard, a GPU check and a speed test,
# screenshots, restart, a one-off check, another port, quit and close.
# The screenshots stay in build/ps5-host/shots.
#
#   ps5/host-test.sh MODEL.gguf
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model=$(realpath "${1:?a small GGUF model}")
work="$root/build/ps5-host"
data="$work/data"
home="$work/app0"
port=8081
rm -rf "$data" "$home" "$work/shots"
mkdir -p "$work/shaders" "$data" "$home" "$work/shots"

for src in "$root"/engine/shaders/*.comp; do
    stem=$(basename "$src" .comp)
    glslangValidator -V --target-env vulkan1.2 -I"$root/engine/shaders" --vn "spv_$stem" \
        -o "$work/shaders/$stem.spv.h" "$src" > /dev/null
done
c++ -std=c++17 -O2 -Wall -pthread -DTS5_PS5=1 -DTS5_VULKAN=1 "-DTS5_DATA=\"$data\"" "-DTS5_HOME=\"$home\"" \
    -I "$root/engine/src" -I "$root/ps5" -I "$root/ps5/ui" -I "$work/shaders" \
    "$root"/engine/src/{gguf,quants,model,cpu_backend,vk_context,vk_backend,tokenizer,engine,server,checks}.cpp \
    "$root"/ps5/{title,library,runner,audio_probe,host_shim}.cpp "$root"/ps5/ui/{canvas,display,pad,dashboard}.cpp \
    -x assembler-with-cpp -Wa,-I"$root/ps5/ui" "$root/ps5/ui/font_data.S" -x none -ldl -o "$work/eboot.bin"

title_pid=
cleanup() { [ -n "$title_pid" ] && kill "$title_pid" 2> /dev/null || true; }
trap cleanup EXIT
fail() {
    echo "host-test: FAIL: $*" >&2
    for f in "$home"/ts5.prev.log "$home"/ts5.log "$data"/notifications.txt; do
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

# Waits for a line matching a regex in the title's log.
wait_log() {
    for _ in $(seq 1 300); do
        grep -qE "$1" "$home/ts5.log" 2> /dev/null && return 0
        kill -0 "$title_pid" 2> /dev/null || fail "the title ended while waiting for: $1"
        sleep 0.1
    done
    fail "no '$1' in the log after 30 s"
}
health_says() { curl -s "http://127.0.0.1:$port/health" | grep -qF "$1"; }
# The next frame, saved as shots/NAME.bmp.
shot() {
    rm -f "$home/screen.bmp"
    : > "$home/screenshot"
    for _ in $(seq 1 50); do
        [ -e "$home/screenshot" ] || [ ! -s "$home/screen.bmp" ] || break
        sleep 0.1
    done
    [ -s "$home/screen.bmp" ] || fail "no screenshot"
    [ "$(stat -c %s "$home/screen.bmp")" = $((54 + 1920 * 1080 * 4)) ] || fail "the screenshot is not 1920x1080"
    cp "$home/screen.bmp" "$work/shots/$1.bmp"
}
press() { echo "$*" > "$home/press"; sleep 0.5; }

name=$(basename "$model")

echo "== no model: the dashboard, a notification, and the API says so"
launch
wait_log "no model"
for _ in $(seq 1 50); do health_says "no model" && break; sleep 0.1; done
health_says "no model" || fail "/health: $(curl -s "http://127.0.0.1:$port/health")"
notified "no model"
shot empty

echo "== a model arrives over FTP: the dashboard lists it, cross serves it"
# Whole, as an FTP upload that finished: a file still arriving can't run yet.
cp "$model" "$home/$name.part"
mv "$home/$name.part" "$home/models/$name"
wait_log "^ts5-title: models: $name"
press cross
wait_health
notified "$name ready at http://"
grep -q "serve -m \"$home/models/$name\"" "$home/ts5.args" || fail "serving did not write ts5.args"
reply=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}')
grep -q '"completion_tokens":4' <<< "$reply" || fail "chat completion: $reply"
shot serving

echo "== square: the GPU check on the serving model, then serving again"
press square
wait_log "^ts5 check: .* PASS$"
notified "GPU check passed"
grep -q "^check	$name	.*	PASS	" "$home/results.tsv" || fail "no check in results.tsv"
wait_health

echo "== ts5.cmd: a speed test from the PC"
echo "bench -m $name --pp 32 --tg 8" > "$home/ts5.cmd"
wait_log "^\| $name \| vk:"
grep -q "^bench	$name	" "$home/results.tsv" || fail "no speed test in results.tsv"
wait_health
shot results

echo "== a second check: GPU only, against the CPU's answers saved by the first"
[ -s "$home/models/$name.ts5ref" ] || fail "the first check saved no answers"
echo "check -m $name" > "$home/ts5.cmd"
wait_log "^ts5 check: .*, cpu answers saved earlier: PASS$"
wait_health

echo "== --env: settings for one job (here the GPU's time per kernel)"
echo "bench -m $name --pp 32 --tg 8 --env TS5_PROFILE=1" > "$home/ts5.cmd"
wait_log "^ts5-title: for this job: TS5_PROFILE=1$"
wait_log "^generation \(8 tokens\), GPU time"
wait_health

echo "== audiotest: no audio on the PC, so the test says so and the title carries on"
: > "$home/audiotest"
wait_log "^ts5-title: audio test: done$"
grep -q "audio test: load /host/common/lib/libSceAudioOut.sprx 0x80020002" "$home/ts5.log" || fail "the audio test logged no module load"
notified "the controller's mic didn't open"
wait_health

echo "== circle: stop serving"
press circle
for _ in $(seq 1 50); do health_says "no model" && break; sleep 0.1; done
health_says "no model" || fail "still serving after circle"

echo "== restart: serves what ts5.args says, its log becomes ts5.prev.log"
: > "$home/restart"
wait_down
wait_health
grep -q "TokenStation5" "$home/ts5.prev.log" || fail "no previous log"

echo "== ts5.once: a one-off check at the next start, then the args file's model"
echo "== ts5.env: engine settings (here every matmul tiled, even one token's)"
echo "check -m $name -n 6" > "$home/ts5.once"
echo "TS5_TILED_MIN=1 TS5_FAST_MATVEC=0" > "$home/ts5.env"
: > "$home/restart"
wait_down
wait_log "^ts5-title: TS5_TILED_MIN=1$"
wait_log "^ts5 check: .*, 6 tokens: .* PASS$"
wait_health
[ ! -e "$home/ts5.once" ] || fail "ts5.once was not consumed"
rm "$home/ts5.env"

echo "== the args file: another port"
echo "serve -m \"$home/models/$name\" --port 18081" > "$home/ts5.args"
: > "$home/restart"
wait_down
port=18081
wait_health

echo "== quit: the title leaves"
: > "$home/quit"
wait "$title_pid" || fail "exit status $?"
title_pid=
notified "closing"

echo "== OPTIONS, then cross: the title closes from the dashboard"
rm -f "$data/notifications.txt"
launch
wait_health
press options
shot close
press cross
wait "$title_pid" || fail "exit status $?"
title_pid=
notified "closing"
echo "host-test: PASS"
