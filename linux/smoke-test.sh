#!/usr/bin/env bash
# Runs the Linux build of TokenStation5 against a scratch data folder and
# checks what a user does on the TV: the dashboard comes up, a model copied
# into models/ is listed and served with cross, the API answers, the chat
# page replies, a screenshot is 1920x1080, a restart comes back, and quit
# closes it. Needs no screen or sound card: SDL's dummy drivers stand in
# (TS5_SMOKE_VIDEO=x11 under xvfb-run to see the real window path), and any
# Vulkan driver does, lavapipe included.
#
#   linux/smoke-test.sh MODEL.gguf [build/linux/tokenstation]
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model=$(realpath "${1:?a small GGUF model}")
bin=$(realpath "${2:-$root/build/linux/tokenstation}")
data=$(mktemp -d)
port=8081
export SDL_VIDEODRIVER=${TS5_SMOKE_VIDEO:-dummy} SDL_AUDIODRIVER=dummy TS5_DESKTOP_NOTIFY=0
export TS5_CHAT_MAX_TOKENS=8

pid=
cleanup() {
    [ -n "$pid" ] && kill "$pid" 2> /dev/null || true
    rm -rf "$data"
}
trap cleanup EXIT
fail() {
    echo "smoke-test: FAIL: $*" >&2
    [ -f "$data/ts5.log" ] && tail -40 "$data/ts5.log" >&2
    exit 1
}
launch() { "$bin" --data "$data" --windowed 2> "$data/stderr.txt" & pid=$!; }
wait_log() {
    for _ in $(seq 1 300); do
        grep -qE "$1" "$data/ts5.log" 2> /dev/null && return 0
        kill -0 "$pid" 2> /dev/null || fail "it ended while waiting for: $1"
        sleep 0.1
    done
    fail "no '$1' in the log after 30 s"
}
health() { curl -sf "http://127.0.0.1:$port/health" 2> /dev/null; }
press() { echo "$*" > "$data/press"; sleep 0.5; }

echo "== start: the dashboard, the screen, no model yet"
launch
wait_log "ts5-title: TokenStation5 "
wait_log "screen: "
wait_log "no model"
grep -q "data in $data" "$data/stderr.txt" || fail "the data folder wasn't $data"

echo "== a model copied into models/ is listed; cross serves it"
name=$(basename "$model")
cp "$model" "$data/models/$name.part"
mv "$data/models/$name.part" "$data/models/$name"
wait_log "^ts5-title: models: $name"
press cross
for _ in $(seq 1 200); do health | grep -q '"ok"' && break; sleep 0.2; done
health | grep -q '"ok"' || fail "/health: $(health)"
grep -q "serve -m \"./models/$name\"" "$data/ts5.args" || fail "ts5.args: $(cat "$data/ts5.args")"

echo "== the API answers"
reply=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}') || fail "no chat completion"
echo "$reply" | grep -q '"choices"' || fail "chat completion: $reply"

echo "== the chat page replies"
echo "Hello from the smoke test" > "$data/chat"
wait_log "ts5-title: chat: message"
for _ in $(seq 1 100); do grep -q "Hello from the smoke test" "$data"/chats/* 2> /dev/null && break; sleep 0.2; done
grep -q "Hello from the smoke test" "$data"/chats/* 2> /dev/null || fail "the chat wasn't saved"

echo "== a screenshot is the whole 1920x1080 frame"
: > "$data/screenshot"
for _ in $(seq 1 50); do [ -e "$data/screenshot" ] || break; sleep 0.1; done
[ "$(stat -c %s "$data/screen.bmp")" = $((54 + 1920 * 1080 * 4)) ] || fail "screen.bmp isn't 1920x1080"
grep -q "TokenStation5" "$data/screen.txt" || fail "screen.txt: $(head -5 "$data/screen.txt")"

echo "== restart comes back serving the same model"
: > "$data/restart"
sleep 1
for _ in $(seq 1 200); do health | grep -q '"ok"' && break; sleep 0.2; done
health | grep -q '"ok"' || fail "no /health after the restart"

echo "== SIGTERM leaves cleanly, as systemd stops it"
kill -TERM "$pid"
for _ in $(seq 1 50); do kill -0 "$pid" 2> /dev/null || break; sleep 0.1; done
kill -0 "$pid" 2> /dev/null && fail "still running 5 s after SIGTERM"
grep -q "leaving (exit)" "$data/ts5.log" || fail "it didn't leave the title's way"
pid=
echo "smoke-test: OK"
