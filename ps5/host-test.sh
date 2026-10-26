#!/usr/bin/env bash
# Runs the PS5 title's own main (ps5/title.cpp) on Linux, with the console's
# functions stood in for by ps5/host_shim.cpp and folders for its own folder
# (/app0) and /data, and walks it through what the controller and the PC do
# to it: serve a model from the dashboard, a GPU check and a speed test,
# screenshots, the audio, whisper and speak tests, the chat page, typing on
# the console's keyboard, the Settings page, restart, a one-off check,
# another port, quit and close. The screenshots stay in build/ps5-host/shots.
#
#   ps5/host-test.sh MODEL.gguf
#   TS5_TEST_WHISPER_MODEL=ggml-tiny.en.bin \
#   TS5_TEST_KOKORO_MODEL=kokoro-82m-f16.gguf TS5_TEST_KOKORO_VOICE=kokoro-voice-af_heart.gguf \
#       ps5/host-test.sh MODEL.gguf
#
# With a whisper model, the whisper test also transcribes whisper.cpp's
# jfk.wav with it and checks the words; with Kokoro's model and a voice, the
# speak test speaks a sentence, and whisper (when there's a model) must hear
# its words. Without them each test checks only that it says it has no model.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model=$(realpath "${1:?a small GGUF model}")
work="$root/build/ps5-host"
data="$work/data"
home="$work/app0"
port=8081
rm -rf "$data" "$home" "$work/download0" "$work/shots"
mkdir -p "$work/shaders" "$data" "$home" "$work/shots"

for src in "$root"/engine/shaders/*.comp; do
    stem=$(basename "$src" .comp)
    glslangValidator -V --target-env vulkan1.2 -I"$root/engine/shaders" --vn "spv_$stem" \
        -o "$work/shaders/$stem.spv.h" "$src" > /dev/null
done

# whisper.cpp and kokoro.cpp (third_party/, CPU only, on whisper's ggml) as
# libts5speech.a, as ps5/build.sh builds it for the console. Its objects stay
# in build/ps5-host/speech and are compiled again only when a file there or
# these flags change.
w="$root/third_party/whisper.cpp"
k="$root/third_party/kokoro.cpp"
wobj="$work/speech"
wflags=(-O2 -march=x86-64-v3 -pthread -D_GNU_SOURCE -DNDEBUG -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE
    -DGGML_SCHED_MAX_COPIES=4 "-DWHISPER_VERSION=\"$(sed -n 's/^version=//p' "$root/scripts/vendor-whisper.sh")\""
    -DKOKORO_NO_MMAP -I "$w/include" -I "$w/ggml/include" -I "$w/ggml/src" -I "$w/ggml/src/ggml-cpu"
    -I "$k/include" -I "$k/src")
mkdir -p "$wobj"
if [ "$(cat "$wobj/flags" 2> /dev/null)" != "${wflags[*]}" ]; then
    rm -f "$wobj"/*.o "$wobj/libts5speech.a"
    printf '%s' "${wflags[*]}" > "$wobj/flags"
fi
wobjects=()
pids=()
t0=$(date +%s)
for src in "$w"/src/whisper.cpp "$w"/src/aneforge/whisper-aneforge.cpp \
        "$w"/ggml/src/{ggml.c,ggml.cpp,ggml-alloc.c,ggml-quants.c,ggml-threading.cpp,gguf.cpp} \
        "$w"/ggml/src/{ggml-backend,ggml-backend-dl,ggml-backend-meta,ggml-backend-reg}.cpp \
        "$w"/ggml/src/ggml-cpu/{ggml-cpu.c,ggml-cpu.cpp,quants.c,ops.cpp,binary-ops.cpp,unary-ops.cpp,vec.cpp,repack.cpp,traits.cpp} \
        "$w"/ggml/src/ggml-cpu/{tiled/tiled.cpp,tiled/tiled-kernel.cpp,llamafile/sgemm.cpp,arch/x86/quants.c,arch/x86/repack.cpp} \
        "$k"/src/{kokoro.cpp,core/gguf_loader.cpp,core/simd_math.cpp}; do
    rel=${src#"$root"/third_party/}
    o="$wobj/${rel//\//_}.o"
    wobjects+=("$o")
    [ -f "$o" ] && [ -z "$(find "$w" "$k" -newer "$o" -print -quit)" ] && continue
    case $src in
        *.c) cc -std=c11 "${wflags[@]}" -c "$src" -o "$o" & ;;
        # kokoro.cpp asks for F32 matrix products with a call ggml 0.26 deprecates.
        "$k"/*) c++ -std=c++17 -Wno-deprecated-declarations "${wflags[@]}" -c "$src" -o "$o" & ;;
        *) c++ -std=c++17 "${wflags[@]}" -c "$src" -o "$o" & ;;
    esac
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
if [ ${#pids[@]} -gt 0 ] || [ ! -f "$wobj/libts5speech.a" ]; then
    rm -f "$wobj/libts5speech.a"
    ar rcs "$wobj/libts5speech.a" "${wobjects[@]}"
    echo "== libts5speech.a: ${#pids[@]} files compiled in $(($(date +%s) - t0)) s"
fi

# WireGuard (ps5/wg, lwIP and wireguard-lwip), as ps5/build.sh builds it.
lw="$root/third_party/lwip/src"
wgl="$root/third_party/wireguard-lwip/src"
wgflags=(-O2 -I "$root/ps5/wg" -I "$lw/include" -I "$wgl")
mkdir -p "$work/wg"
wgobjects=()
pids=()
for src in "$lw"/core/*.c "$lw"/core/ipv4/*.c "$wgl"/*.c "$wgl"/crypto/refc/*.c "$root"/ps5/wg/wg_udp.c "$root"/ps5/wg/wg.cpp \
        "$root"/ps5/ts/*.cpp; do
    [ "$src" = "$lw/core/udp.c" ] && continue
    rel=${src#"$root"/}
    o="$work/wg/${rel//\//_}.o"
    wgobjects+=("$o")
    case $src in
        *.c) cc -std=gnu11 "${wgflags[@]}" -c "$src" -o "$o" & ;;
        *) c++ -std=c++17 -Wall -pthread "${wgflags[@]}" -I "$root/engine/src" -c "$src" -o "$o" & ;;
    esac
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

c++ -std=c++17 -O2 -Wall -pthread -DTS5_PS5=1 -DTS5_VULKAN=1 "-DTS5_DATA=\"$data\"" "-DTS5_HOME=\"$home\"" "-DTS5_STORE=\"$work/download0\"" \
    -I "$root/engine/src" -I "$root/ps5" -I "$root/ps5/ui" -I "$work/shaders" -I "$w/include" -I "$w/ggml/include" \
    -I "$k/include" -I "$root/third_party/qrcodegen" \
    "$root"/engine/src/{gguf,quants,model,cpu_backend,vk_context,vk_backend,tokenizer,engine,server,checks,tools,net,websearch,log}.cpp \
    "$root"/ps5/{title,library,runner,audio_probe,chat,chat_store,stt,voice,settings,speak,text_input,usage_store,downloads,keys,hf_search,backup,host_shim}.cpp "$root"/third_party/qrcodegen/qrcodegen.cpp \
    "$root"/ps5/ui/{canvas,display,pad,dashboard,chat_page,settings_page,home_page,logs_page}.cpp "$root"/ps5/tts/{g2p,text,numbers,spelling,synth}.cpp \
    -x assembler-with-cpp -Wa,-I"$root/ps5/ui" "$root/ps5/ui/font_data.S" -Wa,-I"$root/ps5/tts" "$root/ps5/tts/lexicon_data.S" \
    -x none "${wgobjects[@]}" "$wobj/libts5speech.a" -ldl -o "$work/eboot.bin"

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
# Short chat replies: the test model never ends one by itself.
# What was said by voice goes in the log, so the test can check it.
launch() { (cd "$work" && TS5_CHAT_MAX_TOKENS=8 TS5_VOICE_LOG_TEXT=1 exec ./eboot.bin) & title_pid=$!; }
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

# Serves also load a draft model (the model as its own drafter), proving the
# title keeps the file's token list until the vocabulary check has read it.
export TS5_DRAFT_MODEL="$home/models/$name"

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
notified "is ready. Apps can use it at http://"
grep -q "serve -m \"$home/models/$name\" -c auto " "$home/ts5.args" || fail "serving did not write ts5.args"
wait_log "context [0-9]+ \(what fits\), f16 KV cache$"
wait_log "draft model $name ready"
reply=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}')
grep -q '"completion_tokens":4' <<< "$reply" || fail "chat completion: $reply"
shot serving

echo "== square: the GPU check on the serving model, then serving again"
press square
wait_log "^ts5 check: .* PASS$"
notified "passed the GPU check"
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
grep -q "audio test: open TV 0x80260005" "$home/ts5.log" || fail "the audio test didn't try the TV"
grep -q "audio test: mic open, type 0: 0x80260109" "$home/ts5.log" || fail "the audio test didn't try the mic"
notified "the controller's mic didn't open"
wait_health

# Waits up to $3 seconds for the log to have $2 lines matching a regex.
wait_lines() {
    for _ in $(seq 1 $(($3 * 10))); do
        [ "$(grep -cE "$1" "$home/ts5.log" 2> /dev/null || true)" -ge "$2" ] && return 0
        kill -0 "$title_pid" 2> /dev/null || fail "the title ended while waiting for: $1"
        sleep 0.1
    done
    fail "fewer than $2 '$1' in the log after $3 s"
}

echo "== whispertest: no whisper model yet, so the test says so and the title carries on"
: > "$home/whispertest"
wait_lines "^ts5-title: whisper test: done$" 1 30
grep -q "^ts5-title: whisper test: no model: " "$home/ts5.log" || fail "the whisper test didn't say it has no model"
notified "speech test: no speech model"
wait_health

if [ -n "${TS5_TEST_WHISPER_MODEL:-}" ]; then
    echo "== whispertest: a whisper model in models/ transcribes jfk.wav; the dashboard doesn't list the model"
    wname=$(basename "$TS5_TEST_WHISPER_MODEL")
    case $wname in ggml-*.bin) ;; *) wname=ggml-test.bin ;; esac
    cp "$TS5_TEST_WHISPER_MODEL" "$home/models/$wname.part"
    mv "$home/models/$wname.part" "$home/models/$wname"
    cp "$root/third_party/whisper.cpp/samples/jfk.wav" "$home/whispertest.wav"
    : > "$home/whispertest"
    wait_lines "^ts5-title: whisper test: done$" 2 180
    grep "^ts5-title: whisper test: " "$home/ts5.log" | awk '/whisper test: start/ { n = NR } { line[NR] = $0 } END { for (i = n; i <= NR; i++) print "   " line[i] }'
    text=$(grep "^ts5-title: whisper test: text: " "$home/ts5.log" | tail -1)
    grep -qi "ask not what your country can do for you" <<< "$text" || fail "the transcript: ${text:-none}"
    sleep 2.5  # a scan of the models folder
    ! grep -q "^ts5-title: models: .*ggml-" "$home/ts5.log" || fail "the dashboard lists the whisper model"
    wait_health
    rm "$home/models/$wname" "$home/whispertest.wav"
fi

echo "== speaktest: no Kokoro model yet, so the test says so and the title carries on"
: > "$home/speaktest"
wait_lines "^ts5-title: speak test: done$" 1 30
grep -q "^ts5-title: speak test: no model: " "$home/ts5.log" || fail "the speak test didn't say it has no model"
notified "voice test: no voice"
wait_health

if [ -n "${TS5_TEST_KOKORO_MODEL:-}" ] && [ -n "${TS5_TEST_KOKORO_VOICE:-}" ]; then
    echo "== speaktest: Kokoro in models/ speaks a sentence; the dashboard doesn't list its files"
    cp "$TS5_TEST_KOKORO_MODEL" "$home/models/kokoro-test.gguf.part"
    mv "$home/models/kokoro-test.gguf.part" "$home/models/kokoro-test.gguf"
    cp "$TS5_TEST_KOKORO_VOICE" "$home/models/kokoro-voice-test.gguf"
    echo "Hello from TokenStation. The quick brown fox jumps over the lazy dog." > "$home/speaktest.part"
    mv "$home/speaktest.part" "$home/speaktest"
    wait_lines "^ts5-title: speak test: done$" 2 300
    grep -E "^ts5-title: (speak test|kokoro): " "$home/ts5.log" |
        awk '/speak test: start/ { n = NR } { line[NR] = $0 } END { for (i = n; i <= NR; i++) print "   " line[i] }'
    grep -qE "^ts5-title: speak test: [0-9.]+ s of speech in [0-9]+ ms, " "$home/ts5.log" || fail "the speak test spoke nothing"
    grep -q "^ts5-title: speak test: saved speaktest.wav$" "$home/ts5.log" || fail "the speak test saved no speaktest.wav"
    notified "voice test: "
    sleep 2.5  # a scan of the models folder
    ! grep -q "^ts5-title: models: .*kokoro-" "$home/ts5.log" || fail "the dashboard lists Kokoro's files"
    wait_health
    if [ -n "${TS5_TEST_WHISPER_MODEL:-}" ]; then
        echo "== whispertest: whisper hears what Kokoro said"
        cp "$TS5_TEST_WHISPER_MODEL" "$home/models/ggml-test.bin"
        cp "$home/speaktest.wav" "$home/whispertest.wav"
        : > "$home/whispertest"
        wait_lines "^ts5-title: whisper test: done$" 3 180
        text=$(grep "^ts5-title: whisper test: text: " "$home/ts5.log" | tail -1)
        echo "   $text"
        grep -qi "quick brown fox" <<< "$text" && grep -qi "lazy dog" <<< "$text" ||
            fail "whisper didn't hear the sentence Kokoro spoke: ${text:-none}"
        rm "$home/models/ggml-test.bin" "$home/whispertest.wav"
    fi
    rm "$home/models/kokoro-test.gguf" "$home/models/kokoro-voice-test.gguf" "$home/speaktest.wav"
fi

echo "== chat: a message from the PC; the TV shows the chat page and the reply streams in"
replies() { grep -cE "^ts5-title: chat: reply (done|stopped|failed)" "$home/ts5.log" || true; }
printf 'Say hello in five words.\n' > "$home/chat"
wait_log "^ts5-title: chat: message \(24 bytes\)$"
wait_log "^ts5-title: chat: reply done, [0-9]+ bytes, [0-9]+ tokens"
shot chat
first=$(grep -oE "^ts5: chatcmpl-[0-9]+ prompt [0-9]+" "$home/ts5.log" | tail -1 | awk '{print $4}')

wait_log "^ts5-title: speak: no Kokoro model in models"

echo "== chat: the next message goes with the conversation so far"
printf 'Now in French.\n' > "$home/chat"
for _ in $(seq 1 300); do [ "$(replies)" -ge 2 ] && break; sleep 0.1; done
grep -q "^ts5-title: chat: reply done" <(grep -E "^ts5-title: chat: reply" "$home/ts5.log" | tail -1) ||
    fail "the second reply: $(grep -E '^ts5-title: chat: reply' "$home/ts5.log" | tail -1)"
second=$(grep -oE "^ts5: chatcmpl-[0-9]+ prompt [0-9]+" "$home/ts5.log" | tail -1 | awk '{print $4}')
[ "$second" -gt "$first" ] || fail "the second request didn't carry the first exchange ($first, then $second tokens)"
shot chat2

echo "== chat: triangle stops a reply; square starts a new chat"
printf 'Count from one to two hundred, one number per line.\n' > "$home/chat"
wait_log "^ts5-title: chat: message \(51 bytes\)$"
sleep 0.5
press triangle
for _ in $(seq 1 100); do [ "$(replies)" -ge 3 ] && break; sleep 0.1; done
grep -q "^ts5-title: chat: reply stopped" <(grep -E "^ts5-title: chat: reply" "$home/ts5.log" | tail -1) ||
    fail "triangle didn't stop the reply: $(grep -E '^ts5-title: chat: reply' "$home/ts5.log" | tail -1)"
shot chat-stopped
press square
shot chat-new

echo "== keyboard: cross on the chat page opens the console's keyboard; what's typed goes to the chat, and a cancel sends nothing"
n=$(replies)
press cross
wait_log "^ts5-title: keyboard: open 0x00000000, user 0x00000001$"
printf '  Say hi in three words. ' > "$home/typed"
wait_log "^ts5-title: chat: message by keyboard \(22 bytes\)$"
for _ in $(seq 1 300); do [ "$(replies)" -gt "$n" ] && break; sleep 0.1; done
[ "$(replies)" -gt "$n" ] || fail "no reply to the typed message"
shot chat-typed
press cross
wait_lines "^ts5-title: keyboard: open 0x00000000, user 0x00000001$" 2 30
printf '(cancel)' > "$home/typed"
wait_log "^ts5-title: keyboard: canceled$"
sleep 0.5
[ "$(grep -c "^ts5-title: chat: message by keyboard" "$home/ts5.log")" = 1 ] || fail "a canceled keyboard sent a message"
echo "== keyboard: Enter on a USB keyboard opens it too (not one pressed just as it closed)"
sleep 1
: > "$home/keys"
wait_log "^ts5-title: keys: enter$"
wait_lines "^ts5-title: keyboard: open 0x00000000, user 0x00000001$" 3 30
grep -q "^ts5-title: keys: module 0x00000000, init 0x00000000$" "$home/ts5.log" || fail "libSceKeyboard didn't start"
printf '(cancel)' > "$home/typed"
wait_lines "^ts5-title: keyboard: canceled$" 2 30
press square

echo "== talk: R1 held on the chat page, with no speech model, says so"
press hold-r1
press release-r1
wait_log "^ts5-title: voice: No speech model"
shot talk-no-model

if [ -n "${TS5_TEST_WHISPER_MODEL:-}" ]; then
    echo "== talk: R1 held while jfk.wav plays into the mic; let go, and what was said goes to the chat"
    cp "$TS5_TEST_WHISPER_MODEL" "$home/models/ggml-test.bin"
    cp "$root/third_party/whisper.cpp/samples/jfk.wav" "$home/mic.wav"
    sleep 2.5  # a scan of the models folder
    press hold-r1
    wait_log "^ts5-title: voice: mic open 0x[0-7]"
    sleep 5
    shot talk-listening
    sleep 6  # jfk.wav is 11 s
    press release-r1
    wait_lines "^ts5-title: voice: [0-9]+ bytes of text$" 1 120
    text=$(grep "^ts5-title: voice: text: " "$home/ts5.log" | tail -1)
    grep -qi "ask not what your country can do for you" <<< "$text" || fail "what was said: ${text:-none}"
    wait_log "^ts5-title: chat: message by voice"
    for _ in $(seq 1 300); do [ "$(replies)" -ge 4 ] && break; sleep 0.1; done
    shot talk
    rm "$home/models/ggml-test.bin" "$home/mic.wav"
    press square
fi

if [ -n "${TS5_TEST_KOKORO_MODEL:-}" ] && [ -n "${TS5_TEST_KOKORO_VOICE:-}" ]; then
    echo "== spoken replies: with Kokoro's files, the reply is read aloud as it comes; triangle stops it"
    cp "$TS5_TEST_KOKORO_MODEL" "$home/models/kokoro-test.gguf"
    cp "$TS5_TEST_KOKORO_VOICE" "$home/models/kokoro-voice-test.gguf"
    n=$(replies)
    printf 'Say hello in five words.\n' > "$home/chat"
    wait_log "^ts5-title: speak: kokoro-test.gguf with the voice test, loaded in [0-9]+ ms$"
    wait_lines "^ts5-title: speak: [0-9]+ bytes, [0-9.]+ s of speech in [0-9]+ ms$" 1 180
    grep -E "^ts5-title: speak: " "$home/ts5.log" | sed 's/^/   /'
    # A PC has no TV to play it on.
    wait_log "^ts5-title: speak: the TV's audio didn't open"
    for _ in $(seq 1 300); do [ "$(replies)" -gt "$n" ] && break; sleep 0.1; done
    press triangle
    echo "== spoken replies off in Settings: Kokoro is unloaded at once, and loaded again when they're back on"
    loads=$(grep -cE "^ts5-title: speak: kokoro-test.gguf with the voice test, loaded in [0-9]+ ms$" "$home/ts5.log")
    press l1
    press down down down down
    press right
    wait_log "^ts5-title: speak: replies aren't read aloud, so Kokoro is unloaded$"
    press square
    wait_lines "^ts5-title: speak: kokoro-test.gguf with the voice test, loaded in [0-9]+ ms$" $((loads + 1)) 60
    press up up up up
    press circle
    press l1
    rm "$home/models/kokoro-test.gguf" "$home/models/kokoro-voice-test.gguf"
    press square
fi

echo "== settings: L1 again goes to Settings; right changes a setting in ts5.env, square puts back its default"
press l1
shot settings
press down
press right
wait_log "^ts5-title: settings: TS5_CHAT_FPS=15$"
grep -qx "TS5_CHAT_FPS=15" "$home/ts5.env" || fail "ts5.env: $(cat "$home/ts5.env" 2> /dev/null)"
shot settings-changed
press square
wait_log "^ts5-title: settings: TS5_CHAT_FPS cleared$"
! grep -q "TS5_CHAT_FPS" "$home/ts5.env" || fail "square left it in ts5.env: $(cat "$home/ts5.env")"
echo "== settings: cross on the Tavily key opens the keyboard; the key goes in ts5.env, never the log"
press down down down down down down down down
press cross
wait_log "^ts5-title: keyboard: open 0x00000000, user 0x00000001, a key$"
printf 'tvly-typed-on-the-tv' > "$home/typed"
wait_log "^ts5-title: settings: TAVILY_API_KEY=\\(set\\)$"
grep -qx "TAVILY_API_KEY=tvly-typed-on-the-tv" "$home/ts5.env" || fail "the typed key isn't in ts5.env"
! grep -q "tvly-typed-on-the-tv" "$home/ts5.log" || fail "the typed key is in the log"
shot settings-key
press square
press square
wait_log "^ts5-title: settings: TAVILY_API_KEY cleared$"
! grep -q "TAVILY_API_KEY" "$home/ts5.env" || fail "square twice left the key in ts5.env"
press circle
shot home-again
wait_health

echo "== stop serving: L1 to Models, cross on the model in use, cross again to confirm"
press l1
press cross
shot stop-confirm
press cross
for _ in $(seq 1 50); do health_says "no model" && break; sleep 0.1; done
health_says "no model" || fail "still serving after the confirmed stop"

echo "== restart: serves what ts5.args says, its log becomes ts5.prev.log"
: > "$home/restart"
wait_down
wait_health
grep -q "TokenStation5" "$home/ts5.prev.log" || fail "no previous log"

echo "== ts5.once: a one-off check at the next start, then the args file's model"
echo "== ts5.env: engine settings (here every matmul tiled, even one token's, an 8-bit KV cache and a Tavily key)"
echo "check -m $name -n 6" > "$home/ts5.once"
echo "TS5_TILED_MIN=1 TS5_FAST_MATVEC=0 TS5_KV=q8 TAVILY_API_KEY=tvly-host-test-secret" > "$home/ts5.env"
: > "$home/restart"
wait_down
wait_log "^ts5-title: TS5_TILED_MIN=1$"
wait_log "^ts5-title: TAVILY_API_KEY=\\(set\\)$"
! grep -q "tvly-host-test-secret" "$home/ts5.log" || fail "the Tavily key is in the log"
wait_log "KV cache .*\(q8, "
wait_log "^ts5 check: .*, 6 tokens: .*, cpu [0-9]+ ms: PASS$"
wait_health
[ ! -e "$home/ts5.once" ] || fail "ts5.once was not consumed"
rm "$home/ts5.env"

echo "== the args file: another port"
echo "serve -m \"$home/models/$name\" --port 18081" > "$home/ts5.args"
: > "$home/restart"
wait_down
port=18081
wait_health

echo "== keep-awake, and a dark screen after TS5_DIM_AFTER seconds without a button; a press brings back the dashboard"
echo "TS5_DIM_AFTER=2" > "$home/ts5.env"
: > "$home/restart"
wait_down
wait_health
wait_log "^ts5-title: keeping the console awake \(0\)$"
sleep 3
brightness() {  # mean of the screenshot's RGB, 0 to 255
    python3 -c 'import sys; d = open(sys.argv[1], "rb").read()[54:]
print(int((sum(d[0::4]) + sum(d[1::4]) + sum(d[2::4])) / (3 * len(d) / 4)))' "$work/shots/$1.bmp"
}
shot dark
[ "$(brightness dark)" -lt 2 ] || fail "the screen is not dark after 2 s without a button: $(brightness dark)"
press up
shot woken
[ "$(brightness woken)" -ge 5 ] || fail "a press did not bring the dashboard back: $(brightness woken)"
echo "== a chat message brings back a dark screen too, on the chat page"
sleep 3
printf 'Hi.\n' > "$home/chat"
wait_log "^ts5-title: chat: message \(3 bytes\)$"
shot chat-woken
[ "$(brightness chat-woken)" -ge 5 ] || fail "a chat message did not bring the screen back: $(brightness chat-woken)"
wait_log "^ts5-title: chat: reply done"
rm "$home/ts5.env"

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
