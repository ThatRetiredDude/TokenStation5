#!/usr/bin/env bash
# Drives the TokenStation5 title on the console over FTP (ftpsrv, 2121).
#
#   scripts/ts5-ps5.sh install             upload the title from build/ps5 to /data/homebrew
#   scripts/ts5-ps5.sh model FILE          upload a model (.gguf, or whisper's ggml-*.bin) to the
#                                          title's models folder
#   scripts/ts5-ps5.sh fetch KEY...        download models.tsv models and upload them there
#   scripts/ts5-ps5.sh speech [NAME]       download whisper's ggml-NAME.bin (base.en, the default;
#                                          tiny.en, small.en ...) and upload it there, for R1 talk
#   scripts/ts5-ps5.sh voice [NAME]        download Kokoro's model and voice NAME (af_heart, the
#                                          default; am_michael, bf_emma ...) and upload them there,
#                                          for spoken replies
#   scripts/ts5-ps5.sh models              list the models on the console
#   scripts/ts5-ps5.sh serve NAME.gguf [ts5 options]
#                                          serve that model from now on, and restart
#   scripts/ts5-ps5.sh check NAME.gguf     GPU-vs-CPU logits check, as the dashboard's square
#   scripts/ts5-ps5.sh bench NAME.gguf [--pp N --tg N]
#                                          prompt/generation speed run, as its triangle
#   scripts/ts5-ps5.sh stop                stop serving
#   scripts/ts5-ps5.sh env [NAME=VALUE...] engine settings (ts5.env), then restart; none: clear
#                                          them (the Settings page's and any keys stay)
#   scripts/ts5-ps5.sh websearch [KEY|off] the Tavily key for web search (in ts5.env, keeping the
#                                          other settings), then restart; none: say if one is set
#   scripts/ts5-ps5.sh press BUTTON...     press controller buttons (up down cross square ...)
#   scripts/ts5-ps5.sh screenshot [FILE]   the TV picture, as a BMP (default screen.bmp)
#   scripts/ts5-ps5.sh audiotest [FILE]    tones on the TV and the controller, then 8 s from its
#                                          mic played back; prints the results, saves the
#                                          recording (default audiotest.wav)
#   scripts/ts5-ps5.sh chat MESSAGE...     a message on the TV's chat page; waits for the
#                                          reply there and prints how it went
#   scripts/ts5-ps5.sh whispertest [WAV]   speech to text on the console's CPU with the first
#                                          ggml-*.bin in models: WAV (16 kHz mono 16-bit; else the
#                                          last one sent, or audiotest.wav); prints times and text
#   scripts/ts5-ps5.sh speaktest [TEXT...] text to speech on the console's CPU with Kokoro,
#                                          played on the TV; prints the phonemes and times, saves
#                                          the speech (speaktest.wav)
#   scripts/ts5-ps5.sh log [prev]          the title's log (prev: the run before)
#   scripts/ts5-ps5.sh restart | quit      restart or close the running title
#   scripts/ts5-ps5.sh wait                wait for the API to answer
#
# Everything lives in the title's folder, /data/homebrew/<title ID> over FTP
# and /app0 to the title. NAME is a file in its models folder (or a full path
# as the title sees it). Launch the title from the home screen once after
# install; ShadowMountPlus lists it as TokenStation5.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
. "$root/scripts/env.sh"
ftp="ftp://$PS5_HOST:$PS5_FTP_PORT"
id=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["titleId"])' "$root/ps5/sce_sys/param.json")
dir=${TS5_DIR:-/data/homebrew/$id}

# A file for the title, whole: uploaded under another name, then renamed, since
# the title can see a file the FTP server has made but not yet filled.
put_text() {
    printf '%s\n' "$2" | curl -sS --ftp-create-dirs -T - "$ftp$dir/$1.part" -Q "-RNFR $1.part" -Q "-RNTO $1"
}
# A model as the title sees it.
console_path() { case $1 in /*) echo "$1" ;; *) echo "/app0/models/$1" ;; esac; }

# After a restart request: the old title's API until it goes (at most a few
# seconds; it may not have been running at all).
wait_down() {
    for _ in $(seq 1 20); do
        curl -sf "http://$PS5_HOST:$PS5_API_PORT/health" > /dev/null || return 0
        sleep 0.5
    done
}

wait_api() {
    echo "ts5-ps5: waiting for http://$PS5_HOST:$PS5_API_PORT"
    for _ in $(seq 1 120); do
        if curl -sf "http://$PS5_HOST:$PS5_API_PORT/health" > /dev/null; then
            echo "ts5-ps5: ready at http://$PS5_HOST:$PS5_API_PORT/v1"
            return 0
        fi
        sleep 5
    done
    echo "ts5-ps5: no answer after 10 minutes; see: scripts/ts5-ps5.sh log" >&2
    return 1
}

# Lines of the title's log matching a regex.
log_lines() { curl -sS "$ftp$dir/ts5.log" 2> /dev/null | grep -E "$1" || true; }

# A command for the running title (ts5.cmd), then its result from the log:
# the new lines matching $1, or the title's new error line.
run_cmd() {
    local want=$1 fails="^ts5-title: .*(failed: |couldn't load|ts5.cmd: )"
    shift
    local before errors
    before=$(log_lines "$want" | grep -c . || true)
    errors=$(log_lines "$fails" | grep -c . || true)
    put_text ts5.cmd "$*"
    echo "ts5-ps5: asked for: $*"
    for _ in $(seq 1 300); do
        sleep 2
        local now
        now=$(log_lines "$want")
        if [ "$(printf '%s\n' "$now" | grep -c . || true)" -gt "$before" ]; then
            printf '%s\n' "$now" | tail -n +$((before + 1))
            return 0
        fi
        if [ "$(log_lines "$fails" | grep -c . || true)" -gt "$errors" ]; then
            log_lines "$fails" | tail -1 >&2
            return 1
        fi
    done
    echo "ts5-ps5: no result after 10 minutes; see: scripts/ts5-ps5.sh log" >&2
    return 1
}

cmd=${1:-}
[ $# -gt 0 ] && shift
case $cmd in
    install)
        app="$root/build/ps5/$id"
        [ -f "$app/eboot.bin" ] || { echo "ts5-ps5: build it first: ps5/build.sh" >&2; exit 2; }
        # Files of a running title cannot be replaced.
        put_text quit "" 2> /dev/null && sleep 3 || true
        # File by file: the models, args and logs in the folder stay.
        (cd "$app" && find . -type f | sort | while read -r f; do
            curl -sS --ftp-create-dirs -T "$f" "$ftp$dir/${f#./}"
        done)
        curl -sS --ftp-create-dirs -T /dev/null "$ftp$dir/models/.keep"
        echo "ts5-ps5: installed. Launch TokenStation5 from the home screen."
        ;;
    fetch)
        PS5_MODEL_DIR="$dir/models" "$root/scripts/fetch-models.sh" "$@"
        ;;
    speech)
        # whisper.cpp's own conversions, as its models/download-ggml-model.sh fetches them.
        f="ggml-${1:-base.en}.bin"
        mkdir -p "$TS5_MODEL_CACHE"
        curl -fL -C - --progress-bar -o "$TS5_MODEL_CACHE/$f" \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$f"
        curl --ftp-create-dirs --progress-bar -T "$TS5_MODEL_CACHE/$f" "$ftp$dir/models/$f.part" -o /dev/null \
            -Q "-RNFR $f.part" -Q "-RNTO $f"
        echo "ts5-ps5: uploaded $f to $dir/models; hold R1 on the chat page to talk"
        ;;
    voice)
        # kokoro.cpp's conversions of Kokoro-82M and its voices (Apache-2.0), as
        # its models/download-model.sh fetches them.
        v=${1:-af_heart}
        v=${v#kokoro-voice-}
        v=${v%.gguf}
        mkdir -p "$TS5_MODEL_CACHE"
        hf=https://huggingface.co/simonfxr/kokoro.cpp-GGUF/resolve/main
        for f in kokoro-82m-f16.gguf "voices/kokoro-voice-$v.gguf"; do
            curl -fL -C - --progress-bar -o "$TS5_MODEL_CACHE/${f#voices/}" "$hf/$f"
        done
        for f in kokoro-82m-f16.gguf "kokoro-voice-$v.gguf"; do
            curl --ftp-create-dirs --progress-bar -T "$TS5_MODEL_CACHE/$f" "$ftp$dir/models/$f.part" -o /dev/null \
                -Q "-RNFR $f.part" -Q "-RNTO $f"
        done
        echo "ts5-ps5: uploaded kokoro-82m-f16.gguf and the voice $v to $dir/models; try: scripts/ts5-ps5.sh speaktest"
        ;;
    model)
        f=${1:?a model file}
        curl --ftp-create-dirs --progress-bar -T "$f" "$ftp$dir/models/$(basename "$f")" -o /dev/null
        echo "ts5-ps5: uploaded $(basename "$f") to $dir/models"
        ;;
    models)
        # A LIST: ftpsrv answers NLST (curl --list-only) with 502.
        curl -sS "$ftp$dir/models/" | awk '{ $1 = $2 = $3 = $4 = $5 = $6 = $7 = $8 = ""; sub(/^ +/, ""); print }' |
            grep -i '\.gguf$' || true
        ;;
    serve)
        m=$(console_path "${1:?model name}")
        shift
        put_text ts5.args "serve -m $m $*"
        put_text restart ""
        wait_down
        wait_api
        ;;
    check)
        run_cmd "^ts5 check: " check -m "$(console_path "${1:?model name}")" "${@:2}"
        ;;
    bench)
        m=$(console_path "${1:?model name}")
        run_cmd "^\\| ${m##*/} \\| " bench -m "$m" "${@:2}"
        ;;
    stop)
        put_text ts5.cmd stop
        echo "ts5-ps5: asked the title to stop serving"
        ;;
    env)
        # ts5.env is NAME=VALUE words: what the Settings page sets and any
        # keys stay, the engine's are replaced.
        current=$(curl -sS "$ftp$dir/ts5.env" 2> /dev/null || true)
        page='TS5_CHAT_MAX_TOKENS|TS5_CHAT_FPS|TS5_WHISPER_MODEL|TS5_WHISPER_THREADS|TS5_SPEAK|TS5_SPEAK_OUT|TS5_TTS_VOICE|TS5_TTS_SPEED|TS5_TTS_THREADS|TS5_KEEP_AWAKE|TS5_DIM_AFTER'
        kept=$(printf '%s\n' $current | grep -E "^($page|[A-Z0-9_]*_(KEY|TOKEN))=" | tr '\n' ' ' | sed 's/ *$//' || true)
        put_text ts5.env "${kept:+$kept }$*"
        put_text restart ""
        wait_down
        wait_api
        ;;
    websearch)
        # ts5.env is NAME=VALUE words; this one is TAVILY_API_KEY.
        current=$(curl -sS "$ftp$dir/ts5.env" 2> /dev/null || true)
        others=$(printf '%s\n' $current | grep -v '^TAVILY_API_KEY=' | tr '\n' ' ' | sed 's/ *$//' || true)
        if [ $# -eq 0 ]; then
            printf '%s\n' $current | grep -q '^TAVILY_API_KEY=.' && echo "ts5-ps5: web search is on (a Tavily key is set)" ||
                echo "ts5-ps5: web search is off (no Tavily key); scripts/ts5-ps5.sh websearch tvly-..."
            exit 0
        fi
        case $1 in
            off) put_text ts5.env "$others" ;;
            tvly-*) put_text ts5.env "${others:+$others }TAVILY_API_KEY=$1" ;;
            *) echo "ts5-ps5: a Tavily key starts with tvly- (https://app.tavily.com)" >&2; exit 2 ;;
        esac
        put_text restart ""
        wait_down
        wait_api
        curl -sS "http://$PS5_HOST:$PS5_API_PORT/v1/models" | python3 -c '
import json, sys
for m in json.load(sys.stdin)["data"]:
    c = m.get("capabilities", {})
    print("ts5-ps5: %s: tool calls %s, web search %s" % (m["id"], c.get("tool_format", "?"),
          "on" if c.get("web_search") else "off"))'
        ;;
    press)
        put_text press "$*"
        ;;
    screenshot)
        out=${1:-screen.bmp}
        put_text screenshot ""
        sleep 2
        curl -sS "$ftp$dir/screen.bmp" -o "$out"
        echo "ts5-ps5: $out"
        ;;
    audiotest)
        out=${1:-audiotest.wav}
        before=$(log_lines "^ts5-title: audio test: done$" | grep -c . || true)
        put_text audiotest ""
        echo "ts5-ps5: audio test started: three tones (TV, controller, headset jack), then talk into the controller after the beep, for 8 s"
        for _ in $(seq 1 45); do
            sleep 2
            [ "$(log_lines "^ts5-title: audio test: done$" | grep -c . || true)" -gt "$before" ] && break
        done
        log_lines "^ts5-title: audio test: " | awk '/audio test: start/ { n = NR } { line[NR] = $0 } END { for (i = n; i <= NR; i++) print line[i] }'
        curl -sSf "$ftp$dir/audiotest.wav" -o "$out" 2> /dev/null && echo "ts5-ps5: $out" || true
        ;;
    chat)
        [ $# -gt 0 ] || { echo "ts5-ps5: chat needs a message" >&2; exit 2; }
        replies() { log_lines "^ts5-title: chat: (reply |not sent)" | grep -c . || true; }
        before=$(replies)
        put_text chat "$*"
        echo "ts5-ps5: sent; the TV shows the chat page and the reply as it comes"
        for _ in $(seq 1 300); do
            sleep 2
            [ "$(replies)" -gt "$before" ] && break
        done
        log_lines "^ts5-title: chat: (reply |not sent)" | tail -1
        ;;
    whispertest)
        if [ $# -gt 0 ]; then
            curl -sS --ftp-create-dirs -T "$1" "$ftp$dir/whispertest.wav.part" \
                -Q "-RNFR whispertest.wav.part" -Q "-RNTO whispertest.wav"
            echo "ts5-ps5: uploaded $(basename "$1") as whispertest.wav"
        fi
        before=$(log_lines "^ts5-title: whisper test: done$" | grep -c . || true)
        put_text whispertest ""
        echo "ts5-ps5: whisper test started: loading the model, then transcribing"
        for _ in $(seq 1 300); do
            sleep 2
            [ "$(log_lines "^ts5-title: whisper test: done$" | grep -c . || true)" -gt "$before" ] && break
        done
        log_lines "^ts5-title: whisper test: " | awk '/whisper test: start/ { n = NR } { line[NR] = $0 } END { for (i = n; i <= NR; i++) print line[i] }'
        ;;
    speaktest)
        before=$(log_lines "^ts5-title: speak test: done$" | grep -c . || true)
        put_text speaktest "$*"
        echo "ts5-ps5: speak test started: loading the model, then speaking on the TV"
        for _ in $(seq 1 300); do
            sleep 2
            [ "$(log_lines "^ts5-title: speak test: done$" | grep -c . || true)" -gt "$before" ] && break
        done
        log_lines "^ts5-title: (speak test|kokoro): " | awk '/speak test: start/ { n = NR } { line[NR] = $0 } END { for (i = n; i <= NR; i++) print line[i] }'
        if curl -fsS -o speaktest.wav "$ftp$dir/speaktest.wav" 2> /dev/null; then
            echo "ts5-ps5: saved the speech as speaktest.wav"
        fi
        ;;
    log)
        curl -sS "$ftp$dir/ts5${1:+.$1}.log"
        ;;
    restart | quit)
        put_text "$cmd" ""
        echo "ts5-ps5: asked the title to $cmd"
        ;;
    wait)
        wait_api
        ;;
    *)
        sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
