#!/usr/bin/env bash
# Drives the TokenStation5 title on the console over FTP (ftpsrv, 2121).
#
#   scripts/ts5-ps5.sh install             upload the title from build/ps5 to /data/homebrew
#   scripts/ts5-ps5.sh model FILE.gguf     upload a model to the title's models folder
#   scripts/ts5-ps5.sh models              list the models on the console
#   scripts/ts5-ps5.sh serve NAME.gguf [ts5 options]
#                                          serve that model from now on, and restart
#   scripts/ts5-ps5.sh check NAME.gguf     one-off GPU-vs-CPU logits check, then serve again
#   scripts/ts5-ps5.sh bench NAME.gguf     one-off prompt/generation speed run, then serve again
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

put_text() { printf '%s\n' "$2" | curl -sS --ftp-create-dirs -T - "$ftp$dir/$1"; }
# A model as the title sees it.
console_path() { case $1 in /*) echo "$1" ;; *) echo "/app0/models/$1" ;; esac; }

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

# A one-off command for the next start, then a restart; the log of that run
# becomes ts5.prev.log when the title restarts into serving.
once() {
    put_text ts5.once "$*"
    put_text restart ""
    echo "ts5-ps5: asked for: ts5 $*"
    echo "ts5-ps5: if the title is not running, launch it from the home screen"
    wait_api || true
    curl -sS "$ftp$dir/ts5.prev.log" | grep -E "^ts5 check|^\| " || curl -sS "$ftp$dir/ts5.prev.log" | tail -20
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
    model)
        f=${1:?a .gguf file}
        curl --ftp-create-dirs --progress-bar -T "$f" "$ftp$dir/models/$(basename "$f")" -o /dev/null
        echo "ts5-ps5: uploaded $(basename "$f") to $dir/models"
        ;;
    models)
        curl -sS --list-only "$ftp$dir/models/" | grep -i '\.gguf$' || true
        ;;
    serve)
        m=$(console_path "${1:?model name}")
        shift
        put_text ts5.args "serve -m $m $*"
        put_text restart ""
        wait_api
        ;;
    check | bench)
        once "$cmd" -m "$(console_path "${1:?model name}")" "${@:2}"
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
        sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
