#!/usr/bin/env bash
# Downloads a PS5LM Plus release (abhi245y/PS5LM), checks its sha256, and
# uploads the PPSA99581 title folder to /data/homebrew over FTP.
#
#   PS5_HOST=192.168.1.50 scripts/install-ps5lm.sh          # latest release
#   PS5_HOST=192.168.1.50 scripts/install-ps5lm.sh v0.8.3   # a given tag
#
# Needs ftpsrv on the console (2121). Close the app first if it is running:
# replacing a running title's files fails. ShadowMountPlus registers the title
# on its next scan; then launch "PS5LM Plus" from the home screen.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
. "$root/scripts/env.sh"
repo=abhi245y/PS5LM
tag=${1:-latest}
work="$root/build/ps5lm"
mkdir -p "$work"

if [ "$tag" = latest ]; then
    api="https://api.github.com/repos/$repo/releases/latest"
else
    api="https://api.github.com/repos/$repo/releases/tags/$tag"
fi
urls=$(curl -fsSL "$api" | python3 -c '
import json, sys
for a in json.load(sys.stdin)["assets"]:
    if a["name"].startswith("ps5lm-app-") and a["name"].endswith((".zip", ".zip.sha256")):
        print(a["browser_download_url"])')
[ -n "$urls" ] || { echo "install-ps5lm: no ps5lm-app zip in $tag" >&2; exit 1; }

cd "$work"
for u in $urls; do curl -fL --progress-bar -o "$(basename "$u")" "$u"; done
zip=$(ls -t ps5lm-app-*.zip | head -1)
sha256sum -c "$zip.sha256"
rm -rf title && mkdir title && unzip -q "$zip" -d title
title=$(ls title)
echo "install-ps5lm: uploading $title from $zip to $PS5_HOST"
(cd "title/$title" && find . -type f | while read -r f; do
    curl -sS --ftp-create-dirs -T "$f" "ftp://$PS5_HOST:$PS5_FTP_PORT/data/homebrew/$title/${f#./}"
done)
curl -sS --ftp-create-dirs -Q "MKD $PS5_MODEL_DIR" "ftp://$PS5_HOST:$PS5_FTP_PORT/" -o /dev/null 2>/dev/null || true
echo "install-ps5lm: done. Launch PS5LM Plus from the home screen."
