#!/usr/bin/env bash
# Downloads GGUFs listed in models.tsv to the local cache, then uploads them to
# the console's model library over FTP (about 60 MB/s on a LAN, several times
# faster than the console's own downloader).
#
#   scripts/fetch-models.sh                   # every model in models.tsv
#   scripts/fetch-models.sh qwen25-coder-7b qwen3-8b
#   NO_UPLOAD=1 scripts/fetch-models.sh ...   # download only
#   FORCE=1 scripts/fetch-models.sh qwen35-4b # one ts5 can't run yet, too
#
# A model of an architecture the engine can't run yet is
# skipped before its download, since the title would refuse to load it.
#
# The exact file is found through the Hugging Face API: the first .gguf in the
# repo whose name contains file_match (mmproj files are skipped). Set HF_TOKEN
# for gated repos.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
. "$root/scripts/env.sh"
mkdir -p "$TS5_MODEL_CACHE"
auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")

grep -v '^#' "$root/models.tsv" | while IFS=$'\t' read -r key repo match gib _ctx _kv _role; do
    [ -n "$key" ] || continue
    if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$key"; then continue; fi
    # The architectures the engine runs (supported_arch in engine/src/model.cpp).
    arch=$(curl -fsSL ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$repo" |
        python3 -c 'import json, sys; print((json.load(sys.stdin).get("gguf") or {}).get("architecture") or "")' 2> /dev/null || true)
    case $arch in
        "" | llama | mistral | qwen2 | qwen3 | gemma3 | gemma4 | qwen35) ;;
        *)
            if [ -z "${FORCE:-}" ]; then
                echo "fetch-models: $key: ts5 can't run $arch models yet; skipped (FORCE=1 fetches it anyway)" >&2
                continue
            fi
            ;;
    esac
    file=$(curl -fsSL ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$repo/tree/main" | python3 -c '
import json, sys
match = sys.argv[1].lower()
for f in json.load(sys.stdin):
    p = f["path"]
    if p.endswith(".gguf") and "mmproj" not in p.lower() and match in p.lower():
        print(p); break' "$match")
    [ -n "$file" ] || { echo "fetch-models: $key: no $match .gguf in $repo" >&2; continue; }
    dest="$TS5_MODEL_CACHE/$(basename "$file")"
    echo "fetch-models: $key -> $file (~$gib GiB)"
    curl -fL -C - --progress-bar ${auth[@]+"${auth[@]}"} -o "$dest" "https://huggingface.co/$repo/resolve/main/$file"
    if [ -z "${NO_UPLOAD:-}" ]; then
        echo "fetch-models: uploading $(basename "$dest") to $PS5_HOST:$PS5_MODEL_DIR"
        curl -sS --ftp-create-dirs -T "$dest" "ftp://$PS5_HOST:$PS5_FTP_PORT$PS5_MODEL_DIR/$(basename "$dest")"
    fi
done
