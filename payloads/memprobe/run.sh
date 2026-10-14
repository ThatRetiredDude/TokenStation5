#!/usr/bin/env bash
# Sends ts5-memprobe to the console's ELF loader and saves what it found.
#
#   payloads/memprobe/run.sh [LABEL]    -> memprobe-LABEL.txt (default: now)
#
# Read-only on the console: the payload only reads counters and the process
# table, and writes /data/ts5-memprobe.txt. Uses PS5_HOST and PS5_FTP_PORT
# from scripts/env.sh, and the loader on PS5_ELF_PORT (9021).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../../scripts/env.sh"
port=${PS5_ELF_PORT:-9021}
label=${1:-$(date +%H%M%S)}
out="memprobe-$label.txt"
python3 - "$PS5_HOST" "$port" "$here/ts5-memprobe.elf" <<'PY'
import socket, sys
host, port, elf = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with socket.create_connection((host, port), timeout=10) as s, open(elf, "rb") as f:
    s.sendall(f.read())
PY
sleep 4
curl -sS "ftp://$PS5_HOST:$PS5_FTP_PORT/data/ts5-memprobe.txt" -o "$out"
echo "memprobe: $out ($(wc -l < "$out") lines)"
