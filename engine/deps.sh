#!/usr/bin/env bash
# Fetches what ts5's HTTPS client (engine/src/net.cpp) is built from, pinned
# and hash-checked, into DIR (default: .deps at the repo root). Both the CMake
# build and ps5/build.sh run it; neither is stored in this repo (NOTICE.md).
#
#   DIR/mbedtls-3.6.7/     Mbed TLS, Apache-2.0
#   DIR/cacert-2026.07.22.pem  the Mozilla CA bundle as certifi ships it, MPL-2.0
#
#   engine/deps.sh [DIR]
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
deps=${1:-$root/.deps}
mbedtls_version=3.6.7
mbedtls_sha256=a7e8bcbec0e6f761b4af24f25677626b35f762f68eef79c08677a363212d11f6
certifi_version=2026.07.22
certifi_sha256=9cc2a774b5198dcff14d9be1e66091f538975d867ce029a96bce15a55dfd730f
mkdir -p "$deps"

fetch() {  # url sha256 out
    local tmp="$3.part"
    curl -fsSL --retry 3 -o "$tmp" "$1"
    echo "$2  $tmp" | sha256sum -c --quiet - || { rm -f "$tmp"; echo "deps: $1 does not match its hash" >&2; exit 1; }
    mv "$tmp" "$3"
}

if [ ! -f "$deps/mbedtls-$mbedtls_version/include/mbedtls/ssl.h" ]; then
    fetch "https://github.com/Mbed-TLS/mbedtls/releases/download/mbedtls-$mbedtls_version/mbedtls-$mbedtls_version.tar.bz2" \
        "$mbedtls_sha256" "$deps/mbedtls-$mbedtls_version.tar.bz2"
    # The library and its headers only: the tests and docs are most of it.
    tar -xjf "$deps/mbedtls-$mbedtls_version.tar.bz2" -C "$deps" \
        "mbedtls-$mbedtls_version/include" "mbedtls-$mbedtls_version/library" "mbedtls-$mbedtls_version/LICENSE"
    rm -f "$deps/mbedtls-$mbedtls_version.tar.bz2"
fi
if [ ! -f "$deps/cacert-$certifi_version.pem" ]; then
    fetch "https://raw.githubusercontent.com/certifi/python-certifi/$certifi_version/certifi/cacert.pem" \
        "$certifi_sha256" "$deps/cacert-$certifi_version.pem"
fi
