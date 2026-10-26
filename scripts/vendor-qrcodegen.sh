#!/usr/bin/env bash
# Copies Project Nayuki's QR Code generator (C++, MIT) into
# third_party/qrcodegen, at a pinned commit, and writes its VENDOR.md. The
# title draws the WireGuard join page's address as a QR code on the TV
# (ps5/ui/settings_page.cpp), so a phone can open it without typing.
#
#   scripts/vendor-qrcodegen.sh
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo=https://github.com/nayuki/QR-Code-generator
commit=3c6d0b3cefb4e049dc337e82237c9644399716a8

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q "$tmp/qr"
git -C "$tmp/qr" fetch -q --depth 1 "$repo" "$commit"
git -C "$tmp/qr" checkout -q FETCH_HEAD
[ "$(git -C "$tmp/qr" rev-parse HEAD)" = "$commit" ] || { echo "vendor: $repo is not at $commit" >&2; exit 1; }

dest="$root/third_party/qrcodegen"
rm -rf "$dest"
mkdir -p "$dest"
cp "$tmp/qr/cpp/qrcodegen.hpp" "$tmp/qr/cpp/qrcodegen.cpp" "$dest/"
cat > "$dest/VENDOR.md" << VEOF
# QR Code generator, vendored

[QR Code generator library](https://github.com/nayuki/QR-Code-generator) (C++),
commit \`$commit\`. MIT, "Copyright (c) Project Nayuki"; the license is at the
top of each file. NOTICE.md at the top of this repo lists it.

The title draws the WireGuard join page's address as a QR code on the TV
(\`ps5/ui/settings_page.cpp\`), so a phone's camera can open the page.
Unchanged; \`scripts/vendor-qrcodegen.sh\` copies it.

Kept: \`cpp/qrcodegen.hpp\` and \`cpp/qrcodegen.cpp\`. Left out: the other
languages, the demo.
VEOF
