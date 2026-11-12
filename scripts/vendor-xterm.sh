#!/usr/bin/env bash
# Copies xterm.js and its fit addon (MIT) into third_party/xterm, from npm's
# published packages at pinned versions, checked by SHA-256, and writes its
# VENDOR.md. ts5 serve's browser terminal (engine/src/web_serve.cpp) draws
# ts5-tui with them; they are built into ts5.
#
#   scripts/vendor-xterm.sh
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
xterm_v=5.5.0
xterm_sha=bd954fa721872170188cc5d7e83e88db3c83c9a18a4e8d24c2783d26491f59d2
fit_v=0.10.0
fit_sha=917ac44972453d5eed52edc1e50260c76398ce48cf2290c2e60671102bba0b33

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fetch() {  # url sha256 file
    curl -fsSL -o "$tmp/$3" "$1"
    echo "$2  $tmp/$3" | sha256sum -c --quiet - || { echo "vendor: $1 is not the expected file" >&2; exit 1; }
    mkdir -p "$tmp/${3%.tgz}"
    tar -xzf "$tmp/$3" -C "$tmp/${3%.tgz}"
}
fetch "https://registry.npmjs.org/@xterm/xterm/-/xterm-$xterm_v.tgz" "$xterm_sha" xterm.tgz
fetch "https://registry.npmjs.org/@xterm/addon-fit/-/addon-fit-$fit_v.tgz" "$fit_sha" fit.tgz

dest="$root/third_party/xterm"
rm -rf "$dest"
mkdir -p "$dest"
cp "$tmp/xterm/package/lib/xterm.js" "$tmp/xterm/package/css/xterm.css" "$dest/"
cp "$tmp/xterm/package/LICENSE" "$dest/LICENSE"
cp "$tmp/fit/package/lib/addon-fit.js" "$dest/"
cp "$tmp/fit/package/LICENSE" "$dest/LICENSE-addon-fit"
cat > "$dest/VENDOR.md" << VEOF
# xterm.js, vendored

[xterm.js](https://github.com/xtermjs/xterm.js) $xterm_v (npm \`@xterm/xterm\`,
tarball SHA-256 \`$xterm_sha\`) and its fit addon $fit_v (npm \`@xterm/addon-fit\`,
SHA-256 \`$fit_sha\`). MIT: [LICENSE](LICENSE) and
[LICENSE-addon-fit](LICENSE-addon-fit). NOTICE.md at the top of this repo lists them.

ts5 serve's browser terminal (\`engine/src/web_serve.cpp\`, Linux) shows
ts5-tui in a web page with them; they are built into \`ts5\` and served from
\`/ts5/web/\`. Unchanged; \`scripts/vendor-xterm.sh\` copies them.

Kept: \`lib/xterm.js\`, \`css/xterm.css\`, the addon's \`lib/addon-fit.js\`, the
licenses. Left out: the source maps, the type definitions, the sources.
VEOF
