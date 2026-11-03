#!/usr/bin/env bash
# Puts together what goes to Steam: the Linux build and its launcher, in
# build/steam/linux/depot, ready for steam/steampipe/upload.sh.
#
#   steam/linux/build-depot.sh VERSION        builds first (linux/build.sh)
#   TS5_PREBUILT=build/linux steam/linux/build-depot.sh VERSION
#
# Build on Debian 12 or in the Steam Runtime 3.0 (sniper) SDK, which is
# Debian 12: the depot must need nothing newer than its C library (2.36).
# The depot has no root helper, sudo rule or service: Steam installs as the
# user, in a container, and can't install those.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
version=${1:?a version, like 0.1.0}
out="$root/build/steam/linux"
depot="$out/depot"
bin=${TS5_PREBUILT:-}
if [ -z "$bin" ]; then
    TS5_VERSION="$version" "$root/linux/build.sh"
    bin="$root/build/linux"
fi
bin=$(cd "$bin" && pwd)

rm -rf "$depot"
mkdir -p "$depot/bin" "$depot/licenses"
install -m755 "$root/steam/linux/launch.sh" "$depot/launch.sh"
install -m755 "$bin/tokenstation" "$depot/bin/tokenstation"
install -m755 "$bin/ts5" "$depot/bin/ts5"
install -m644 "$root/LICENSE.md" "$depot/LICENSE.md"
install -m644 "$root/LICENSING.md" "$depot/LICENSING.md"
install -m644 "$root/ps5/LICENSE" "$depot/LICENSE-GPL-3.0"
install -m644 "$root/NOTICE.md" "$depot/NOTICE.md"
install -m644 "$root/linux/README.md" "$depot/README.md"
cp -r "$root/licenses/." "$depot/licenses/"

# The app links GPL-3.0 code (ps5/), so its source goes with it: a written
# offer, and the exact commit it was built from.
commit=$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)
cat > "$depot/SOURCE.txt" <<TXT
TokenStation5 $version for Linux (Steam)

Source code: https://github.com/ThatRetiredDude/TokenStation5
This build: commit $commit

The app as a whole is GPL-3.0-or-later (LICENSE-GPL-3.0), because it is
built from the ps5/ folder; LICENSING.md says which parts are under which
license, and NOTICE.md lists the third-party code inside it. The source of
this exact build stays available at the link above for at least three years.
TXT

# Nothing newer than Debian 12's C library, so it runs in sniper.
if objdump -T "$depot/bin/tokenstation" "$depot/bin/ts5" | grep -E "GLIBC_2\.(3[7-9]|[4-9][0-9])"; then
    echo "build-depot: the build needs a C library newer than 2.36; build it on Debian 12 or in the sniper SDK" >&2
    exit 1
fi
"$depot/bin/tokenstation" --version
echo "$version" > "$out/version"
echo "build-depot: $depot ($(du -sh "$depot" | cut -f1))"
