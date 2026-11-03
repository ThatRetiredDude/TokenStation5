#!/usr/bin/env bash
# The Arch package, from linux/build-deb.sh's tokenstation-root.tar.gz
# (run that first, on Debian 12). Run as a normal user on Arch (makepkg
# won't run as root).
#
#   linux/build-arch.sh VERSION [OUTDIR]
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version=${1:?version}
version=${version#linux-v}
out=${2:-$root/build/linux}
work=$(mktemp -d)
cp "$root/linux/arch/PKGBUILD" "$root/linux/arch/tokenstation.install" "$out/tokenstation-root.tar.gz" "$work/"
(cd "$work" && TS5_PKGVER="$version" PKGDEST="$out" makepkg -f --nodeps)
rm -rf "$work"
ls "$out"/tokenstation-*.pkg.tar.*
