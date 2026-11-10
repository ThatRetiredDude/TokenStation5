#!/usr/bin/env bash
# The Fedora package, from linux/build-deb.sh's tokenstation-root.tar.gz
# (run that first, on Debian 12). Needs rpm-build.
#
#   linux/build-rpm.sh VERSION [OUTDIR]
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version=${1:?version}
version=${version#linux-v}
out=${2:-$root/build/linux}
top=$(mktemp -d)
mkdir -p "$top/SOURCES" "$top/SPECS"
cp "$out/tokenstation-root.tar.gz" "$top/SOURCES/"
cp "$root/linux/rpm/tokenstation.spec" "$top/SPECS/"
rpmbuild -bb --define "_topdir $top" --define "ts_version $version" "$top/SPECS/tokenstation.spec"
cp "$top"/RPMS/*/tokenstation-*.rpm "$out/"
rm -rf "$top"
ls "$out"/tokenstation-*.rpm
