#!/usr/bin/env bash
# Builds the TokenStation Linux package: the ts5 engine (Vulkan) plus a
# service that serves the OpenAI-style API on boot.
#
#   linux/build-deb.sh [VERSION] [OUTDIR]
#
# VERSION defaults to 0.0.0+<short commit>; OUTDIR to build/linux. Needs
# cmake, a C++17 compiler, glslang-tools, libvulkan-dev and dpkg-deb.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version=${1:-0.0.0+$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo local)}
version=${version#linux-v}
out=${2:-$root/build/linux}
arch=$(dpkg --print-architecture)
build=$out/engine
pkg=$out/tokenstation_${version}_${arch}
doc_cu=$pkg/usr/share/doc/tokenstation/cu-test

cmake -S "$root/engine" -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j"$(nproc)"
cmake -S "$root/linux/cu-test" -B "$out/cu-test" -DCMAKE_BUILD_TYPE=Release
cmake --build "$out/cu-test" -j"$(nproc)"

rm -rf "$pkg"
install -Dm755 "$build/ts5" "$pkg/usr/bin/ts5"
install -Dm755 "$root/linux/files/start.sh" "$pkg/usr/lib/tokenstation/start.sh"
install -Dm755 "$out/cu-test/gpu-selftest" "$pkg/usr/lib/tokenstation/gpu-selftest"
install -Dm755 "$root/linux/cu-test/tokenstation-cu" "$pkg/usr/bin/tokenstation-cu"
install -Dm644 "$root/linux/cu-test/LICENSE" "$doc_cu/LICENSE"
install -Dm755 "$root/linux/files/boost.sh" "$pkg/usr/lib/tokenstation/boost.sh"
install -Dm644 "$root/linux/files/tokenstation.service" "$pkg/lib/systemd/system/tokenstation.service"
install -Dm644 "$root/linux/files/tokenstation.default" "$pkg/etc/default/tokenstation"
doc=$pkg/usr/share/doc/tokenstation
install -Dm644 "$root/LICENSE.md" "$doc/LICENSE.md"
install -Dm644 "$root/NOTICE.md" "$doc/NOTICE.md"
install -Dm644 "$root/linux/README.md" "$doc/README.md"
cp -r "$root/licenses" "$doc/licenses"
git -C "$root" rev-parse HEAD > "$doc/SOURCE.txt" 2>/dev/null || echo unknown > "$doc/SOURCE.txt"

mkdir -p "$pkg/DEBIAN"
sed -e "s/@VERSION@/$version/" -e "s/@ARCH@/$arch/" "$root/linux/debian/control" > "$pkg/DEBIAN/control"
install -m755 "$root/linux/debian/postinst" "$pkg/DEBIAN/postinst"
install -m755 "$root/linux/debian/prerm" "$pkg/DEBIAN/prerm"
install -m755 "$root/linux/debian/postrm" "$pkg/DEBIAN/postrm"
echo /etc/default/tokenstation > "$pkg/DEBIAN/conffiles"

dpkg-deb --root-owner-group --build "$pkg" "$pkg.deb"
tar -C "$pkg" -czf "$out/tokenstation_${version}_${arch}.tar.gz" usr lib etc
echo "built $pkg.deb"
