#!/usr/bin/env bash
# Builds the TokenStation Linux package: the full app (tokenstation: the TV
# dashboard, chat, voice, keys, Tailscale, downloads; linux/build.sh), the
# engine's command line (ts5), the API-only service, the TV service, and the
# PS5 tools (tokenstation-cu, tokenstation-memory, and the Settings page's
# helper with its sudo rule).
#
#   linux/build-deb.sh [VERSION] [OUTDIR]
#
# VERSION defaults to 0.0.0+<short commit>; OUTDIR to build/linux. Needs
# what linux/build.sh needs (a C++17 compiler, glslang-tools, libvulkan-dev,
# libsdl2-dev), plus cmake and dpkg-deb.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version=${1:-0.0.0+$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo local)}
version=${version#linux-v}
out=${2:-$root/build/linux}
arch=$(dpkg --print-architecture)
pkg=$out/tokenstation_${version}_${arch}
doc=$pkg/usr/share/doc/tokenstation

TS5_VERSION="linux-$version" TS5_LINUX_OUT="$out" bash "$root/linux/build.sh"
cmake -S "$root/linux/cu-test" -B "$out/cu-test" -DCMAKE_BUILD_TYPE=Release
cmake --build "$out/cu-test" -j"$(nproc)"

rm -rf "$pkg"
install -Dm755 "$out/tokenstation" "$pkg/usr/bin/tokenstation"
install -Dm755 "$out/ts5" "$pkg/usr/bin/ts5"
install -Dm755 "$root/linux/files/ts5-tune" "$pkg/usr/bin/ts5-tune"
install -Dm755 "$root/linux/files/start.sh" "$pkg/usr/lib/tokenstation/start.sh"
install -Dm755 "$root/linux/files/boost.sh" "$pkg/usr/lib/tokenstation/boost.sh"
install -Dm755 "$out/cu-test/gpu-selftest" "$pkg/usr/lib/tokenstation/gpu-selftest"
install -Dm755 "$root/linux/cu-test/tokenstation-cu" "$pkg/usr/bin/tokenstation-cu"
install -Dm755 "$root/linux/files/tokenstation-memory" "$pkg/usr/bin/tokenstation-memory"
install -Dm755 "$root/linux/files/hw" "$pkg/usr/lib/tokenstation/hw"
install -Dm755 "$root/linux/files/system-setup" "$pkg/usr/lib/tokenstation/system-setup"
install -Dm440 "$root/linux/files/tokenstation.sudoers" "$pkg/etc/sudoers.d/tokenstation"
install -Dm644 "$root/linux/files/tokenstation.service" "$pkg/lib/systemd/system/tokenstation.service"
install -Dm644 "$root/linux/files/tokenstation-tv.service" "$pkg/lib/systemd/system/tokenstation-tv.service"
install -Dm644 "$root/linux/files/tokenstation.default" "$pkg/etc/default/tokenstation"
install -Dm644 "$root/linux/files/tokenstation.desktop" "$pkg/usr/share/applications/tokenstation.desktop"
install -Dm644 "$root/ps5/sce_sys/icon0.png" "$pkg/usr/share/icons/hicolor/512x512/apps/tokenstation.png"
install -Dm644 "$root/LICENSE.md" "$doc/LICENSE.md"
install -Dm644 "$root/ps5/LICENSE" "$doc/LICENSE-GPL-3.0"
install -Dm644 "$root/NOTICE.md" "$doc/NOTICE.md"
install -Dm644 "$root/linux/README.md" "$doc/README.md"
install -Dm644 "$root/linux/cu-test/README.md" "$doc/cu-test/README.md"
install -Dm644 "$root/linux/cu-test/LICENSE" "$doc/cu-test/LICENSE"
cp -r "$root/licenses" "$doc/licenses"
git -C "$root" rev-parse HEAD > "$doc/SOURCE.txt" 2>/dev/null || echo unknown > "$doc/SOURCE.txt"

mkdir -p "$pkg/DEBIAN"
sed -e "s/@VERSION@/$version/" -e "s/@ARCH@/$arch/" "$root/linux/debian/control" > "$pkg/DEBIAN/control"
install -m755 "$root/linux/debian/postinst" "$pkg/DEBIAN/postinst"
install -m755 "$root/linux/debian/prerm" "$pkg/DEBIAN/prerm"
install -m755 "$root/linux/debian/postrm" "$pkg/DEBIAN/postrm"
echo /etc/default/tokenstation > "$pkg/DEBIAN/conffiles"

command -v visudo >/dev/null && visudo -cqf "$pkg/etc/sudoers.d/tokenstation"
dpkg-deb --root-owner-group --build "$pkg" "$pkg.deb"
# The same files for the Fedora and Arch packages (linux/build-rpm.sh,
# linux/build-arch.sh), with the services under /usr/lib, as those expect.
rootdir=$out/tokenstation-root
rm -rf "$rootdir"
mkdir -p "$rootdir"
cp -a "$pkg/usr" "$pkg/etc" "$rootdir/"
mkdir -p "$rootdir/usr/lib/systemd"
cp -a "$pkg/lib/systemd/system" "$rootdir/usr/lib/systemd/"
tar --owner=0 --group=0 -C "$rootdir" -czf "$out/tokenstation-root.tar.gz" usr etc

# The portable build, for systems without apt or with a read-only /usr
# (Bazzite, SteamOS): the same programs, and install.sh to put them in
# ~/.local.
port=$out/tokenstation-${version}-linux-${arch}
rm -rf "$port"
install -Dm755 "$out/tokenstation" "$port/bin/tokenstation"
install -Dm755 "$out/ts5" "$port/bin/ts5"
install -Dm755 "$root/linux/files/ts5-tune" "$port/bin/ts5-tune"
install -Dm755 "$root/linux/cu-test/tokenstation-cu" "$port/bin/tokenstation-cu"
install -Dm755 "$root/linux/files/tokenstation-memory" "$port/bin/tokenstation-memory"
install -Dm755 "$root/linux/files/hw" "$port/lib/hw"
install -Dm755 "$out/cu-test/gpu-selftest" "$port/lib/gpu-selftest"
install -Dm644 "$root/linux/files/tokenstation.desktop" "$port/share/tokenstation.desktop"
install -Dm644 "$root/ps5/sce_sys/icon0.png" "$port/share/tokenstation.png"
install -Dm755 "$root/linux/files/install.sh" "$port/install.sh"
cp -r "$doc" "$port/doc"
tar -C "$out" -czf "$port.tar.gz" "$(basename "$port")"
echo "built $pkg.deb and $port.tar.gz"
