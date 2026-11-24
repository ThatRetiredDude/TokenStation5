#!/bin/sh
# Builds the TokenStation5 OS initrd: toybox (init and tools), mksh (the
# shell; toybox's own sh is not finished) and dropbear (ssh), all static
# with musl, plus os/rootfs.
#   os/build.sh OUT_DIR      -> OUT_DIR/ts5os-initrd.cpio.gz
# Needs: git, make, musl-tools, cpio, gzip, Linux headers (/usr/include/linux).
set -eu
TOYBOX_TAG=0.8.12
DROPBEAR_TAG=DROPBEAR_2025.88
MKSH_TAG=mksh-R59c
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
W=$OUT/work
mkdir -p "$W"

# musl ships no Linux headers; point it at the system's.
mkdir -p "$W/kh"
ln -sfn /usr/include/linux "$W/kh/linux"
ln -sfn /usr/include/asm-generic "$W/kh/asm-generic"
ln -sfn "/usr/include/$(gcc -print-multiarch)/asm" "$W/kh/asm"
[ -d /usr/include/mtd ] && ln -sfn /usr/include/mtd "$W/kh/mtd"

[ -d "$W/toybox" ] || git clone -q --depth 1 -b "$TOYBOX_TAG" https://github.com/landley/toybox "$W/toybox"
[ -d "$W/mksh" ] || git clone -q --depth 1 -b "$MKSH_TAG" https://github.com/MirBSD/mksh "$W/mksh"
[ -d "$W/dropbear" ] || git clone -q --depth 1 -b "$DROPBEAR_TAG" https://github.com/mkj/dropbear "$W/dropbear"

# mksh: the shell, for scripts and for people.
(cd "$W/mksh" && [ -x mksh ] || CC=musl-gcc CFLAGS=-Os LDFLAGS=-static sh Build.sh -r >/dev/null && strip mksh)

# toybox: the default set plus init, getty, dhcp and route.
(cd "$W/toybox"
 make defconfig >/dev/null
 for c in INIT GETTY DHCP ROUTE; do sed -i "s/# CONFIG_$c is not set/CONFIG_$c=y/" .config; done
 CFLAGS="-Os -isystem $W/kh" LDFLAGS=--static CC=musl-gcc make -j"$(nproc)" toybox >/dev/null)

# dropbear: key logins only, no X11, one binary.
(cd "$W/dropbear"
 cat >localoptions.h <<'OPT'
#define DROPBEAR_SVR_PASSWORD_AUTH 0
#define DROPBEAR_X11FWD 0
#define DEFAULT_PATH "/bin:/sbin"
OPT
 CC=musl-gcc CFLAGS="-Os -isystem $W/kh" ./configure --enable-static --disable-zlib --disable-lastlog \
   --disable-utmp --disable-utmpx --disable-wtmp --disable-wtmpx --disable-pututline --disable-pututxline \
   --disable-syslog >/dev/null
 make -j"$(nproc)" PROGRAMS="dropbear dropbearkey scp" MULTI=1 STATIC=1 >/dev/null
 strip dropbearmulti)

R=$W/root
rm -rf "$R"
mkdir -p "$R/bin" "$R/sbin" "$R/etc/ts5" "$R/proc" "$R/sys" "$R/dev" "$R/run" "$R/tmp" "$R/data" "$R/root" "$R/lib/modules"
cp "$W/toybox/toybox" "$R/bin/toybox"
for t in $("$R/bin/toybox"); do [ "$t" = toybox ] || ln -sf toybox "$R/bin/$t"; done
cp "$W/mksh/mksh" "$R/bin/mksh"
ln -sf mksh "$R/bin/sh"
cp "$W/dropbear/dropbearmulti" "$R/bin/dropbearmulti"
for t in dropbear dropbearkey scp; do ln -sf dropbearmulti "$R/bin/$t"; done
cp -a "$HERE/rootfs/." "$R/"
ln -sf bin/init "$R/init"
ln -sf /proc/mounts "$R/etc/mtab"
chmod 600 "$R/etc/shadow"
(cd "$R" && find . | LC_ALL=C sort | cpio -o -H newc -R 0:0 --quiet | gzip -9n) >"$OUT/ts5os-initrd.cpio.gz"
ls -l "$OUT/ts5os-initrd.cpio.gz"
