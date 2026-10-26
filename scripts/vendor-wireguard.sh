#!/usr/bin/env bash
# Copies what the title's WireGuard tunnel (ps5/wg) is built from into
# third_party, at pinned commits, with their licenses:
#
#   third_party/lwip/            lwIP 2.2.1, the TCP/IP stack inside the tunnel
#                                (its core and IPv4 only), BSD-3-Clause
#   third_party/wireguard-lwip/  WireGuard as an lwIP network interface, with
#                                its portable C crypto, BSD-3-Clause
#
# One change of ours: wireguard-lwip's peer limit, 1 upstream, becomes 16
# (src/wireguard-platform.h, marked "ts5"), so several phones and laptops can
# each have their own key. Then it writes each folder's VENDOR.md.
#
#   scripts/vendor-wireguard.sh
#
# To update: change the commits below, run it, build the title and test the
# tunnel on the PS5 (ps5/README.md, "Away from home with WireGuard").
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lwip_repo=https://github.com/lwip-tcpip/lwip
lwip_tag=STABLE-2_2_1_RELEASE
lwip_commit=77dcd25a72509eb83f72b033d219b1d40cd8eb95
wg_repo=https://github.com/smartalock/wireguard-lwip
wg_commit=c54f20dbe76ac8b3411ad21e0ed7deea6f0cfd4d
max_peers=16

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
clone() {  # repo commit dir
    git init -q "$3"
    git -C "$3" fetch -q --depth 1 "$1" "$2"
    git -C "$3" checkout -q FETCH_HEAD
    [ "$(git -C "$3" rev-parse HEAD)" = "$2" ] || { echo "vendor: $1 is not at $2" >&2; exit 1; }
}
clone "$lwip_repo" "$lwip_commit" "$tmp/lwip"
clone "$wg_repo" "$wg_commit" "$tmp/wg"

# lwIP: the core and IPv4 (no IPv6, no Ethernet, no apps), its headers, the license.
dest="$root/third_party/lwip"
rm -rf "$dest"
mkdir -p "$dest/src/core/ipv4" "$dest/src/include"
cp "$tmp/lwip/COPYING" "$dest/"
cp "$tmp/lwip"/src/core/*.c "$dest/src/core/"
cp "$tmp/lwip"/src/core/ipv4/*.c "$dest/src/core/ipv4/"
cp -r "$tmp/lwip/src/include/lwip" "$dest/src/include/"
rm -rf "$dest/src/include/lwip/apps"
# The core includes these whatever the options; nothing of theirs is built.
cp -r "$tmp/lwip/src/include/netif" "$dest/src/include/"
cat > "$dest/VENDOR.md" << EOF
# lwIP, vendored

[lwIP]($lwip_repo) 2.2.1 (tag \`$lwip_tag\`, commit \`$lwip_commit\`).
BSD-3-Clause: [COPYING](COPYING), "Copyright (c) 2001, 2002 Swedish Institute of
Computer Science". NOTICE.md at the top of this repo lists it.

It is the TCP/IP stack inside the title's WireGuard tunnel (\`ps5/wg\`): the
PS5 gives a title no network interface of its own to make, so the tunnel's
packets are taken apart here, in the title. \`ps5/wg/lwipopts.h\` and
\`ps5/wg/arch/cc.h\` configure it (no OS: one thread drives it). Unchanged;
\`scripts/vendor-wireguard.sh\` copies it.

Kept: \`src/core\` and \`src/core/ipv4\` (IPv4, TCP, the buffers), the headers
in \`src/include/lwip\` but its \`apps\`, and those in \`src/include/netif\`, which the
core includes whatever the options (none of their code is kept). Left out: IPv6, the Ethernet, PPP and
6LoWPAN interfaces, the apps (HTTP, MQTT, SNMP ...), the ports, tests and docs.
EOF

# wireguard-lwip: the protocol, the lwIP interface and the portable C crypto.
dest="$root/third_party/wireguard-lwip"
rm -rf "$dest"
mkdir -p "$dest/src/crypto/refc"
cp "$tmp/wg/LICENSE" "$dest/"
cp "$tmp/wg"/src/{wireguard.c,wireguard.h,wireguardif.c,wireguardif.h,wireguard-platform.h,crypto.c,crypto.h} "$dest/src/"
cp "$tmp/wg"/src/crypto/refc/* "$dest/src/crypto/refc/"
sed -i "s|^#define WIREGUARD_MAX_PEERS 1\$|#define WIREGUARD_MAX_PEERS $max_peers  // ts5: 1 upstream|" "$dest/src/wireguard-platform.h"
grep -q "WIREGUARD_MAX_PEERS $max_peers" "$dest/src/wireguard-platform.h" || { echo "vendor: the peer limit wasn't changed" >&2; exit 1; }
cat > "$dest/VENDOR.md" << EOF
# wireguard-lwip, vendored

[wireguard-lwip]($wg_repo), commit \`$wg_commit\`. BSD-3-Clause:
[LICENSE](LICENSE), "Copyright (c) 2021 Daniel Hope (www.floorsense.nz)"; its
crypto (\`src/crypto/refc\`) is its own ChaCha20-Poly1305 (the same license),
BLAKE2s from RFC 7693, Poly1305 from poly1305-donna (public domain or MIT) and
X25519 from Mike Hamburg's STROBE (MIT, "Copyright (c) 2015-2016 Cryptography
Research, Inc.", \`src/crypto/refc/x25519-license.txt\`). NOTICE.md at the top
of this repo lists it.

WireGuard for the title's tunnel (\`ps5/wg\`), as an lwIP network interface.
\`scripts/vendor-wireguard.sh\` copies it, with one change of ours: up to
$max_peers peers, not 1 (\`src/wireguard-platform.h\`, marked "ts5"), so each
device has its own key. Its UDP goes through the PS5's own socket:
\`ps5/wg/wg_udp.c\` stands in for the lwIP UDP calls \`wireguardif.c\` makes.

Kept: \`src/\` (the protocol, the interface, the crypto glue) and
\`src/crypto/refc\`. Left out: the Cortex-M assembly, the examples.
EOF
echo "vendored lwIP $lwip_tag and wireguard-lwip ${wg_commit:0:7}"
