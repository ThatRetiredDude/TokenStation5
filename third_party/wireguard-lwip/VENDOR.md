# wireguard-lwip, vendored

[wireguard-lwip](https://github.com/smartalock/wireguard-lwip), commit `c54f20dbe76ac8b3411ad21e0ed7deea6f0cfd4d`. BSD-3-Clause:
[LICENSE](LICENSE), "Copyright (c) 2021 Daniel Hope (www.floorsense.nz)"; its
crypto (`src/crypto/refc`) is its own ChaCha20-Poly1305 (the same license),
BLAKE2s from RFC 7693, Poly1305 from poly1305-donna (public domain or MIT) and
X25519 from Mike Hamburg's STROBE (MIT, "Copyright (c) 2015-2016 Cryptography
Research, Inc.", `src/crypto/refc/x25519-license.txt`). NOTICE.md at the top
of this repo lists it.

WireGuard for the title's tunnel (`ps5/wg`), as an lwIP network interface.
`scripts/vendor-wireguard.sh` copies it, with one change of ours: up to
16 peers, not 1 (`src/wireguard-platform.h`, marked "ts5"), so each
device has its own key. Its UDP goes through the PS5's own socket:
`ps5/wg/wg_udp.c` stands in for the lwIP UDP calls `wireguardif.c` makes.

Kept: `src/` (the protocol, the interface, the crypto glue) and
`src/crypto/refc`. Left out: the Cortex-M assembly, the examples.
