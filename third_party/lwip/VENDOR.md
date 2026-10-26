# lwIP, vendored

[lwIP](https://github.com/lwip-tcpip/lwip) 2.2.1 (tag `STABLE-2_2_1_RELEASE`, commit `77dcd25a72509eb83f72b033d219b1d40cd8eb95`).
BSD-3-Clause: [COPYING](COPYING), "Copyright (c) 2001, 2002 Swedish Institute of
Computer Science". NOTICE.md at the top of this repo lists it.

It is the TCP/IP stack inside the title's WireGuard tunnel (`ps5/wg`): the
PS5 gives a title no network interface of its own to make, so the tunnel's
packets are taken apart here, in the title. `ps5/wg/lwipopts.h` and
`ps5/wg/arch/cc.h` configure it (no OS: one thread drives it). Unchanged;
`scripts/vendor-wireguard.sh` copies it.

Kept: `src/core` and `src/core/ipv4` (IPv4, TCP, the buffers), the headers
in `src/include/lwip` but its `apps`, and those in `src/include/netif`, which the
core includes whatever the options (none of their code is kept). Left out: IPv6, the Ethernet, PPP and
6LoWPAN interfaces, the apps (HTTP, MQTT, SNMP ...), the ports, tests and docs.
