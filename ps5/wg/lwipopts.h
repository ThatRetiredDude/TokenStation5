// lwIP's settings for the WireGuard tunnel (ps5/wg/wg.cpp): one thread runs
// the whole stack (no OS layer, no locks), IPv4 and TCP only, memory from
// malloc. The tunnel carries API requests and their streamed replies, so the
// windows are sized for a phone on a mobile network rather than a LAN.
#pragma once

#define NO_SYS 1
#define SYS_LIGHTWEIGHT_PROT 0
#define LWIP_SOCKET 0
#define LWIP_NETCONN 0

#define LWIP_IPV4 1
#define LWIP_IPV6 0
#define LWIP_ARP 0
#define LWIP_ETHERNET 0
#define LWIP_ICMP 1  // ping 10.55.0.1 through the tunnel
#define LWIP_RAW 0
#define LWIP_UDP 1   // wireguardif.c's calls; ps5/wg/wg_udp.c stands in for them
#define LWIP_TCP 1
#define LWIP_DNS 0
#define LWIP_IGMP 0
#define IP_FORWARD 0
// WireGuard's MTU is 1420 and phones send to it: no fragments either way.
#define IP_REASSEMBLY 0
#define IP_FRAG 0

#define MEM_LIBC_MALLOC 1
#define MEMP_MEM_MALLOC 1
#define MEM_ALIGNMENT 8

#define TCP_MSS 1380  // 1420 less the IPv4 and TCP headers
#define LWIP_WND_SCALE 1
#define TCP_RCV_SCALE 2
#define TCP_WND (128 * 1024)
#define TCP_SND_BUF (43 * TCP_MSS)  // under 64 KiB, as lwIP's send counters need
#define TCP_SND_QUEUELEN (4 * TCP_SND_BUF / TCP_MSS)
#define MEMP_NUM_TCP_PCB 32
#define MEMP_NUM_TCP_PCB_LISTEN 2
#define LWIP_TCP_KEEPALIVE 1
#define TCP_LISTEN_BACKLOG 1
#define TCP_DEFAULT_LISTEN_BACKLOG 16

#define LWIP_NETIF_LOOPBACK 0
#define LWIP_HAVE_LOOPIF 0
#define LWIP_STATS 0
#define LWIP_NETIF_API 0

#ifdef __cplusplus
extern "C" {
#endif
unsigned int ts5_wg_rand(void);
#ifdef __cplusplus
}
#endif
#define LWIP_RAND() ts5_wg_rand()
#define LWIP_DONT_PROVIDE_BYTEORDER_FUNCTIONS 1  // the C library has htons and the rest
