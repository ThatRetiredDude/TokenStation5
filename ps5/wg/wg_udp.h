// Between ps5/wg/wg.cpp and wg_udp.c, lwIP's UDP on the PS5's socket.
#pragma once
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// The socket WireGuard's packets go out of.
void ts5_wg_udp_use(int fd);
// A packet that came in on it, from ip_be (network order) and port.
void ts5_wg_udp_input(const void *data, size_t len, uint32_t ip_be, uint16_t port);

// The two WireGuard devices: the PS5's own server, and the PS5 on a tailnet.
#define TS5_UDP_WIREGUARD 0
#define TS5_UDP_TAILSCALE 1
// Which device the next udp_new is for (wireguardif_init makes one).
void ts5_udp_making(int which);
// A packet for one of them.
void ts5_udp_input(int which, const void *data, size_t len, uint32_t ip_be, uint16_t port);
// Defined in wg.cpp: the tailnet device's packet to the peer that has the
// stand-in address ip_be:port, through Tailscale's relay. False if no peer has it.
bool ts5_ts_send(const void *data, size_t len, uint32_t ip_be, uint16_t port);
// Defined in wg.cpp: random bytes from the system.
void ts5_wg_random(void *out, size_t n);
// Packets sent, sends that failed and the last one's errno, and the size of
// the last packet sent: for the log.
extern unsigned long ts5_wg_sent, ts5_wg_send_failed;
extern int ts5_wg_send_errno;
extern size_t ts5_wg_sent_bytes;
#ifdef __cplusplus
}
#endif
