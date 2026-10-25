// Between ps5/wg/wg.cpp and wg_udp.c, lwIP's UDP on the PS5's socket.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// The socket WireGuard's packets go out of.
void ts5_wg_udp_use(int fd);
// A packet that came in on it, from ip_be (network order) and port.
void ts5_wg_udp_input(const void *data, size_t len, uint32_t ip_be, uint16_t port);
// Defined in wg.cpp: random bytes from the system.
void ts5_wg_random(void *out, size_t n);
#ifdef __cplusplus
}
#endif
