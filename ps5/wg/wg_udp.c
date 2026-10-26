// lwIP's UDP, as wireguardif.c uses it, on the PS5's own socket. lwIP's
// udp.c isn't built: WireGuard's encrypted packets come and go through a
// socket the title opens (ps5/wg/wg.cpp), and nothing inside the tunnel
// speaks UDP. So this file has the calls wireguardif.c makes and the ones
// lwIP's core makes, which drop what they're given.
//
// Two WireGuard devices make calls here, each with its pcb: the PS5's own
// server (TS5_UDP_WIREGUARD), on the socket, and its device on a tailnet
// (TS5_UDP_TAILSCALE), whose packets go through Tailscale's relay instead
// (ps5/ts/tailscale.h): wg.cpp says which is being made before it adds the
// device, and passes on the second's packets with ts5_ts_send.
//
// Also the platform functions wireguard-lwip asks for, and lwIP's clock.
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>

#include "lwip/pbuf.h"
#include "lwip/sys.h"
#include "lwip/udp.h"
#include "wireguard-platform.h"
#include "wg_udp.h"

static struct udp_pcb device_pcbs[2];
static int making = TS5_UDP_WIREGUARD;
static int os_socket = -1;
unsigned long ts5_wg_sent, ts5_wg_send_failed;
int ts5_wg_send_errno;
size_t ts5_wg_sent_bytes;

void ts5_wg_udp_use(int fd) { os_socket = fd; }

void ts5_udp_making(int which) { making = which; }

void ts5_udp_input(int which, const void *data, size_t len, uint32_t ip_be, uint16_t port) {
    struct udp_pcb *pcb = &device_pcbs[which];
    if (!pcb->recv || len == 0 || len > 0xFFFF) return;
    struct pbuf *p = pbuf_alloc(PBUF_RAW, (u16_t)len, PBUF_RAM);
    if (!p) return;
    memcpy(p->payload, data, len);
    ip_addr_t from;
    ip_addr_set_ip4_u32_val(from, ip_be);
    // wireguardif_network_rx frees it.
    pcb->recv(pcb->recv_arg, pcb, p, &from, port);
}

void ts5_wg_udp_input(const void *data, size_t len, uint32_t ip_be, uint16_t port) {
    ts5_udp_input(TS5_UDP_WIREGUARD, data, len, ip_be, port);
}

struct udp_pcb *udp_new(void) {
    memset(&device_pcbs[making], 0, sizeof device_pcbs[making]);
    return &device_pcbs[making];
}

err_t udp_bind(struct udp_pcb *pcb, const ip_addr_t *ipaddr, u16_t port) {
    (void)ipaddr;
    pcb->local_port = port;  // wg.cpp bound the socket already
    return ERR_OK;
}

void udp_bind_netif(struct udp_pcb *pcb, const struct netif *netif) {
    (void)pcb;
    (void)netif;
}

void udp_recv(struct udp_pcb *pcb, udp_recv_fn recv, void *recv_arg) {
    pcb->recv = recv;
    pcb->recv_arg = recv_arg;
}

void udp_remove(struct udp_pcb *pcb) {
    pcb->recv = NULL;
    pcb->recv_arg = NULL;
}

err_t udp_sendto(struct udp_pcb *pcb, struct pbuf *p, const ip_addr_t *dst_ip, u16_t dst_port) {
    unsigned char buf[2048];
    if (p->tot_len > sizeof buf) return ERR_MEM;
    if (pcb == &device_pcbs[TS5_UDP_TAILSCALE]) {
        const u16_t n = pbuf_copy_partial(p, buf, p->tot_len, 0);
        return ts5_ts_send(buf, n, ip4_addr_get_u32(ip_2_ip4(dst_ip)), dst_port) ? ERR_OK : ERR_RTE;
    }
    if (os_socket < 0) return ERR_CONN;
    const u16_t n = pbuf_copy_partial(p, buf, p->tot_len, 0);
    struct sockaddr_in to;
    memset(&to, 0, sizeof to);
    to.sin_family = AF_INET;
    to.sin_port = htons(dst_port);
    to.sin_addr.s_addr = ip4_addr_get_u32(ip_2_ip4(dst_ip));
    if (sendto(os_socket, buf, n, 0, (const struct sockaddr *)&to, sizeof to) == (ssize_t)n) {
        ts5_wg_sent++;
        ts5_wg_sent_bytes = n;
        return ERR_OK;
    }
    ts5_wg_send_errno = errno;
    // The first few in the log; then the count, when the title asks.
    if (++ts5_wg_send_failed <= 5) ts5_wg_diag("can't send %u bytes to port %u (errno %d)", n, dst_port, errno);
    return ERR_IF;
}

// lwIP's core: no UDP inside the tunnel.
void udp_init(void) {}
void udp_input(struct pbuf *p, struct netif *inp) {
    (void)inp;
    pbuf_free(p);
}
void udp_netif_ip_addr_changed(const ip_addr_t *old_addr, const ip_addr_t *new_addr) {
    (void)old_addr;
    (void)new_addr;
}

static uint64_t monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

u32_t sys_now(void) { return (u32_t)monotonic_ms(); }

uint32_t wireguard_sys_now(void) { return (uint32_t)monotonic_ms(); }

void wireguard_random_bytes(void *bytes, size_t size) { ts5_wg_random(bytes, size); }

// TAI64N: seconds since 1970 plus 2^62 and the 10 leap seconds TAI was ahead
// then, big-endian, and nanoseconds. Peers only check it grows.
void wireguard_tai64n_now(uint8_t *output) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const uint64_t s = 0x400000000000000aULL + (uint64_t)ts.tv_sec;
    const uint32_t ns = (uint32_t)ts.tv_nsec;
    for (int i = 0; i < 8; i++) output[i] = (uint8_t)(s >> (56 - 8 * i));
    for (int i = 0; i < 4; i++) output[8 + i] = (uint8_t)(ns >> (24 - 8 * i));
}

bool wireguard_is_under_load(void) { return false; }
