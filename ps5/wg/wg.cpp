#include "wg.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <list>

#include "log.h"
#include "net.h"

// The system's TCP_MSS (netinet/tcp.h) isn't the tunnel's (lwipopts.h).
#undef TCP_MSS
extern "C" {
#include "lwip/init.h"
#include "lwip/netif.h"
#include "lwip/tcp.h"
#include "lwip/timeouts.h"
#include "crypto.h"
#include "wireguard.h"
#include "wireguardif.h"
#include "wg_udp.h"
}

extern "C" void ts5_wg_random(void *out, size_t n) {
    if (!ts5::random_bytes((unsigned char *)out, n)) {
        // No random numbers means no safe keys: stop rather than go on.
        ts5::log_line("ts5-title: wireguard: the system gave no random numbers");
        abort();
    }
}

extern "C" unsigned int ts5_wg_rand(void) {
    unsigned int r;
    ts5_wg_random(&r, sizeof r);
    return r;
}

extern "C" void ts5_wg_diag(const char *fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
    ts5::log_line("ts5-title: wireguard: lwip: %s", line);
}

extern "C" void ts5_wg_assert(const char *what, const char *file, int line) {
    ts5::log_line("ts5-title: wireguard: lwip assertion failed: %s (%s:%d)", what, file, line);
    abort();
}

namespace wg {

static_assert(kMaxPeers == WIREGUARD_MAX_PEERS, "wg.h's kMaxPeers is wireguard-lwip's peer limit");

namespace {

std::string b64(const uint8_t *key) {
    char out[64];
    size_t n = sizeof out;
    if (!wireguard_base64_encode(key, WIREGUARD_PUBLIC_KEY_LEN, out, &n)) return "";
    return std::string(out, strnlen(out, sizeof out));
}

bool unb64(const std::string &s, uint8_t *key) {
    size_t n = WIREGUARD_PUBLIC_KEY_LEN;
    return !s.empty() && wireguard_base64_decode(s.c_str(), key, &n) && n == WIREGUARD_PUBLIC_KEY_LEN;
}

bool parse_ip(const std::string &s, uint32_t &be) {
    in_addr a{};
    if (inet_pton(AF_INET, s.c_str(), &a) != 1) return false;
    be = a.s_addr;
    return true;
}

std::string ip_text(uint32_t be) {
    char buf[INET_ADDRSTRLEN] = {};
    in_addr a{};
    a.s_addr = be;
    inet_ntop(AF_INET, &a, buf, sizeof buf);
    return buf;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One connection from a device to the API: lwIP's side in the tunnel, and
// a socket to the API on this PS5.
struct Conn {
    tcp_pcb *pcb = nullptr;  // null once lwIP is done with it
    int fd = -1;
    uint16_t local_port = 0;
    bool connecting = true;
    bool tunnel_eof = false;  // the device sent everything
    bool api_eof = false;     // the API sent everything
    bool shut_wr = false;
    std::string to_api;       // from the device, not written to the API yet
};

}  // namespace

bool make_keys(std::string &private_key, std::string &public_key) {
    uint8_t priv[WIREGUARD_PRIVATE_KEY_LEN], pub[WIREGUARD_PUBLIC_KEY_LEN];
    if (!ts5::random_bytes(priv, sizeof priv)) return false;
    priv[0] &= 248;
    priv[31] = (uint8_t)((priv[31] & 127) | 64);
    x25519_base(pub, priv, 1);
    private_key = b64(priv);
    public_key = b64(pub);
    crypto_zero(priv, sizeof priv);
    return !private_key.empty() && !public_key.empty();
}

std::string public_key_of(const std::string &private_key) {
    uint8_t priv[WIREGUARD_PRIVATE_KEY_LEN], pub[WIREGUARD_PUBLIC_KEY_LEN];
    if (!unb64(private_key, priv)) return "";
    x25519_base(pub, priv, 1);
    crypto_zero(priv, sizeof priv);
    return b64(pub);
}

Tunnel::~Tunnel() {
    quit_ = true;
    if (thread_.joinable()) thread_.join();
}

bool Tunnel::apply(const Config &c, std::string &why) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_) {
        if (!c.on) {
            want_ = c;
            return true;
        }
        uint8_t key[WIREGUARD_PRIVATE_KEY_LEN];
        uint32_t ip;
        if (!unb64(c.private_key, key)) {
            why = "no private key";
            return false;
        }
        crypto_zero(key, sizeof key);
        if (!parse_ip(c.address, ip)) {
            why = "the tunnel address " + c.address + " isn't an IPv4 address";
            return false;
        }
        want_ = c;
        changed_ = true;
        started_ = true;
        thread_ = std::thread([this] { loop(); });
        return true;
    }
    if (c.address != want_.address)
        ts5::log_line("ts5-title: wireguard: a new tunnel address takes effect at the next start");
    if (public_key_of(c.private_key).empty()) {
        why = "no private key";
        return false;
    }
    const std::string address = want_.address;
    want_ = c;
    want_.address = address;
    changed_ = true;
    return true;
}

std::string Tunnel::problem() {
    std::lock_guard<std::mutex> lk(mu_);
    return problem_;
}

std::vector<PeerState> Tunnel::peers() {
    std::lock_guard<std::mutex> lk(mu_);
    return states_;
}

std::string Tunnel::client_for(uint16_t local_port) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = clients_.find(local_port);
    return it == clients_.end() ? std::string() : it->second;
}

namespace {

// Everything the thread keeps. lwIP's callbacks find it through their arg.
struct Loop {
    Tunnel *owner = nullptr;
    std::mutex *mu = nullptr;
    std::map<uint16_t, std::string> *clients = nullptr;
    netif nif{};
    wireguardif_init_data init{};
    std::string private_key;
    tcp_pcb *listener = nullptr;
    int udp = -1, udp_port = 0;
    unsigned long received = 0, traced = 0;  // packets in since the port opened; how many the detailed log showed
    bool poll_failed = false;
    std::map<std::string, bool> up;  // by public key: had a session at the last look, for the log
    int api_port = 8081;
    std::map<std::string, uint8_t> peers;  // public key: wireguardif's index
    std::map<std::string, Peer> peer_cfg;
    std::list<Conn *> conns;
};

void drop_client(Loop &L, Conn *c) {
    if (!c->local_port) return;
    std::lock_guard<std::mutex> lk(*L.mu);
    L.clients->erase(c->local_port);
    c->local_port = 0;
}

void forget_pcb(Conn *c) {
    if (!c->pcb) return;
    tcp_arg(c->pcb, nullptr);
    tcp_recv(c->pcb, nullptr);
    tcp_sent(c->pcb, nullptr);
    tcp_err(c->pcb, nullptr);
    c->pcb = nullptr;
}

err_t on_recv(void *arg, tcp_pcb *pcb, pbuf *p, err_t err) {
    Conn *c = (Conn *)arg;
    (void)pcb;
    if (!c) {
        if (p) pbuf_free(p);
        return ERR_OK;
    }
    if (!p || err != ERR_OK) {
        if (p) pbuf_free(p);
        c->tunnel_eof = true;
        return ERR_OK;
    }
    const size_t at = c->to_api.size();
    c->to_api.resize(at + p->tot_len);
    pbuf_copy_partial(p, &c->to_api[at], p->tot_len, 0);
    pbuf_free(p);
    // tcp_recved when the API has taken it, so the window holds the rest back.
    return ERR_OK;
}

void on_err(void *arg, err_t err) {
    Conn *c = (Conn *)arg;
    (void)err;
    if (c) c->pcb = nullptr;  // lwIP freed it
}

// Non-blocking, both ways: on the PS5, O_NONBLOCK through fcntl didn't take
// on a socket (2026-10-25: a recv on the API's socket held the tunnel's
// thread for good), so FIONBIO too, and every call passes MSG_DONTWAIT.
void nonblocking(int fd) {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
    int one = 1;
    ioctl(fd, FIONBIO, &one);
    fcntl(fd, F_SETFD, FD_CLOEXEC);
}

bool detail() {
    const char *on = getenv("TS5_NET_LOG");
    return on && strcmp(on, "1") == 0;
}

err_t on_accept(void *arg, tcp_pcb *pcb, err_t err) {
    Loop &L = *(Loop *)arg;
    if (err != ERR_OK || !pcb) return ERR_VAL;
    const std::string from = ip_text(ip4_addr_get_u32(ip_2_ip4(&pcb->remote_ip)));
    // A socket from 127.0.0.1 to the API, its port noted first, so the API
    // knows the request came through the tunnel the moment it accepts.
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof a;
    if (fd < 0 || bind(fd, (sockaddr *)&a, sizeof a) != 0 || getsockname(fd, (sockaddr *)&a, &len) != 0) {
        ts5::log_line("ts5-title: wireguard: no socket for %s's connection (errno %d)", from.c_str(), errno);
        if (fd >= 0) close(fd);
        tcp_abort(pcb);
        return ERR_ABRT;
    }
    nonblocking(fd);
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    Conn *c = new Conn;
    c->pcb = pcb;
    c->fd = fd;
    c->local_port = ntohs(a.sin_port);
    {
        std::lock_guard<std::mutex> lk(*L.mu);
        (*L.clients)[c->local_port] = from;
    }
    sockaddr_in api{};
    api.sin_family = AF_INET;
    api.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    api.sin_port = htons((uint16_t)L.api_port);
    if (connect(fd, (sockaddr *)&api, sizeof api) == 0) c->connecting = false;
    else if (errno != EINPROGRESS) {
        ts5::log_line("ts5-title: wireguard: can't reach the API for %s (errno %d)", from.c_str(), errno);
        drop_client(L, c);
        close(fd);
        delete c;
        tcp_abort(pcb);
        return ERR_ABRT;
    }
    tcp_arg(pcb, c);
    tcp_recv(pcb, on_recv);
    tcp_err(pcb, on_err);
    tcp_nagle_disable(pcb);  // streamed tokens go out as they come
    L.conns.push_back(c);
    if (detail())
        ts5::log_line("ts5-title: wireguard: tcp: %s connects, API socket from port %u%s", from.c_str(), c->local_port,
                      c->connecting ? ", connecting" : ", connected");
    return ERR_OK;
}

void close_udp(Loop &L) {
    if (L.udp < 0) return;
    ts5_wg_udp_use(-1);
    close(L.udp);
    L.udp = -1;
    L.udp_port = 0;
}

// Opens the UDP socket on port; "" or why not.
std::string open_udp(Loop &L, int port) {
    close_udp(L);
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return "no UDP socket (errno " + std::to_string(errno) + ")";
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(fd, (sockaddr *)&a, sizeof a) != 0) {
        const int e = errno;
        close(fd);
        return e == EADDRINUSE ? "port " + std::to_string(port) + " is in use"
               : e == EACCES   ? "port " + std::to_string(port) + " is refused by the PS5"
                               : "can't use port " + std::to_string(port) + " (errno " + std::to_string(e) + ")";
    }
    nonblocking(fd);
    L.udp = fd;
    L.udp_port = port;
    L.received = L.traced = 0;
    ts5_wg_udp_use(fd);
    return "";
}

void sync_peers(Loop &L, const std::vector<Peer> &want) {
    std::map<std::string, Peer> next;
    for (const Peer &p : want) next[p.public_key] = p;
    // Out: peers no longer wanted, or changed.
    for (auto it = L.peers.begin(); it != L.peers.end();) {
        auto n = next.find(it->first);
        const Peer &old = L.peer_cfg[it->first];
        if (n == next.end() || n->second.address != old.address || n->second.endpoint != old.endpoint ||
            n->second.keepalive != old.keepalive) {
            wireguardif_remove_peer(&L.nif, it->second);
            L.peer_cfg.erase(it->first);
            it = L.peers.erase(it);
        } else {
            L.peer_cfg[it->first].name = n->second.name;
            ++it;
        }
    }
    // In: the new ones.
    for (const auto &[key, p] : next) {
        if (L.peers.count(key)) continue;
        uint32_t ip;
        if (!parse_ip(p.address, ip)) {
            ts5::log_line("ts5-title: wireguard: %s's address %s isn't an IPv4 address", p.name.c_str(), p.address.c_str());
            continue;
        }
        wireguardif_peer peer;
        wireguardif_peer_init(&peer);
        peer.public_key = key.c_str();
        peer.preshared_key = nullptr;
        ip_addr_set_ip4_u32_val(peer.allowed_ip, ip);
        ip_addr_set_ip4_u32_val(peer.allowed_mask, 0xFFFFFFFFu);
        peer.keep_alive = (u16_t)p.keepalive;
        uint32_t ep = 0;
        int ep_port = 0;
        const size_t colon = p.endpoint.rfind(':');
        if (colon != std::string::npos && parse_ip(p.endpoint.substr(0, colon), ep)) {
            ep_port = atoi(p.endpoint.c_str() + colon + 1);
            ip_addr_set_ip4_u32_val(peer.endpoint_ip, ep);
            peer.endport_port = (u16_t)ep_port;
        }
        uint8_t index = WIREGUARDIF_INVALID_INDEX;
        if (wireguardif_add_peer(&L.nif, &peer, &index) != ERR_OK || index == WIREGUARDIF_INVALID_INDEX) {
            ts5::log_line("ts5-title: wireguard: can't add %s (bad key, or more than %d devices)", p.name.c_str(),
                          WIREGUARD_MAX_PEERS);
            continue;
        }
        if (ep_port > 0) wireguardif_connect(&L.nif, index);
        L.peers[key] = index;
        L.peer_cfg[key] = p;
    }
}

void drop_conns(Loop &L) {
    for (Conn *c : L.conns) {
        if (c->pcb) {
            tcp_pcb *pcb = c->pcb;
            forget_pcb(c);
            tcp_abort(pcb);
        }
        drop_client(L, c);
        close(c->fd);
        delete c;
    }
    L.conns.clear();
}

// Moves what it can between the device and the API; false once it's over.
bool pump(Loop &L, Conn *c, short revents) {
    // Whatever poll said (on the PS5 it didn't say a socket was ready): a
    // connection with a peer has connected; one with an error has failed.
    (void)revents;
    if (c->connecting) {
        int e = 0;
        socklen_t len = sizeof e;
        getsockopt(c->fd, SOL_SOCKET, SO_ERROR, &e, &len);
        sockaddr_in peer{};
        socklen_t plen = sizeof peer;
        if (e != 0) {
            ts5::log_line("ts5-title: wireguard: can't reach the API (errno %d)", e);
            c->api_eof = true;
            c->to_api.clear();
            c->connecting = false;
        } else if (getpeername(c->fd, (sockaddr *)&peer, &plen) == 0) {
            c->connecting = false;
            if (detail()) ts5::log_line("ts5-title: wireguard: tcp: API socket from port %u connected", c->local_port);
        }
    }
    if (c->connecting) return true;
    // Device to API.
    if (!c->to_api.empty() && !c->api_eof) {
        const ssize_t n = ::send(c->fd, c->to_api.data(), c->to_api.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n > 0) {
            c->to_api.erase(0, (size_t)n);
            // The window opens again by what the API took (tcp_recved takes 16 bits at a time).
            for (size_t left = (size_t)n; c->pcb && left > 0;) {
                const u16_t k = (u16_t)(left < 0xFFFF ? left : 0xFFFF);
                tcp_recved(c->pcb, k);
                left -= k;
            }
        } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            c->api_eof = true;
            c->to_api.clear();
        }
    }
    if (c->tunnel_eof && c->to_api.empty() && !c->shut_wr) {
        shutdown(c->fd, SHUT_WR);
        c->shut_wr = true;
    }
    // API to device, as much as lwIP's send buffer takes.
    if (c->pcb && !c->api_eof) {
        size_t room = tcp_sndbuf(c->pcb);
        while (room > 0) {
            char buf[16384];
            const ssize_t n = ::recv(c->fd, buf, room < sizeof buf ? room : sizeof buf, MSG_DONTWAIT);
            if (n > 0) {
                if (tcp_write(c->pcb, buf, (u16_t)n, TCP_WRITE_FLAG_COPY) != ERR_OK) break;  // lwIP is out of room; retry later
                room -= (size_t)n;
                continue;
            }
            if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) c->api_eof = true;
            break;
        }
        tcp_output(c->pcb);
    }
    if (c->api_eof && c->pcb) {
        // Its queued data still goes out before the FIN; lwIP frees it after.
        tcp_pcb *pcb = c->pcb;
        forget_pcb(c);
        if (tcp_close(pcb) != ERR_OK) tcp_abort(pcb);
    }
    if (!c->pcb) {
        drop_client(L, c);
        close(c->fd);
        return false;
    }
    return true;
}

}  // namespace

void Tunnel::loop() {
    Loop L;
    L.owner = this;
    L.mu = &mu_;
    L.clients = &clients_;
    Config cfg;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cfg = want_;
    }
    lwip_init();
    uint32_t ip = 0;
    parse_ip(cfg.address, ip);  // apply() checked it
    ip4_addr_t addr, mask, gw;
    ip4_addr_set_u32(&addr, ip);
    ip4_addr_set_u32(&mask, htonl(0xFFFFFF00u));
    ip4_addr_set_u32(&gw, 0);
    L.private_key = cfg.private_key;
    L.init.private_key = L.private_key.c_str();
    L.init.listen_port = (u16_t)cfg.port;
    L.init.bind_netif = nullptr;
    if (!netif_add(&L.nif, &addr, &mask, &gw, &L.init, wireguardif_init, ip_input)) {
        std::lock_guard<std::mutex> lk(mu_);
        problem_ = "WireGuard couldn't start";
        ts5::log_line("ts5-title: wireguard: the interface didn't start (a bad private key?)");
        return;
    }
    netif_set_default(&L.nif);
    netif_set_up(&L.nif);
    netif_set_link_up(&L.nif);
    tcp_pcb *pcb = tcp_new();
    if (!pcb || tcp_bind(pcb, IP_ADDR_ANY, (u16_t)cfg.api_port) != ERR_OK || !(L.listener = tcp_listen(pcb))) {
        std::lock_guard<std::mutex> lk(mu_);
        problem_ = "WireGuard couldn't start";
        ts5::log_line("ts5-title: wireguard: no listener on port %d in the tunnel", cfg.api_port);
        return;
    }
    tcp_arg(L.listener, &L);
    tcp_accept(L.listener, on_accept);
    L.api_port = cfg.api_port;
    ts5::log_line("ts5-title: wireguard: the PS5 is %s in the tunnel, key %s", cfg.address.c_str(),
                  public_key_of(cfg.private_key).c_str());

    double last_states = 0, pass_at = 0;
    std::vector<pollfd> fds;
    while (!quit_) {
        bool changed;
        {
            std::lock_guard<std::mutex> lk(mu_);
            changed = changed_;
            changed_ = false;
            if (changed) cfg = want_;
        }
        if (changed) {
            std::string why;
            if (!cfg.on) {
                if (L.udp >= 0) ts5::log_line("ts5-title: wireguard: off");
                close_udp(L);
                drop_conns(L);
            } else if (L.udp < 0 || L.udp_port != cfg.port) {
                why = open_udp(L, cfg.port);
                if (why.empty()) ts5::log_line("ts5-title: wireguard: on, UDP port %d", cfg.port);
                else ts5::log_line("ts5-title: wireguard: %s", why.c_str());
            }
            if (cfg.private_key != L.private_key) {
                // A new key pair: every session ends, and each peer's keys are
                // worked out again from the new one.
                uint8_t key[WIREGUARD_PRIVATE_KEY_LEN];
                if (unb64(cfg.private_key, key)) {
                    drop_conns(L);
                    sync_peers(L, {});
                    wireguard_device_init((wireguard_device *)L.nif.state, key);
                    crypto_zero(key, sizeof key);
                    L.private_key = cfg.private_key;
                    ts5::log_line("ts5-title: wireguard: new key pair, public key %s",
                                  public_key_of(cfg.private_key).c_str());
                }
            }
            sync_peers(L, cfg.peers);
            std::lock_guard<std::mutex> lk(mu_);
            problem_ = why;
        }

        fds.clear();
        if (L.udp >= 0) fds.push_back({L.udp, POLLIN, 0});
        for (Conn *c : L.conns) {
            short ev = 0;
            if (c->connecting || (!c->to_api.empty() && !c->api_eof)) ev |= POLLOUT;
            if (c->pcb && !c->api_eof && !c->connecting && tcp_sndbuf(c->pcb) > 0) ev |= POLLIN;
            fds.push_back({c->fd, ev, 0});
        }
        u32_t sleep_ms = sys_timeouts_sleeptime();
        if (sleep_ms > 50) sleep_ms = 50;  // apply() and lwIP's send buffer are checked this often
        // With connections open, more often: on the PS5 poll doesn't wake for
        // a socket that's ready, so this is how soon their data moves on.
        if (!L.conns.empty() && sleep_ms > 5) sleep_ms = 5;
        if (fds.empty()) usleep(sleep_ms * 1000);
        else if (poll(fds.data(), (nfds_t)fds.size(), (int)sleep_ms) < 0) {
            // Then every socket is tried: they're all non-blocking.
            if (errno != EINTR && !L.poll_failed) {
                ts5::log_line("ts5-title: wireguard: poll failed (errno %d); trying every socket each time", errno);
                L.poll_failed = true;
            }
            for (pollfd &f : fds) f.revents = f.events;
            usleep(sleep_ms * 1000);
        }

        size_t i = 0;
        if (L.udp >= 0) {
            // Read whatever came, whatever poll said.
            unsigned char buf[2048];
            for (int k = 0; k < 256; k++) {
                sockaddr_in from{};
                socklen_t len = sizeof from;
                const ssize_t n = recvfrom(L.udp, buf, sizeof buf, MSG_DONTWAIT, (sockaddr *)&from, &len);
                if (n <= 0) break;
                const unsigned long sent = ts5_wg_sent, failed = ts5_wg_send_failed;
                ts5_wg_udp_input(buf, (size_t)n, from.sin_addr.s_addr, ntohs(from.sin_port));
                char ip[INET_ADDRSTRLEN] = "?";
                inet_ntop(AF_INET, &from.sin_addr, ip, sizeof ip);
                if (++L.received == 1)
                    ts5::log_line("ts5-title: wireguard: the first packet came in, from %s:%u", ip, ntohs(from.sin_port));
                // Detailed log: each packet (the first 200 since the port
                // opened), its WireGuard type (1 a handshake, 4 data) and
                // whether the PS5 answered it.
                if (detail() && L.traced < 200) {
                    L.traced++;
                    const std::string what = ts5_wg_sent > sent ? "answered, " + std::to_string(ts5_wg_sent_bytes) + " bytes"
                                             : ts5_wg_send_failed > failed
                                                 ? "the answer couldn't be sent (errno " + std::to_string(ts5_wg_send_errno) + ")"
                                                 : "no answer";
                    ts5::log_line("ts5-title: wireguard: udp: %zd bytes from %s:%u, type %u: %s", n, ip,
                                  ntohs(from.sin_port), (unsigned)buf[0], what.c_str());
                }
            }
            i = 1;
        }
        for (auto it = L.conns.begin(); it != L.conns.end(); i++) {
            Conn *c = *it;
            const short rev = i < fds.size() ? fds[i].revents : 0;
            if (pump(L, c, rev)) {
                ++it;
            } else {
                delete c;
                it = L.conns.erase(it);
            }
        }
        sys_check_timeouts();

        const double t = now_s();
        // A pass that took long: something held the thread.
        if (pass_at > 0 && t - pass_at > 1)
            ts5::log_line("ts5-title: wireguard: one pass of the tunnel took %.1f s", t - pass_at);
        pass_at = t;
        if (t - last_states >= 0.5) {
            last_states = t;
            std::vector<PeerState> states;
            const wireguard_device *dev = (const wireguard_device *)L.nif.state;
            const uint32_t now_ms = wireguard_sys_now();
            for (const auto &[key, index] : L.peers) {
                const Peer &p = L.peer_cfg[key];
                PeerState s;
                s.name = p.name;
                s.address = p.address;
                ip_addr_t from;
                u16_t port = 0;
                s.up = wireguardif_peer_is_up(&L.nif, index, &from, &port) == ERR_OK;
                if (port && !ip_addr_isany(&from)) s.from = ip_text(ip4_addr_get_u32(ip_2_ip4(&from))) + ":" + std::to_string(port);
                if (dev && index < WIREGUARD_MAX_PEERS && dev->peers[index].last_rx)
                    s.seen_s = (double)(uint32_t)(now_ms - dev->peers[index].last_rx) / 1000.0;
                bool &was = L.up[key];
                if (s.up != was)
                    ts5::log_line(s.up ? "ts5-title: wireguard: %s connected, from %s" : "ts5-title: wireguard: %s's session ended%s",
                                  p.name.c_str(), s.up ? s.from.c_str() : "");
                was = s.up;
                states.push_back(s);
            }
            std::lock_guard<std::mutex> lk(mu_);
            states_ = std::move(states);
        }
    }
    drop_conns(L);
    close_udp(L);
}

}  // namespace wg
