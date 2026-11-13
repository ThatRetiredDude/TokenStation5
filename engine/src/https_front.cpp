#include "https_front.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <mbedtls/asn1.h>
#include <mbedtls/error.h>
#include <mbedtls/pk.h>
#include <mbedtls/ssl.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>

#include "log.h"
#include "net.h"

namespace ts5 {

namespace {

// At most this many connections at once; more are closed at once.
constexpr int kMaxConns = 64;
// A handshake that takes longer is dropped.
constexpr int kHandshakeMs = 20000;

int rng(void *, unsigned char *out, size_t n) { return random_bytes(out, n) ? 0 : MBEDTLS_ERR_SSL_INTERNAL_ERROR; }

const char *kAlpn[] = {"http/1.1", nullptr};

std::string tls_error(int rc) {
    char b[160];
    mbedtls_strerror(rc, b, sizeof b);
    char h[24];
    snprintf(h, sizeof h, " (-0x%04x)", (unsigned)-rc);
    return std::string(b) + h;
}

// A certificate, its key and the TLS settings made from them; each
// connection keeps the one it began with.
struct Creds {
    mbedtls_x509_crt chain;
    mbedtls_pk_context key;
    mbedtls_ssl_config conf;
    std::vector<std::string> names;  // the certificate's DNS names, in lower case
    Creds() {
        mbedtls_x509_crt_init(&chain);
        mbedtls_pk_init(&key);
        mbedtls_ssl_config_init(&conf);
    }
    ~Creds() {
        mbedtls_ssl_config_free(&conf);
        mbedtls_pk_free(&key);
        mbedtls_x509_crt_free(&chain);
    }
    Creds(const Creds &) = delete;
    Creds &operator=(const Creds &) = delete;
};

std::string lower(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

// Only a connection that asks for one of the certificate's names gets it. One
// that asks for none (https://100.x.y.z:PORT/, an address: Safari's try at
// HTTPS for an http:// address) or for another name is ended before the
// certificate goes out, as when nothing answered HTTPS there, so Safari
// goes on with http:// instead of warning about a certificate.
int check_name(mbedtls_ssl_context *ssl) {
    size_t len = 0;
    const unsigned char *sni = mbedtls_ssl_get_hs_sni(ssl, &len);
    const auto *cr = (const Creds *)mbedtls_ssl_get_user_data_p(ssl);
    if (!sni || !len || !cr) return MBEDTLS_ERR_SSL_UNRECOGNIZED_NAME;
    const std::string want = lower(std::string((const char *)sni, len));
    for (const auto &n : cr->names)
        if (n == want) return 0;
    return MBEDTLS_ERR_SSL_UNRECOGNIZED_NAME;
}

int fd_send(void *ctx, const unsigned char *buf, size_t len) {
    const ssize_t n = ::send(*(int *)ctx, buf, len, MSG_NOSIGNAL);
    if (n >= 0) return (int)n;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return MBEDTLS_ERR_SSL_WANT_WRITE;
    return -0x004E;  // MBEDTLS_ERR_NET_SEND_FAILED
}

int fd_recv(void *ctx, unsigned char *buf, size_t len) {
    const ssize_t n = recv(*(int *)ctx, buf, len, 0);
    if (n >= 0) return (int)n;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return MBEDTLS_ERR_SSL_WANT_READ;
    if (errno == ECONNRESET) return -0x0050;  // MBEDTLS_ERR_NET_CONN_RESET
    return -0x004C;                           // MBEDTLS_ERR_NET_RECV_FAILED
}

// Waits for fd to be ready for what a WANT_READ/WANT_WRITE asks; false on a
// timeout or an error.
bool wait_for(int fd, int want, int ms) {
    pollfd p{fd, (short)(want == MBEDTLS_ERR_SSL_WANT_WRITE ? POLLOUT : POLLIN), 0};
    while (true) {
        const int r = poll(&p, 1, ms);
        if (r > 0) return true;
        if (r == 0 || errno != EINTR) return false;
    }
}

bool tls_write_all(mbedtls_ssl_context &ssl, int fd, const char *p, size_t n) {
    while (n) {
        const int rc = mbedtls_ssl_write(&ssl, (const unsigned char *)p, n);
        if (rc > 0) {
            p += rc;
            n -= (size_t)rc;
        } else if (rc == MBEDTLS_ERR_SSL_WANT_WRITE || rc == MBEDTLS_ERR_SSL_WANT_READ) {
            if (!wait_for(fd, rc, 60000)) return false;
        } else {
            return false;
        }
    }
    return true;
}

bool send_all(int fd, const unsigned char *p, size_t n) {
    while (n) {
        const ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        p += w;
        n -= (size_t)w;
    }
    return true;
}

// Handshake problems are common (a scanner, a browser giving up): one line a
// minute at most.
void note_handshake(const std::string &device, int rc) {
    static std::mutex mu;
    static double last = -1e9;
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> lk(mu);
        if (now - last < 60) return;
        last = now;
    }
    log_line("ts5: https: a connection from %s ended before its TLS was done: %s",
             device.empty() ? "this machine" : device.c_str(),
             rc == MBEDTLS_ERR_SSL_UNRECOGNIZED_NAME ? "it didn't ask for the certificate's name (an address?)"
             : rc                                    ? tls_error(rc).c_str()
                                                     : "timed out");
}

// One connection: TLS, then both ways to the API until either side ends.
void serve(std::shared_ptr<Creds> cr, int fd, std::string device, int api_port, HttpsFront::Note note,
           std::atomic<int> *active) {
    mbedtls_ssl_context ssl;
    mbedtls_ssl_init(&ssl);
    int api = -1;
    int rc = mbedtls_ssl_setup(&ssl, &cr->conf);
    mbedtls_ssl_set_bio(&ssl, &fd, fd_send, fd_recv, nullptr);
    mbedtls_ssl_set_user_data_p(&ssl, cr.get());  // for check_name
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(kHandshakeMs);
    while (rc == 0 && (rc = mbedtls_ssl_handshake(&ssl)) != 0) {
        if (rc != MBEDTLS_ERR_SSL_WANT_READ && rc != MBEDTLS_ERR_SSL_WANT_WRITE) break;
        const int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(until - std::chrono::steady_clock::now()).count();
        if (left <= 0 || !wait_for(fd, rc, left)) {
            rc = 1;
            break;
        }
        rc = 0;
    }
    if (rc != 0) {
        note_handshake(device, rc == 1 ? 0 : rc);
    } else {
        // The API, from a port of our own that the tunnel names first.
        api = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t len = sizeof a;
        bool ok = api >= 0 && bind(api, (sockaddr *)&a, sizeof a) == 0 && getsockname(api, (sockaddr *)&a, &len) == 0;
        if (ok && !device.empty() && note) note(ntohs(a.sin_port), device);
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        to.sin_port = htons((uint16_t)api_port);
        ok = ok && connect(api, (sockaddr *)&to, sizeof to) == 0;
        if (!ok) {
            log_line("ts5: https: can't reach the API on port %d (errno %d)", api_port, errno);
        } else {
            const int one = 1;
            setsockopt(api, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            bool client_open = true;
            unsigned char buf[16384];
            while (true) {
                const bool pending = client_open && mbedtls_ssl_get_bytes_avail(&ssl) > 0;
                // (A device done sending is left out: poll would report its hang-up each time.)
                pollfd p[2] = {{client_open ? fd : -1, POLLIN, 0}, {api, POLLIN, 0}};
                const int r = poll(p, 2, pending ? 0 : -1);
                if (r < 0 && errno != EINTR) break;
                bool end = false;
                if (client_open && (pending || (p[0].revents & (POLLIN | POLLHUP | POLLERR)))) {
                    // Everything TLS has whole now; WANT_READ: the rest of a record is on its way.
                    while (true) {
                        const int n = mbedtls_ssl_read(&ssl, buf, sizeof buf);
                        if (n > 0) {
                            if (!send_all(api, buf, (size_t)n)) {
                                end = true;
                                break;
                            }
                            continue;
                        }
                        if (n == MBEDTLS_ERR_SSL_WANT_READ || n == MBEDTLS_ERR_SSL_WANT_WRITE) break;
                        // The device is done sending (close_notify, or gone): the API
                        // still answers what it has.
                        client_open = false;
                        shutdown(api, SHUT_WR);
                        if (n != 0 && n != MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) end = true;
                        break;
                    }
                }
                if (end) break;
                if (p[1].revents & (POLLIN | POLLHUP | POLLERR)) {
                    const ssize_t n = recv(api, buf, sizeof buf, 0);
                    if (n < 0 && errno == EINTR) continue;
                    if (n <= 0) {
                        mbedtls_ssl_close_notify(&ssl);
                        break;
                    }
                    if (!tls_write_all(ssl, fd, (const char *)buf, (size_t)n)) break;
                }
            }
        }
    }
    if (api >= 0) close(api);
    mbedtls_ssl_free(&ssl);
    close(fd);
    cr.reset();
    active->fetch_sub(1);
}

}  // namespace

struct HttpsFront::Impl {
    int lfd = -1, wake[2] = {-1, -1};
    int api_port = 0;
    NameOf name_of;
    Note note;
    std::thread thread;
    std::atomic<bool> quit{false};
    std::atomic<int> active{0};
    std::mutex mu;
    std::shared_ptr<Creds> creds;

    void accept_loop() {
        while (!quit) {
            pollfd p[2] = {{lfd, POLLIN, 0}, {wake[0], POLLIN, 0}};
            if (poll(p, 2, -1) < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (p[1].revents) break;
            if (!(p[0].revents & POLLIN)) continue;
            sockaddr_in peer{};
            socklen_t len = sizeof peer;
            const int fd = accept4(lfd, (sockaddr *)&peer, &len, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (fd < 0) continue;
            // The name first, whatever happens: the tunnel's entry for it goes.
            const std::string device = name_of ? name_of(ntohs(peer.sin_port)) : std::string();
            std::shared_ptr<Creds> cr;
            {
                std::lock_guard<std::mutex> lk(mu);
                cr = creds;
            }
            if (!cr || active.load() >= kMaxConns) {
                close(fd);
                continue;
            }
            const int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            active.fetch_add(1);
            std::thread(serve, cr, fd, device, api_port, note, &active).detach();
        }
    }
};

HttpsFront::HttpsFront() : d_(new Impl) {}

HttpsFront::~HttpsFront() {
    if (d_->thread.joinable()) {
        d_->quit = true;
        if (write(d_->wake[1], "x", 1) < 0) {
        }
        d_->thread.join();
    }
    if (d_->lfd >= 0) close(d_->lfd);
    for (int fd : d_->wake)
        if (fd >= 0) close(fd);
    // Connections still open keep their own settings (shared) and end on
    // their own; the server's process ends with them.
}

int HttpsFront::start(int api_port, NameOf name_of, Note note, std::string &why, int port) {
    if (psa_crypto_init() != PSA_SUCCESS) {
        why = "can't start the crypto library";
        return 0;
    }
    d_->api_port = api_port;
    d_->name_of = std::move(name_of);
    d_->note = std::move(note);
    d_->lfd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    const int one = 1;
    if (d_->lfd >= 0) setsockopt(d_->lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons((uint16_t)port);
    socklen_t len = sizeof a;
    if (d_->lfd < 0 || bind(d_->lfd, (sockaddr *)&a, sizeof a) != 0 || listen(d_->lfd, 64) != 0 ||
        getsockname(d_->lfd, (sockaddr *)&a, &len) != 0 || pipe2(d_->wake, O_CLOEXEC) != 0) {
        why = "can't listen on 127.0.0.1:" + std::to_string(port) + " (" + strerror(errno) + ")";
        return 0;
    }
    d_->thread = std::thread([this] { d_->accept_loop(); });
    return ntohs(a.sin_port);
}

bool HttpsFront::set_certificate(const std::string &cert_pem, const std::string &key_pem, std::string &why) {
    auto cr = std::make_shared<Creds>();
    int rc = mbedtls_x509_crt_parse(&cr->chain, (const unsigned char *)cert_pem.c_str(), cert_pem.size() + 1);
    if (rc != 0) {
        why = "the certificate can't be read: " + tls_error(rc);
        return false;
    }
    rc = mbedtls_pk_parse_key(&cr->key, (const unsigned char *)key_pem.c_str(), key_pem.size() + 1, nullptr, 0, rng,
                              nullptr);
    if (rc != 0) {
        why = "the certificate's key can't be read: " + tls_error(rc);
        return false;
    }
    for (const mbedtls_x509_sequence *n = &cr->chain.subject_alt_names; n && n->buf.p; n = n->next)
        if (n->buf.tag == (MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_DNS_NAME))
            cr->names.push_back(lower(std::string((const char *)n->buf.p, n->buf.len)));
    if (cr->names.empty()) {
        why = "the certificate names no DNS name";
        return false;
    }
    rc = mbedtls_ssl_config_defaults(&cr->conf, MBEDTLS_SSL_IS_SERVER, MBEDTLS_SSL_TRANSPORT_STREAM,
                                     MBEDTLS_SSL_PRESET_DEFAULT);
    if (rc == 0) {
        mbedtls_ssl_conf_rng(&cr->conf, rng, nullptr);
        mbedtls_ssl_conf_min_tls_version(&cr->conf, MBEDTLS_SSL_VERSION_TLS1_2);
        rc = mbedtls_ssl_conf_alpn_protocols(&cr->conf, kAlpn);
    }
    if (rc == 0) rc = mbedtls_ssl_conf_own_cert(&cr->conf, &cr->chain, &cr->key);
    mbedtls_ssl_conf_cert_cb(&cr->conf, check_name);
    if (rc != 0) {
        why = "TLS setup failed: " + tls_error(rc);
        return false;
    }
    std::lock_guard<std::mutex> lk(d_->mu);
    d_->creds = cr;
    return true;
}

bool HttpsFront::adopt(int fd, const std::string &device) {
    std::shared_ptr<Creds> cr;
    {
        std::lock_guard<std::mutex> lk(d_->mu);
        cr = d_->creds;
    }
    if (!cr || !d_->api_port || d_->active.load() >= kMaxConns) return false;
    const int fl = fcntl(fd, F_GETFL);
    if (fl < 0 || fcntl(fd, F_SETFL, fl | O_NONBLOCK) != 0) return false;
    d_->active.fetch_add(1);
    std::thread(serve, cr, fd, device, d_->api_port, d_->note, &d_->active).detach();
    return true;
}

bool HttpsFront::has_certificate() {
    std::lock_guard<std::mutex> lk(d_->mu);
    return d_->creds != nullptr;
}

}  // namespace ts5
