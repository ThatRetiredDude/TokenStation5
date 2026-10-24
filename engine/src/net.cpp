#include "net.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>

#ifdef TS5_HTTPS
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/entropy.h>
#include <mbedtls/error.h>
#include <mbedtls/platform_util.h>
#include <mbedtls/sha256.h>
#include <mbedtls/ssl.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>
#endif

#ifdef TS5_PS5
// The console's resolver (libSceNet): a title has no /etc/resolv.conf, and its
// libc exports no getaddrinfo. Random numbers come from the kernel's
// kern.arandom, through libkernel's sysctl, which needs no other module (the
// title died on its first HTTPS request while it used libSceRandom).
extern "C" {
int sceNetInit(void);
int sceNetPoolCreate(const char *name, int size, int flags);
int sceNetResolverCreate(const char *name, int pool, int flags);
int sceNetResolverStartNtoa(int resolver, const char *host, uint32_t *addr, int timeout_us, int retries, int flags);
int sceNetResolverDestroy(int resolver);
int sysctl(const int *name, unsigned namelen, void *old, size_t *oldlen, const void *new_, size_t newlen);
}
#else
#include <fcntl.h>
#endif

namespace ts5 {

namespace {

// One line per step of a request in the log, so a request that never comes
// back shows how far it got.
void trace(const char *fmt, ...) {
    // Seconds since the first line, so the gaps between lines show (a
    // redirect hop on the LAN took 9-17 s with every timed step in ms).
    static const auto t0 = std::chrono::steady_clock::now();
    const double at = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // One write for the line, so lines from several threads don't mix.
    char line[1024];
    int n = snprintf(line, sizeof line, "ts5: net: [%.3f s] ", at);
    va_list ap;
    va_start(ap, fmt);
    if (n > 0 && n < (int)sizeof line) vsnprintf(line + n, sizeof line - (size_t)n, fmt, ap);
    va_end(ap);
    std::string out = line;
    out += '\n';
    fwrite(out.data(), 1, out.size(), stderr);
    fflush(stderr);
}

bool fill_random(unsigned char *out, size_t n) {
#ifdef TS5_PS5
    static const int name[2] = {1, 37};  // CTL_KERN, KERN_ARND
    static bool first = true;
    if (first) trace("random numbers from kern.arandom");
    while (n > 0) {
        size_t k = n < 256 ? n : 256;  // the most one call gives
        if (sysctl(name, 2, out, &k, nullptr, 0) != 0 || k == 0) {
            trace("kern.arandom failed (errno %d)", errno);
            return false;
        }
        out += k;
        n -= k;
    }
    if (first) trace("random numbers ok");
    first = false;
    return true;
#else
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    while (n > 0) {
        ssize_t k = read(fd, out, n);
        if (k <= 0) {
            if (k < 0 && errno == EINTR) continue;
            close(fd);
            return false;
        }
        out += k;
        n -= (size_t)k;
    }
    close(fd);
    return true;
#endif
}

// IPv4 addresses for host (network byte order).
bool resolve(const std::string &host, std::vector<uint32_t> &addrs, std::string &err) {
    in_addr a{};
    if (inet_pton(AF_INET, host.c_str(), &a) == 1) {
        addrs.push_back(a.s_addr);
        return true;
    }
#ifdef TS5_PS5
    static std::mutex mu;
    static int pool = -1;
    std::lock_guard<std::mutex> lk(mu);
    if (pool < 0) {
        sceNetInit();  // already done by the system for most titles; harmless again
        pool = sceNetPoolCreate("ts5-net", 64 * 1024, 0);
        if (pool < 0) {
            err = "sceNetPoolCreate failed (" + std::to_string(pool) + ")";
            return false;
        }
    }
    int r = sceNetResolverCreate("ts5-dns", pool, 0);
    if (r < 0) {
        err = "sceNetResolverCreate failed (" + std::to_string(r) + ")";
        return false;
    }
    uint32_t addr = 0;
    int rc = sceNetResolverStartNtoa(r, host.c_str(), &addr, 0, 0, 0);  // the system's timeout and retries
    sceNetResolverDestroy(r);
    if (rc < 0 || addr == 0) {
        char b[96];
        snprintf(b, sizeof b, "cannot resolve %s (0x%08x)", host.c_str(), (unsigned)rc);
        err = b;
        return false;
    }
    addrs.push_back(addr);
    return true;
#else
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host.c_str(), nullptr, &hints, &res);
    if (rc != 0) {
        err = "cannot resolve " + host + ": " + gai_strerror(rc);
        return false;
    }
    for (addrinfo *p = res; p; p = p->ai_next) addrs.push_back(((sockaddr_in *)p->ai_addr)->sin_addr.s_addr);
    freeaddrinfo(res);
    return !addrs.empty();
#endif
}

// Milliseconds since t, for the log's timings.
long ms_since(std::chrono::steady_clock::time_point t) {
    return (long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
}

int connect_to(const std::string &host, int port, int timeout_s, std::string &err) {
    std::vector<uint32_t> addrs;
    trace("resolving %s", host.c_str());
    auto t = std::chrono::steady_clock::now();
    if (!resolve(host, addrs, err)) return -1;
    char ip[INET_ADDRSTRLEN] = "?";
    inet_ntop(AF_INET, &addrs[0], ip, sizeof ip);
    trace("%s is %s (%ld ms); connecting to port %d", host.c_str(), ip, ms_since(t), port);
    t = std::chrono::steady_clock::now();
    for (uint32_t addr : addrs) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            err = std::string("socket: ") + strerror(errno);
            return -1;
        }
        timeval tv{timeout_s, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons((uint16_t)port);
        sa.sin_addr.s_addr = addr;
        if (connect(fd, (sockaddr *)&sa, sizeof sa) == 0) {
            trace("connected (%ld ms)", ms_since(t));
            return fd;
        }
        char ip[INET_ADDRSTRLEN] = "?";
        inet_ntop(AF_INET, &sa.sin_addr, ip, sizeof ip);
        err = "cannot connect to " + host + " (" + ip + ":" + std::to_string(port) + "): " + strerror(errno);
        close(fd);
    }
    return -1;
}

// A connection: plain, or TLS on top.
struct Conn {
    int fd = -1;
#ifdef TS5_HTTPS
    bool tls = false;
    mbedtls_ssl_context ssl;
#endif
    ~Conn() {
#ifdef TS5_HTTPS
        if (tls) mbedtls_ssl_free(&ssl);
#endif
        if (fd >= 0) close(fd);
    }
    bool write_all(const std::string &s, std::string &err) {
        size_t off = 0;
        while (off < s.size()) {
#ifdef TS5_HTTPS
            if (tls) {
                int n = mbedtls_ssl_write(&ssl, (const unsigned char *)s.data() + off, s.size() - off);
                if (n == MBEDTLS_ERR_SSL_WANT_WRITE || n == MBEDTLS_ERR_SSL_WANT_READ) continue;
                if (n <= 0) {
                    err = "TLS write failed (-0x" + hex(-n) + ")";
                    return false;
                }
                off += (size_t)n;
                continue;
            }
#endif
            ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                err = std::string("send: ") + strerror(errno);
                return false;
            }
            off += (size_t)n;
        }
        return true;
    }
    // Everything until the server closes.
    bool read_all(std::string &out, std::string &err) {
        char buf[16384];
        while (true) {
            if (out.size() > (16u << 20)) {
                err = "response larger than 16 MiB";
                return false;
            }
#ifdef TS5_HTTPS
            if (tls) {
                int n = mbedtls_ssl_read(&ssl, (unsigned char *)buf, sizeof buf);
                if (n > 0) { out.append(buf, (size_t)n); continue; }
                if (n == 0 || n == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) return true;
                if (n == MBEDTLS_ERR_SSL_WANT_READ || n == MBEDTLS_ERR_SSL_WANT_WRITE ||
                    n == MBEDTLS_ERR_SSL_RECEIVED_NEW_SESSION_TICKET)
                    continue;
                // Some servers just close the socket after the response.
                if (n == MBEDTLS_ERR_NET_CONN_RESET_ && !out.empty()) return true;
                err = "TLS read failed (-0x" + hex(-n) + ")";
                return false;
            }
#endif
            ssize_t n = recv(fd, buf, sizeof buf, 0);
            if (n > 0) { out.append(buf, (size_t)n); continue; }
            if (n == 0) return true;
            if (errno == EINTR) continue;
            err = errno == EAGAIN || errno == EWOULDBLOCK ? std::string("timed out waiting for the server")
                                                         : std::string("recv: ") + strerror(errno);
            return false;
        }
    }
    // What has arrived, up to cap bytes: how many, 0 once the server has
    // closed, -1 on an error (err says). For a body too big to hold.
    int read_some(char *buf, size_t cap, std::string &err) {
        while (true) {
#ifdef TS5_HTTPS
            if (tls) {
                int n = mbedtls_ssl_read(&ssl, (unsigned char *)buf, cap);
                if (n > 0) return n;
                if (n == 0 || n == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY || n == MBEDTLS_ERR_NET_CONN_RESET_) return 0;
                if (n == MBEDTLS_ERR_SSL_WANT_READ || n == MBEDTLS_ERR_SSL_WANT_WRITE ||
                    n == MBEDTLS_ERR_SSL_RECEIVED_NEW_SESSION_TICKET)
                    continue;
                err = n == -0x004C ? std::string("timed out waiting for the server")  // MBEDTLS_ERR_NET_RECV_FAILED
                                   : "TLS read failed (-0x" + hex(-n) + ")";
                return -1;
            }
#endif
            ssize_t n = recv(fd, buf, cap, 0);
            if (n >= 0) return (int)n;
            if (errno == EINTR) continue;
            err = errno == EAGAIN || errno == EWOULDBLOCK ? std::string("timed out waiting for the server")
                                                         : std::string("recv: ") + strerror(errno);
            return -1;
        }
    }
    static std::string hex(int v) {
        char b[16];
        snprintf(b, sizeof b, "%04x", v);
        return b;
    }
#ifdef TS5_HTTPS
    static constexpr int MBEDTLS_ERR_NET_CONN_RESET_ = -0x0050;  // mbedtls/net_sockets.h
#endif
};

#ifdef TS5_HTTPS

extern "C" const char ts5_ca_pem[];  // ca_bundle.cpp

struct Tls {
    bool ready = false;
    std::string why;  // when not ready
    mbedtls_entropy_context entropy;
    mbedtls_ctr_drbg_context drbg;
    mbedtls_x509_crt ca;
    mbedtls_ssl_config conf;
    std::string ca_from;
};

// The shared generator, one caller at a time: mbedtls is built without its
// threading layer, and downloads, web searches and the Hugging Face search
// make connections from several threads at once.
int locked_random(void *drbg, unsigned char *out, size_t len) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    return mbedtls_ctr_drbg_random(drbg, out, len);
}

// The first HTTPS request sets up what every one shares.
Tls &tls() {
    static Tls t;
    static std::once_flag once;
    std::call_once(once, [] {
        mbedtls_entropy_init(&t.entropy);
        mbedtls_ctr_drbg_init(&t.drbg);
        mbedtls_x509_crt_init(&t.ca);
        mbedtls_ssl_config_init(&t.conf);
        trace("first HTTPS request: starting crypto");
        if (psa_crypto_init() != PSA_SUCCESS) { t.why = "psa_crypto_init failed"; return; }
        trace("crypto started; seeding the TLS random generator");
        static const char pers[] = "ts5";
        int rc = mbedtls_ctr_drbg_seed(&t.drbg, mbedtls_entropy_func, &t.entropy, (const unsigned char *)pers,
                                       sizeof pers);
        if (rc != 0) { t.why = "no random numbers for TLS (-0x" + Conn::hex(-rc) + ")"; return; }
        trace("TLS random generator seeded; reading the CA bundle");
        // TS5_CA_FILE, else the system's bundle, else the one built in.
        std::vector<std::string> files;
        if (const char *f = getenv("TS5_CA_FILE")) files.push_back(f);
#ifndef TS5_PS5
        files.insert(files.end(), {"/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                                   "/etc/ssl/cert.pem"});
#endif
        for (auto &f : files) {
            if (access(f.c_str(), R_OK) != 0) continue;
            if (mbedtls_x509_crt_parse_file(&t.ca, f.c_str()) >= 0 && t.ca.version != 0) {
                t.ca_from = f;
                break;
            }
            mbedtls_x509_crt_free(&t.ca);
            mbedtls_x509_crt_init(&t.ca);
        }
        if (t.ca_from.empty()) {
            // The bundle has a few certificates Mbed TLS cannot read; it skips
            // them and keeps the rest (a positive return counts them).
            rc = mbedtls_x509_crt_parse(&t.ca, (const unsigned char *)ts5_ca_pem, strlen(ts5_ca_pem) + 1);
            if (rc < 0 || t.ca.version == 0) { t.why = "cannot read the built-in CA bundle"; return; }
            t.ca_from = "built in";
        }
        trace("CA bundle read (%s)", t.ca_from.c_str());
        rc = mbedtls_ssl_config_defaults(&t.conf, MBEDTLS_SSL_IS_CLIENT, MBEDTLS_SSL_TRANSPORT_STREAM,
                                         MBEDTLS_SSL_PRESET_DEFAULT);
        if (rc != 0) { t.why = "TLS setup failed (-0x" + Conn::hex(-rc) + ")"; return; }
        mbedtls_ssl_conf_authmode(&t.conf, MBEDTLS_SSL_VERIFY_REQUIRED);
        mbedtls_ssl_conf_ca_chain(&t.conf, &t.ca, nullptr);
        mbedtls_ssl_conf_rng(&t.conf, locked_random, &t.drbg);
        t.ready = true;
    });
    return t;
}

int bio_send(void *ctx, const unsigned char *buf, size_t len) {
    int fd = *(int *)ctx;
    ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
    if (n >= 0) return (int)n;
    if (errno == EINTR) return MBEDTLS_ERR_SSL_WANT_WRITE;
    return -0x004E;  // MBEDTLS_ERR_NET_SEND_FAILED
}

int bio_recv(void *ctx, unsigned char *buf, size_t len) {
    int fd = *(int *)ctx;
    ssize_t n = recv(fd, buf, len, 0);
    if (n >= 0) return (int)n;
    if (errno == EINTR) return MBEDTLS_ERR_SSL_WANT_READ;
    if (errno == ECONNRESET) return -0x0050;  // MBEDTLS_ERR_NET_CONN_RESET
    return -0x004C;                           // MBEDTLS_ERR_NET_RECV_FAILED (and the timeout)
}

bool start_tls(Conn &c, const std::string &host, std::string &err) {
    trace("connected; setting up TLS");
    Tls &t = tls();
    if (!t.ready) {
        err = t.why;
        return false;
    }
    mbedtls_ssl_init(&c.ssl);
    c.tls = true;
    int rc = mbedtls_ssl_setup(&c.ssl, &t.conf);
    if (rc == 0) rc = mbedtls_ssl_set_hostname(&c.ssl, host.c_str());
    if (rc != 0) {
        err = "TLS setup failed (-0x" + Conn::hex(-rc) + ")";
        return false;
    }
    mbedtls_ssl_set_bio(&c.ssl, &c.fd, bio_send, bio_recv, nullptr);
    while ((rc = mbedtls_ssl_handshake(&c.ssl)) != 0) {
        if (rc == MBEDTLS_ERR_SSL_WANT_READ || rc == MBEDTLS_ERR_SSL_WANT_WRITE) continue;
        uint32_t flags = mbedtls_ssl_get_verify_result(&c.ssl);
        if (flags != 0 && flags != (uint32_t)-1) {
            char why[256];
            mbedtls_x509_crt_verify_info(why, sizeof why, "", flags);
            err = "the certificate of " + host + " was not accepted (CA bundle: " + t.ca_from + "): " + why;
            while (!err.empty() && err.back() == '\n') err.pop_back();
        } else {
            char why[128];
            mbedtls_strerror(rc, why, sizeof why);
            err = "TLS handshake with " + host + " failed: " + why;
        }
        return false;
    }
    trace("TLS handshake done (%s, %s)", mbedtls_ssl_get_version(&c.ssl), mbedtls_ssl_get_ciphersuite(&c.ssl));
    return true;
}

#endif  // TS5_HTTPS

// The body of a response sent with Transfer-Encoding: chunked.
bool dechunk(const std::string &in, std::string &out) {
    size_t i = 0;
    while (i < in.size()) {
        size_t e = in.find("\r\n", i);
        if (e == std::string::npos) return false;
        size_t n = strtoul(in.substr(i, e - i).c_str(), nullptr, 16);
        if (n == 0) return true;
        if (e + 2 + n > in.size()) return false;
        out.append(in, e + 2, n);
        i = e + 2 + n + 2;
    }
    return true;
}

// ---- Downloads (http_get_file) ----

// The SHA-256 of what has been written so far. Its state goes out and comes
// back as bytes, so a download can go on from a checkpoint (Sha256Resume).
#ifdef TS5_HTTPS
struct Hasher {
    mbedtls_sha256_context ctx;
    Hasher() { mbedtls_sha256_init(&ctx); }
    ~Hasher() { mbedtls_sha256_free(&ctx); }
    static bool available() { return true; }
    void start() { mbedtls_sha256_starts(&ctx, 0); }
    void update(const char *p, size_t n) { mbedtls_sha256_update(&ctx, (const unsigned char *)p, n); }
    // The digest in lowercase hex. Ends the hash.
    std::string finish() {
        unsigned char d[32];
        mbedtls_sha256_finish(&ctx, d);
        char hex[65];
        for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", d[i]);
        return hex;
    }
    std::vector<unsigned char> state() const {
        const unsigned char *p = (const unsigned char *)&ctx;
        return std::vector<unsigned char>(p, p + sizeof ctx);
    }
    bool restore(const std::vector<unsigned char> &s) {
        if (s.size() != sizeof ctx) return false;
        memcpy(&ctx, s.data(), sizeof ctx);
        return true;
    }
};
#else
// Without Mbed TLS there is no SHA-256, so a download can't be checked.
struct Hasher {
    static bool available() { return false; }
    void start() {}
    void update(const char *, size_t) {}
    std::string finish() { return ""; }
    std::vector<unsigned char> state() const { return {}; }
    bool restore(const std::vector<unsigned char> &) { return false; }
};
#endif

// The parts of http[s]://host[:port]/path?query. `path` keeps the query for
// the request line; `shown` is the host and the path without it, for the
// log, because a signed CDN URL's query acts as a credential.
struct Url {
    bool https = false;
    std::string host, hostport, path, shown;
    int port = 80;
};

bool parse_url(const std::string &url, Url &u, std::string &err) {
    size_t s = url.find("://");
    if (s == std::string::npos) {
        err = "not a URL";
        return false;
    }
    const std::string scheme = url.substr(0, s);
    u.https = scheme == "https";
    if (!u.https && scheme != "http") {
        err = "unsupported URL scheme " + scheme;
        return false;
    }
    size_t p = url.find('/', s + 3);
    u.hostport = url.substr(s + 3, p == std::string::npos ? std::string::npos : p - s - 3);
    u.path = p == std::string::npos ? "/" : url.substr(p);
    u.port = u.https ? 443 : 80;
    u.host = u.hostport;
    size_t colon = u.hostport.rfind(':');
    if (colon != std::string::npos) {
        u.host = u.hostport.substr(0, colon);
        u.port = atoi(u.hostport.c_str() + colon + 1);
    }
    if (u.host.empty()) {
        err = "a URL without a host";
        return false;
    }
    u.shown = u.host + " " + u.path.substr(0, u.path.find('?'));
    return true;
}

// Where a Location header points: a whole URL, or a path on the same server.
std::string resolve(const Url &from, const std::string &location) {
    if (location.compare(0, 7, "http://") == 0 || location.compare(0, 8, "https://") == 0) return location;
    const std::string scheme = from.https ? "https:" : "http:";
    if (location.compare(0, 2, "//") == 0) return scheme + location;
    if (!location.empty() && location[0] == '/') return scheme + "//" + from.hostport + location;
    const std::string dir = from.path.substr(0, from.path.find('?'));
    return scheme + "//" + from.hostport + dir.substr(0, dir.rfind('/') + 1) + location;
}

// A header's value from a response head, whatever the case of its name; ""
// without it.
std::string header(const std::string &head, const char *name) {
    std::string lower = head, key = std::string("\r\n") + name + ":";
    for (auto &ch : lower) ch = (char)tolower((unsigned char)ch);
    for (auto &ch : key) ch = (char)tolower((unsigned char)ch);
    size_t p = lower.find(key);
    if (p == std::string::npos) return "";
    p += key.size();
    size_t e = head.find("\r\n", p);
    std::string v = head.substr(p, e == std::string::npos ? std::string::npos : e - p);
    size_t a = v.find_first_not_of(" \t"), b = v.find_last_not_of(" \t");
    return a == std::string::npos ? "" : v.substr(a, b - a + 1);
}

// A response body as it comes: counted by Content-Length, chunked, or
// everything until the server closes. (dechunk() above is the whole-string
// form a POST's answer uses.)
class Body {
public:
    enum Kind { Counted, Chunked, ToClose };
    // ahead: what came after the headers in the same read.
    Body(Conn &c, std::string ahead, Kind kind, uint64_t length)
        : c_(c), stash_(std::move(ahead)), kind_(kind), left_(length) {}
    // The next piece into buf: how many bytes, 0 at the end, -1 on an error (err says).
    int next(char *buf, size_t cap, std::string &err);
    // False when a counted body ended short.
    bool complete() const { return kind_ != Counted || left_ == 0; }

private:
    // More into the stash. False at the end, or on an error (err says).
    bool more(std::string &err) {
        stash_.erase(0, pos_);
        pos_ = 0;
        char buf[65536];
        int n = c_.read_some(buf, sizeof buf, err);
        if (n <= 0) return false;
        stash_.append(buf, (size_t)n);
        return true;
    }
    Conn &c_;
    std::string stash_;  // read, not yet taken
    size_t pos_ = 0;     // taken up to here
    Kind kind_;
    uint64_t left_;      // Counted: bytes still to come; Chunked: left in this chunk
    enum { Size, Data, AfterData, Trailers, End } chunk_ = Size;
};

int Body::next(char *buf, size_t cap, std::string &err) {
    if (kind_ != Chunked) {
        if (kind_ == Counted && left_ == 0) return 0;
        if (kind_ == Counted && cap > left_) cap = (size_t)left_;
        int n;
        if (pos_ < stash_.size()) {
            n = (int)std::min(cap, stash_.size() - pos_);
            memcpy(buf, stash_.data() + pos_, (size_t)n);
            pos_ += (size_t)n;
        } else {
            n = c_.read_some(buf, cap, err);
            if (n <= 0) return n;  // 0: closed, maybe short (complete() says)
        }
        if (kind_ == Counted) left_ -= (uint64_t)n;
        return n;
    }
    for (;;) {
        const size_t have = stash_.size() - pos_;
        switch (chunk_) {
            case Size: {
                size_t e = stash_.find("\r\n", pos_);
                if (e == std::string::npos) {
                    if (have > 65536) {
                        err = "bad chunked response";
                        return -1;
                    }
                    if (!more(err)) {
                        if (err.empty()) err = "incomplete chunked response";
                        return -1;
                    }
                    continue;
                }
                left_ = strtoull(stash_.c_str() + pos_, nullptr, 16);
                pos_ = e + 2;
                chunk_ = left_ == 0 ? Trailers : Data;
                continue;
            }
            case Data: {
                if (have == 0) {
                    if (!more(err)) {
                        if (err.empty()) err = "incomplete chunked response";
                        return -1;
                    }
                    continue;
                }
                size_t n = (size_t)std::min<uint64_t>(std::min(have, cap), left_);
                memcpy(buf, stash_.data() + pos_, n);
                pos_ += n;
                left_ -= n;
                if (left_ == 0) chunk_ = AfterData;
                return (int)n;
            }
            case AfterData:  // the CRLF after a chunk's data
                if (have < 2) {
                    if (!more(err)) {
                        if (err.empty()) err = "incomplete chunked response";
                        return -1;
                    }
                    continue;
                }
                pos_ += 2;
                chunk_ = Size;
                continue;
            case Trailers: {  // after the last chunk: trailer lines, then a blank one
                size_t e = stash_.find("\r\n", pos_);
                if (e == std::string::npos) {
                    if (!more(err)) {  // closed here: nothing was missing
                        chunk_ = End;
                        return err.empty() ? 0 : -1;
                    }
                    continue;
                }
                if (e == pos_) chunk_ = End;
                pos_ = e + 2;
                continue;
            }
            case End: return 0;
        }
    }
}

// Beside a part, "<part>.from" says which URL it is of (without the query
// string: a signed one changes with every link) and what the server called
// that version of the file (its ETag, else its Last-Modified; maybe
// nothing), one a line. A resume only goes on from a part of the same URL,
// and asks for the rest with If-Range when there is a validator, so a file
// that changed on the server comes whole (200) instead of its new tail
// landing after the old head.
bool read_from(const std::string &path, std::string &url, std::string &tag) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string text;
    char buf[4096];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
    fclose(f);
    const size_t a = text.find('\n');
    if (a == std::string::npos) return false;
    const size_t b = text.find('\n', a + 1);
    url = text.substr(0, a);
    tag = text.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1);
    return !url.empty();
}

bool write_from(const std::string &path, const std::string &url, const std::string &tag) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fprintf(f, "%s\n%s\n", url.c_str(), tag.c_str());
    return fclose(f) == 0;
}

// The download itself, on the thread http_get_file gives it. The file is
// written from `start` (the checkpoint's offset, else `offset`), 256 KiB at
// a time, and the hash follows the writes; a checkpoint is reported only
// after a flush, so it never describes bytes that aren't on disk yet.
bool get_file(const std::string &url, const std::string &path, uint64_t offset, const std::string &expected_sha256,
              FetchProgress &progress, std::string &err, Sha256Resume *sha) {
    constexpr int timeout_s = 30;
    constexpr size_t block = 256 << 10;
    constexpr uint64_t checkpoint_every = 64ull << 20;
    using clock = std::chrono::steady_clock;
    auto seconds = [](clock::duration d) { return std::chrono::duration<double>(d).count(); };

    if (!expected_sha256.empty() && !Hasher::available()) {
        err = "this ts5 was built without HTTPS, so it can't check a SHA-256";
        return false;
    }
    std::string want = expected_sha256;
    for (auto &ch : want) ch = (char)tolower((unsigned char)ch);
    Url u;
    if (!parse_url(url, u, err)) return false;
#ifndef TS5_HTTPS
    if (u.https) {
        err = "this ts5 was built without HTTPS";
        return false;
    }
#endif

    // Where to go on from: the checkpoint's offset with its hash state, else
    // the file's first `offset` bytes, hashed here first. Only a part of this
    // same URL is gone on from (its .from file says which); another one, or
    // one without a .from, starts over.
    struct stat st;
    if (offset > 0 && (stat(path.c_str(), &st) != 0 || (uint64_t)st.st_size < offset)) {
        offset = stat(path.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
        trace("the file is shorter than said; going on from its %llu bytes", (unsigned long long)offset);
    }
    const std::string from_path = path + ".from", source = url.substr(0, url.find('?'));
    std::string from_url, validator;
    if (offset > 0 && !read_from(from_path, from_url, validator)) {
        trace("the part doesn't say where it's from; starting over");
        offset = 0;
    } else if (offset > 0 && from_url != source) {
        trace("the part is of another URL; starting over");
        offset = 0;
    }
    uint64_t start = offset;
    Hasher hash;
    hash.start();
    bool rehash = start > 0;
    if (sha && !sha->state.empty()) {
        if (sha->offset <= offset && hash.restore(sha->state)) {
            start = sha->offset;
            rehash = false;
        } else {
            trace("the checkpoint doesn't fit the file; hashing the file instead");
        }
    }
    // The buffers before the file, so they are still there when the Closer's
    // fclose flushes what is left in fbuf on a failure return.
    std::vector<char> fbuf(1 << 20), wbuf(block);
    FILE *f = fopen(path.c_str(), start > 0 ? "r+b" : "wb");
    if (!f) {
        err = std::string("cannot open the file: ") + strerror(errno);
        return false;
    }
    struct Closer {
        FILE *f;
        ~Closer() {
            if (f) fclose(f);
        }
    } closer{f};
    setvbuf(f, fbuf.data(), _IOFBF, fbuf.size());
    progress.done = start;
    // Waits while paused. False once cancelled.
    auto go_on = [&] {
        while (progress.pause && !progress.cancel) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!progress.cancel) return true;
        err = "cancelled";
        return false;
    };
    // A full disk is the one failure the user can do something about, so it
    // gets its own words.
    auto write_failed = [&] {
        err = errno == ENOSPC ? std::string("storage full") : std::string("cannot write the file: ") + strerror(errno);
        return false;
    };
    // A part that turned out wrong (another size or hash than it should
    // have, or longer than the file) goes, with its .from, so the next call
    // starts over instead of failing the same way again. A part a dropped
    // connection, a timeout, a full disk or a cancel cut short is kept.
    auto damaged = [&] {
        trace("removing the part: it can't be gone on from");
        fclose(f);
        closer.f = nullptr;
        unlink(path.c_str());
        unlink(from_path.c_str());
        return false;
    };
    if (rehash) {
        trace("hashing the %llu bytes already there", (unsigned long long)start);
        for (uint64_t left = start; left > 0;) {
            size_t n = fread(wbuf.data(), 1, (size_t)std::min<uint64_t>(block, left), f);
            if (n == 0) {
                err = "cannot read the file";
                return false;
            }
            hash.update(wbuf.data(), n);
            left -= n;
            if (!go_on()) return false;
        }
    }
    if (start > 0 && (fseeko(f, (off_t)start, SEEK_SET) != 0 || ftruncate(fileno(f), (off_t)start) != 0)) {
        err = std::string("cannot truncate the file: ") + strerror(errno);
        return false;
    }

    uint64_t done = start, total = 0;
    for (int hop = 0;; hop++) {
        if (start > 0) trace("GET %s from byte %llu", u.shown.c_str(), (unsigned long long)start);
        else trace("GET %s", u.shown.c_str());
        Conn c;
        c.fd = connect_to(u.host, u.port, timeout_s, err);
        if (c.fd < 0) return false;
#ifdef TS5_HTTPS
        if (u.https && !start_tls(c, u.host, err)) return false;
#endif
        std::string req = "GET " + u.path + " HTTP/1.1\r\nHost: " + u.hostport + "\r\nConnection: close\r\n" +
                          "User-Agent: TokenStation5\r\nAccept-Encoding: identity\r\n";
        if (start > 0) req += "Range: bytes=" + std::to_string(start) + "-\r\n";
        if (start > 0 && !validator.empty()) req += "If-Range: " + validator + "\r\n";
        req += "\r\n";
        const auto asked = clock::now();
        if (!c.write_all(req, err)) return false;

        // The status line and headers; whatever follows them is body.
        std::string head;
        size_t he;
        while ((he = head.find("\r\n\r\n")) == std::string::npos) {
            if (head.size() > (64u << 10)) {
                err = "the response's headers are too long";
                return false;
            }
            char buf[16384];
            int n = c.read_some(buf, sizeof buf, err);
            if (n < 0) return false;
            if (n == 0) {
                err = "the server closed the connection before answering";
                return false;
            }
            head.append(buf, (size_t)n);
        }
        std::string ahead = head.substr(he + 4);
        head.resize(he);
        if (head.compare(0, 5, "HTTP/") != 0) {
            err = "not an HTTP response from " + u.host;
            return false;
        }
        size_t sp = head.find(' ');
        const int status = sp == std::string::npos ? 0 : atoi(head.c_str() + sp + 1);
        trace("the server answered %d (%ld ms)", status, ms_since(asked));
        const std::string location = header(head, "Location");
        if ((status == 301 || status == 302 || status == 303 || status == 307 || status == 308) && !location.empty()) {
            if (hop >= 5) {
                err = "more than 5 redirects";
                return false;
            }
            Url next;
            if (!parse_url(resolve(u, location), next, err)) return false;
            trace("redirected to %s", next.shown.c_str());
            const auto closing = clock::now();
#ifdef TS5_HTTPS
            if (c.tls) mbedtls_ssl_free(&c.ssl);
            c.tls = false;
#endif
            close(c.fd);
            c.fd = -1;
            trace("closed (%ld ms)", ms_since(closing));
            u = next;
            continue;
        }
        // 206 goes on from where the file stops, if the range agrees; 200 is
        // the whole file, so the part starts over; 416 past the end means
        // the part already is the whole file, or is longer than it.
        const std::string range = header(head, "Content-Range");
        unsigned long long rs = 0, re = 0;
        char rt[32] = "";
        if (status == 416 && start > 0 && sscanf(range.c_str(), "bytes */%31s", rt) == 1 &&
            isdigit((unsigned char)rt[0])) {
            if (strtoull(rt, nullptr, 10) == start) {
                total = start;
                break;
            }
            err = "the part is " + std::to_string(start) + " bytes, longer than the file's " + rt;
            return damaged();
        }
        if (status == 206) {
            if (sscanf(range.c_str(), "bytes %llu-%llu/%31s", &rs, &re, rt) < 2) {
                err = "the server answered 206 without a usable Content-Range";
                return false;
            }
            if (rs != start) {
                err = "the server's range starts at " + std::to_string(rs) + ", not " + std::to_string(start);
                return false;
            }
            total = isdigit((unsigned char)rt[0]) ? strtoull(rt, nullptr, 10) : re + 1;
        } else if (status == 200) {
            if (start > 0) {
                trace("the server sent the whole file: starting over");
                if (fseeko(f, 0, SEEK_SET) != 0 || ftruncate(fileno(f), 0) != 0) {
                    err = std::string("cannot truncate the file: ") + strerror(errno);
                    return false;
                }
                hash.start();
                start = done = 0;
                progress.done = 0;
            }
            // What this version of the file is called, for a resume's
            // If-Range: its ETag, unless weak (W/..., which may not go
            // there); without one, its Last-Modified.
            validator = header(head, "ETag");
            if (validator.empty()) validator = header(head, "Last-Modified");
            else if (validator.compare(0, 2, "W/") == 0) validator.clear();
            if (!write_from(from_path, source, validator))
                trace("cannot note where the part is from (%s)", strerror(errno));
        } else {
            err = "the server at " + u.host + " answered " + std::to_string(status);
            return false;
        }
        // Chunked wins over a Content-Length sent along with it (RFC 9112,
        // 6.3): counted, the chunk lines would pass for bytes of the file.
        const std::string length = header(head, "Content-Length");
        std::string te = header(head, "Transfer-Encoding");
        for (auto &ch : te) ch = (char)tolower((unsigned char)ch);
        const Body::Kind kind = te.find("chunked") != std::string::npos   ? Body::Chunked
                                : !length.empty()                         ? Body::Counted
                                                                          : Body::ToClose;
        const uint64_t length_n = kind == Body::Counted ? strtoull(length.c_str(), nullptr, 10) : 0;
        if (kind == Body::Counted && total == 0) total = start + length_n;
        progress.total = total;
        // A body only the server's close ends can't be told from one cut
        // short, so without a size it is taken only with a hash to check.
        if (kind == Body::ToClose && total == 0 && want.empty()) {
            err = "the server didn't say how big the file is, and there's no hash to check it by";
            return false;
        }
        trace("reading the body (%s)", kind == Body::Counted   ? (std::to_string(length_n) + " bytes").c_str()
                                       : kind == Body::Chunked ? "chunked"
                                                               : "until the server closes");
        Body body(c, std::move(ahead), kind, length_n);
        size_t filled = 0;
        auto t_win = clock::now();
        uint64_t b_win = done, last_checkpoint = done;
        for (;;) {
            int n = body.next(wbuf.data() + filled, block - filled, err);
            if (n < 0) return false;
            filled += (size_t)n;
            if (filled < block && n > 0) continue;
            // A full block, or the last one.
            if (filled > 0) {
                hash.update(wbuf.data(), filled);
                if (fwrite(wbuf.data(), 1, filled, f) != filled || ferror(f)) return write_failed();
                done += filled;
                filled = 0;
                progress.done = done;
            }
            const auto now = clock::now();
            if (seconds(now - t_win) >= 2) {
                progress.speed = (double)(done - b_win) / seconds(now - t_win);
                t_win = now;
                b_win = done;
            }
            if (sha && sha->on_checkpoint && Hasher::available() && done - last_checkpoint >= checkpoint_every) {
                if (fflush(f) != 0) return write_failed();
                last_checkpoint = done;
                sha->on_checkpoint(done, hash.state());
            }
            if (n == 0) break;
            if (!go_on()) return false;
        }
        if (!body.complete()) {
            err = "the connection closed after " + std::to_string(done) + " of " + std::to_string(total) + " bytes";
            return false;
        }
        break;
    }
    progress.total = total;
    if (fflush(f) != 0) return write_failed();
    if (total > 0 && done != total) {
        err = "the server sent " + std::to_string(done) + " of " + std::to_string(total) + " bytes";
        return done < total ? false : damaged();  // short: the next call asks for the rest
    }
    if (!want.empty()) {
        const std::string got = hash.finish();
        if (got != want) {
            err = "the file's SHA-256 is " + got + ", not " + want;
            return damaged();
        }
    }
    unlink(from_path.c_str());  // the file is whole: nothing to go on from
    trace("done (%llu bytes)", (unsigned long long)done);
    return true;
}

// A GET read into memory, up to max_bytes of the body (a Range header asks
// the server for no more), following up to 5 redirects with the same headers.
bool get(const std::string &url0, const std::vector<std::string> &headers, HttpResponse &resp, std::string &err,
         size_t max_bytes, int timeout_s) {
    std::string url = url0;
    for (int hop = 0; hop <= 5; hop++) {
        Url u;
        if (!parse_url(url, u, err)) return false;
#ifndef TS5_HTTPS
        if (u.https) {
            err = "this ts5 was built without HTTPS";
            return false;
        }
#endif
        Conn c;
        c.fd = connect_to(u.host, u.port, timeout_s, err);
        if (c.fd < 0) return false;
#ifdef TS5_HTTPS
        if (u.https && !start_tls(c, u.host, err)) return false;
#endif
        std::string req = "GET " + u.path + " HTTP/1.1\r\nHost: " + u.hostport +
                          "\r\nConnection: close\r\nUser-Agent: TokenStation5\r\nAccept-Encoding: identity\r\n";
        for (auto &h : headers) req += h + "\r\n";
        req += "\r\n";
        if (!c.write_all(req, err)) return false;
        trace("GET %s", u.shown.c_str());
        std::string raw;
        char buf[16384];
        size_t he;
        while ((he = raw.find("\r\n\r\n")) == std::string::npos) {
            if (raw.size() > (64u << 10)) {
                err = "the answer's headers are too long";
                return false;
            }
            int n = c.read_some(buf, sizeof buf, err);
            if (n <= 0) {
                if (err.empty()) err = "the server closed the connection before answering";
                return false;
            }
            raw.append(buf, (size_t)n);
        }
        if (raw.compare(0, 5, "HTTP/") != 0) {
            err = "not an HTTP answer from " + u.host;
            return false;
        }
        const std::string head = raw.substr(0, he);
        const size_t sp = head.find(' ');
        const int status = sp == std::string::npos ? 0 : atoi(head.c_str() + sp + 1);
        if (status >= 300 && status < 400 && status != 304) {
            const std::string loc = header(head, "location");
            if (loc.empty()) {
                err = "a redirect without a Location";
                return false;
            }
            url = resolve(u, loc);
            continue;
        }
        resp.status = status;
        std::string te = header(head, "transfer-encoding");
        for (auto &ch : te) ch = (char)tolower((unsigned char)ch);
        const std::string cl = header(head, "content-length");
        const Body::Kind kind = te.find("chunked") != std::string::npos ? Body::Chunked
                                : !cl.empty()                           ? Body::Counted
                                                                        : Body::ToClose;
        Body body(c, raw.substr(he + 4), kind, cl.empty() ? 0 : strtoull(cl.c_str(), nullptr, 10));
        resp.body.clear();
        while (resp.body.size() < max_bytes) {
            const int n = body.next(buf, std::min(sizeof buf, max_bytes - resp.body.size()), err);
            if (n < 0) return false;
            if (n == 0) break;
            resp.body.append(buf, (size_t)n);
        }
        return true;
    }
    err = "too many redirects";
    return false;
}

}  // namespace

bool random_bytes(unsigned char *out, size_t n) { return fill_random(out, n); }

std::string sha256_hex(const std::string &data) {
#ifdef TS5_HTTPS
    Hasher h;
    h.start();
    h.update(data.data(), data.size());
    return h.finish();
#else
    (void)data;
    return "";
#endif
}

bool post(const std::string &url, const std::vector<std::string> &headers, const std::string &body,
          HttpResponse &resp, std::string &err, int timeout_s) {
    // scheme://host[:port]/path
    size_t s = url.find("://");
    if (s == std::string::npos) {
        err = "bad URL " + url;
        return false;
    }
    const std::string scheme = url.substr(0, s);
    const bool https = scheme == "https";
    if (!https && scheme != "http") {
        err = "unsupported URL scheme " + scheme;
        return false;
    }
#ifndef TS5_HTTPS
    if (https) {
        err = "this ts5 was built without HTTPS";
        return false;
    }
#endif
    size_t p = url.find('/', s + 3);
    std::string hostport = url.substr(s + 3, p == std::string::npos ? std::string::npos : p - s - 3);
    const std::string path = p == std::string::npos ? "/" : url.substr(p);
    int port = https ? 443 : 80;
    std::string host = hostport;
    size_t colon = hostport.rfind(':');
    if (colon != std::string::npos) {
        host = hostport.substr(0, colon);
        port = atoi(hostport.c_str() + colon + 1);
    }

    Conn c;
    c.fd = connect_to(host, port, timeout_s, err);
    if (c.fd < 0) return false;
#ifdef TS5_HTTPS
    if (https && !start_tls(c, host, err)) return false;
#endif
    std::string req = "POST " + path + " HTTP/1.1\r\nHost: " + hostport + "\r\nConnection: close\r\n" +
                      "Content-Length: " + std::to_string(body.size()) + "\r\nUser-Agent: TokenStation5\r\n";
    for (auto &h : headers) req += h + "\r\n";
    req += "\r\n" + body;
    std::string raw;
    if (!c.write_all(req, err)) return false;
    trace("request sent (%zu bytes); reading the answer", req.size());
    if (!c.read_all(raw, err)) return false;
    trace("answer read (%zu bytes)", raw.size());

    size_t he = raw.find("\r\n\r\n");
    if (he == std::string::npos || raw.compare(0, 5, "HTTP/") != 0) {
        err = "not an HTTP response from " + host;
        return false;
    }
    const std::string head = raw.substr(0, he);
    size_t sp = head.find(' ');
    resp.status = sp == std::string::npos ? 0 : atoi(head.c_str() + sp + 1);
    std::string lower = head;
    for (auto &ch : lower) ch = (char)tolower((unsigned char)ch);
    resp.body.clear();
    if (lower.find("\r\ntransfer-encoding: chunked") != std::string::npos) {
        if (!dechunk(raw.substr(he + 4), resp.body)) {
            err = "incomplete chunked response from " + host;
            return false;
        }
    } else {
        resp.body = raw.substr(he + 4);
    }
    return true;
}

// TLS (certificate checks, key exchange) wants more stack than a server
// thread may have (a PS5 thread's default is small), so a request runs on a
// thread of its own with 1 MiB.
bool http_post(const std::string &url, const std::vector<std::string> &headers, const std::string &body,
               HttpResponse &resp, std::string &err, int timeout_s) {
    struct Job {
        const std::string &url;
        const std::vector<std::string> &headers;
        const std::string &body;
        HttpResponse &resp;
        std::string &err;
        int timeout_s;
        bool ok;
    } job{url, headers, body, resp, err, timeout_s, false};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1 << 20);
    pthread_t th;
    auto run = [](void *p) -> void * {
        Job &j = *(Job *)p;
        j.ok = post(j.url, j.headers, j.body, j.resp, j.err, j.timeout_s);
        return nullptr;
    };
    int rc = pthread_create(&th, &attr, run, &job);
    pthread_attr_destroy(&attr);
    if (rc != 0) return post(url, headers, body, resp, err, timeout_s);  // no thread: try on this one
    pthread_join(th, nullptr);
    return job.ok;
}

// A GET too, on a thread of its own with 1 MiB, for the same reason.
bool http_get(const std::string &url, const std::vector<std::string> &headers, HttpResponse &resp, std::string &err,
              size_t max_bytes, int timeout_s) {
    struct Job {
        const std::string &url;
        const std::vector<std::string> &headers;
        HttpResponse &resp;
        std::string &err;
        size_t max_bytes;
        int timeout_s;
        bool ok;
    } job{url, headers, resp, err, max_bytes, timeout_s, false};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1 << 20);
    pthread_t th;
    auto run = [](void *p) -> void * {
        Job &j = *(Job *)p;
        j.ok = get(j.url, j.headers, j.resp, j.err, j.max_bytes, j.timeout_s);
        return nullptr;
    };
    int rc = pthread_create(&th, &attr, run, &job);
    pthread_attr_destroy(&attr);
    if (rc != 0) return get(url, headers, resp, err, max_bytes, timeout_s);  // no thread: try on this one
    pthread_join(th, nullptr);
    return job.ok;
}

// A download too, on a thread of its own with 1 MiB, for the same reason.
bool http_get_file(const std::string &url, const std::string &path, uint64_t offset,
                   const std::string &expected_sha256, FetchProgress &progress, std::string &err, Sha256Resume *sha) {
    struct Job {
        const std::string &url, &path, &expected_sha256;
        uint64_t offset;
        FetchProgress &progress;
        std::string &err;
        Sha256Resume *sha;
        bool ok;
    } job{url, path, expected_sha256, offset, progress, err, sha, false};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 1 << 20);
    pthread_t th;
    auto run = [](void *p) -> void * {
        Job &j = *(Job *)p;
        j.ok = get_file(j.url, j.path, j.offset, j.expected_sha256, j.progress, j.err, j.sha);
        return nullptr;
    };
    int rc = pthread_create(&th, &attr, run, &job);
    pthread_attr_destroy(&attr);
    if (rc != 0) job.ok = get_file(url, path, offset, expected_sha256, progress, err, sha);  // no thread: on this one
    else pthread_join(th, nullptr);
    // A download that failed before its first byte leaves nothing behind.
    struct stat st;
    if (!job.ok && stat(path.c_str(), &st) == 0 && st.st_size == 0) {
        unlink(path.c_str());
        unlink((path + ".from").c_str());
    }
    return job.ok;
}

}  // namespace ts5

#ifdef TS5_HTTPS
// Mbed TLS's entropy source (MBEDTLS_ENTROPY_HARDWARE_ALT).
extern "C" int mbedtls_hardware_poll(void *, unsigned char *output, size_t len, size_t *olen) {
    if (!ts5::fill_random(output, len)) return -1;
    *olen = len;
    return 0;
}

// UTC from seconds since 1970 (MBEDTLS_PLATFORM_GMTIME_R_ALT): Howard Hinnant's
// days-to-civil algorithm.
extern "C" struct tm *mbedtls_platform_gmtime_r(const mbedtls_time_t *tt, struct tm *out) {
    int64_t t = (int64_t)*tt, days = t / 86400, secs = t % 86400;
    if (secs < 0) secs += 86400, days--;
    int64_t z = days + 719468, era = (z >= 0 ? z : z - 146096) / 146097;
    int64_t doe = z - era * 146097, yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y = yoe + era * 400, doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153;
    int64_t d = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9;
    memset(out, 0, sizeof *out);
    out->tm_year = (int)(y + (m <= 2) - 1900);
    out->tm_mon = (int)m - 1;
    out->tm_mday = (int)d;
    out->tm_hour = (int)(secs / 3600);
    out->tm_min = (int)(secs / 60 % 60);
    out->tm_sec = (int)(secs % 60);
    return out;
}
#endif
