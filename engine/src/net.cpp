#include "net.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>

#ifdef TS5_HTTPS
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/entropy.h>
#include <mbedtls/error.h>
#include <mbedtls/platform_util.h>
#include <mbedtls/ssl.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>
#endif

#ifdef TS5_PS5
// The console's resolver and random numbers (libSceNet, libSceRandom): a title
// has no /etc/resolv.conf, and its libc exports no getaddrinfo.
extern "C" {
int sceNetInit(void);
int sceNetPoolCreate(const char *name, int size, int flags);
int sceNetResolverCreate(const char *name, int pool, int flags);
int sceNetResolverStartNtoa(int resolver, const char *host, uint32_t *addr, int timeout_us, int retries, int flags);
int sceNetResolverDestroy(int resolver);
int sceRandomGetRandomNumber(void *buf, size_t size);
}
#else
#include <fcntl.h>
#endif

namespace ts5 {

namespace {

// One line per step of a request in the log, so a request that never comes
// back shows how far it got.
void trace(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ts5: net: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

bool fill_random(unsigned char *out, size_t n) {
#ifdef TS5_PS5
    while (n > 0) {
        size_t k = n < 64 ? n : 64;  // the most one call gives
        if (sceRandomGetRandomNumber(out, k) != 0) return false;
        out += k;
        n -= k;
    }
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

int connect_to(const std::string &host, int port, int timeout_s, std::string &err) {
    std::vector<uint32_t> addrs;
    trace("resolving %s", host.c_str());
    if (!resolve(host, addrs, err)) return -1;
    char ip[INET_ADDRSTRLEN] = "?";
    inet_ntop(AF_INET, &addrs[0], ip, sizeof ip);
    trace("%s is %s; connecting to port %d", host.c_str(), ip, port);
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
        if (connect(fd, (sockaddr *)&sa, sizeof sa) == 0) return fd;
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
        static const char pers[] = "ts5";
        int rc = mbedtls_ctr_drbg_seed(&t.drbg, mbedtls_entropy_func, &t.entropy, (const unsigned char *)pers,
                                       sizeof pers);
        if (rc != 0) { t.why = "no random numbers for TLS (-0x" + Conn::hex(-rc) + ")"; return; }
        trace("random numbers ok; reading the CA bundle");
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
        mbedtls_ssl_conf_rng(&t.conf, mbedtls_ctr_drbg_random, &t.drbg);
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

}  // namespace

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
