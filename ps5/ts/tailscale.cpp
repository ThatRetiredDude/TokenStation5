#include "tailscale.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>

#include "h2.h"
#include "json.h"
#include "log.h"
#include "nacl.h"
#include "net.h"
#include "noise.h"

namespace ts {

namespace {

using ts5::Json;

// The protocol version this client speaks (Tailscale's tailcfg
// CapabilityVersion): 123 is Tailscale 1.86's.
constexpr int kCapVersion = 123;
// No message from the control server this long: the connection is dead.
constexpr double kControlSilence = 120;
// Disco messages (Tailscale's disco package): "TS💬", the sender's disco
// key, then a NaCl box (nonce, tag, ciphertext) of type, version, payload.
constexpr char kDiscoMagic[] = "TS\xf0\x9f\x92\xac";
constexpr size_t kDiscoMagicLen = 6, kDiscoHeader = kDiscoMagicLen + 32;
enum : uint8_t { kDiscoPing = 1, kDiscoPong = 2, kDiscoCallMeMaybe = 3 };
// A direct path that answered within this long carries the packets (Go's
// trustUDPAddrDuration); a device in use is pinged this often.
constexpr double kTrustDirect = 6.5, kPingEvery = 3;
// The UDP ports tried for direct paths (the PS5 refuses some, 51820 and
// 40000 among them); 0 is any.
constexpr uint16_t kDirectPorts[] = {9031, 9032, 9033, 0};

void log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void log(const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    ts5::log_line("ts5-title: tailscale: %s", line);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void wipe(void *p, size_t n) {
    volatile uint8_t *b = (volatile uint8_t *)p;
    while (n--) *b++ = 0;
}

std::string hex(const uint8_t *p, size_t n) {
    static const char d[] = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[p[i] >> 4];
        s += d[p[i] & 15];
    }
    return s;
}

std::string base64(const std::string &in) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = ((uint8_t)in[i] << 16) | ((uint8_t)in[i + 1] << 8) | (uint8_t)in[i + 2];
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += t[(v >> 6) & 63];
        out += t[v & 63];
    }
    if (i < in.size()) {
        uint32_t v = (uint8_t)in[i] << 16;
        if (i + 1 < in.size()) v |= (uint8_t)in[i + 1] << 8;
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += i + 1 < in.size() ? t[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

struct Url {
    bool tls = true;
    std::string host;
    int port = 443;
    bool default_port = true;
};

bool is_disco(const std::string &p) {
    return p.size() >= kDiscoHeader + 24 + 16 + 2 && memcmp(p.data(), kDiscoMagic, kDiscoMagicLen) == 0;
}

// An IPv4 address and port as one number (address in the high bits).
uint64_t addr_key(const sockaddr_in &a) { return ((uint64_t)ntohl(a.sin_addr.s_addr) << 16) | ntohs(a.sin_port); }

sockaddr_in addr_of(uint64_t k) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl((uint32_t)(k >> 16));
    a.sin_port = htons((uint16_t)k);
    return a;
}

std::string addr_text(uint64_t k) {
    char ip[INET_ADDRSTRLEN] = {};
    const sockaddr_in a = addr_of(k);
    inet_ntop(AF_INET, &a.sin_addr, ip, sizeof ip);
    return std::string(ip) + ":" + std::to_string((uint16_t)k);
}

// "1.2.3.4:5678"; IPv6 ones are left out.
bool parse_endpoint(const std::string &s, uint64_t &k) {
    const size_t colon = s.rfind(':');
    if (colon == std::string::npos || s.find(':') != colon) return false;
    sockaddr_in a{};
    if (inet_pton(AF_INET, s.substr(0, colon).c_str(), &a.sin_addr) != 1) return false;
    const int port = atoi(s.c_str() + colon + 1);
    if (port <= 0 || port > 65535) return false;
    a.sin_port = htons((uint16_t)port);
    k = addr_key(a);
    return true;
}

// The PS5's address on its home network: the one a packet out would leave
// from (nothing is sent). 0 when there's no network.
uint32_t local_ip() {
    const int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) return 0;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(53);
    inet_pton(AF_INET, "1.1.1.1", &a.sin_addr);
    sockaddr_in me{};
    socklen_t len = sizeof me;
    uint32_t ip = 0;
    if (connect(s, (const sockaddr *)&a, sizeof a) == 0 && getsockname(s, (sockaddr *)&me, &len) == 0)
        ip = me.sin_addr.s_addr;
    close(s);
    return ip;
}

bool parse_url(const std::string &s, Url &u) {
    std::string rest;
    if (s.rfind("https://", 0) == 0) {
        u.tls = true;
        u.port = 443;
        rest = s.substr(8);
    } else if (s.rfind("http://", 0) == 0) {
        u.tls = false;
        u.port = 80;
        rest = s.substr(7);
    } else {
        return false;
    }
    rest = rest.substr(0, rest.find('/'));
    const size_t colon = rest.rfind(':');
    if (colon != std::string::npos) {
        u.port = atoi(rest.c_str() + colon + 1);
        u.default_port = false;
        rest = rest.substr(0, colon);
    }
    u.host = rest;
    return !u.host.empty() && u.port > 0;
}

std::string trim_url(std::string s) {
    while (!s.empty() && s.back() == '/') s.pop_back();
    return s;
}

// The keys kept between starts, in the title's folder.
struct Keys {
    std::string server;  // the control server they're for
    uint8_t machine[32] = {}, node[32] = {}, disco[32] = {};
    std::string log_id;
    std::string account;
    bool have = false;
    ~Keys() {
        wipe(machine, sizeof machine);
        wipe(node, sizeof node);
        wipe(disco, sizeof disco);
    }
};

bool unhex(const std::string &s, uint8_t *out, size_t n) {
    uint8_t k[32];
    if (n > 32 || !parse_key(s, "", k)) return false;
    memcpy(out, k, n);
    wipe(k, sizeof k);
    return true;
}

bool load_keys(const std::string &path, Keys &k) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    int got = 0;
    while (fgets(line, sizeof line, f)) {
        std::string l = line;
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
        const size_t tab = l.find('\t');
        if (tab == std::string::npos) continue;
        const std::string name = l.substr(0, tab), v = l.substr(tab + 1);
        if (name == "server") k.server = v;
        else if (name == "machine") got += unhex(v, k.machine, 32);
        else if (name == "node") got += unhex(v, k.node, 32);
        else if (name == "disco") got += unhex(v, k.disco, 32);
        else if (name == "log_id") k.log_id = v;
        else if (name == "account") k.account = v;
    }
    fclose(f);
    wipe(line, sizeof line);
    k.have = got == 3 && !k.log_id.empty();
    return k.have;
}

bool save_keys(const std::string &path, const Keys &k) {
    const std::string tmp = path + ".new";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return false;
    fprintf(f, "# ts5 tailscale v1\nserver\t%s\nmachine\t%s\nnode\t%s\ndisco\t%s\nlog_id\t%s\naccount\t%s\n",
            k.server.c_str(), hex(k.machine, 32).c_str(), hex(k.node, 32).c_str(), hex(k.disco, 32).c_str(),
            k.log_id.c_str(), k.account.c_str());
    const bool ok = fflush(f) == 0;
    fclose(f);
    return ok && rename(tmp.c_str(), path.c_str()) == 0;
}

bool new_keys(Keys &k, const std::string &server) {
    uint8_t id[32];
    if (!new_private_key(k.machine) || !new_private_key(k.node) || !new_private_key(k.disco) ||
        !ts5::random_bytes(id, sizeof id))
        return false;
    k.server = server;
    k.log_id = hex(id, sizeof id);
    k.account.clear();
    k.have = true;
    return true;
}

// The connection to the control server: HTTP upgraded to Noise, HTTP/2 inside.
class Control {
public:
    bool connect(const Url &u, const uint8_t machine[32], const uint8_t control_pub[32], std::string &why) {
        // Tailscale's own apps try port 80 first (Noise needs no TLS under
        // it), then 443 with TLS; a server with a port of its own, just that.
        std::vector<std::pair<int, bool>> tries;
        if (u.default_port && u.tls) tries = {{80, false}, {443, true}};
        else tries = {{u.port, u.tls}};
        for (auto [port, tls] : tries) {
            if (try_connect(u.host, port, tls, machine, control_pub, why)) return true;
            log("control server %s:%d: %s", u.host.c_str(), port, why.c_str());
        }
        return false;
    }

    // One request; its stream.
    bool post(const std::string &path, const std::string &body, const std::string &node_key, int &id, std::string &why) {
        std::vector<std::pair<std::string, std::string>> h = {{"content-type", "application/json"}};
        if (!node_key.empty()) h.push_back({"ts-lb", node_key});
        return send(h2_.request("POST", host_, path, h, body, id), why);
    }

    // Reads what comes within wait_ms and handles it. False when the
    // connection is over.
    bool pump(int wait_ms, std::string &why) {
        char buf[16384];
        const int n = s_.read(buf, sizeof buf, wait_ms, why);
        if (n == -2) return true;
        if (n <= 0) {
            if (n == 0) why = "the control server closed the connection";
            return false;
        }
        last_rx_ = now_s();
        std::string plain;
        if (!noise_.open(buf, (size_t)n, plain, why)) return false;
        if (!early_done_) {
            early_ += plain;
            plain.clear();
            // An "early payload" may come first (5 bytes of magic, 4 of
            // length, JSON); without it, HTTP/2 starts at once.
            if (early_.size() < 9) return true;
            if (early_.compare(0, 5, "\xff\xff\xffTS", 5) == 0) {
                const uint32_t len = ((uint8_t)early_[5] << 24) | ((uint8_t)early_[6] << 16) | ((uint8_t)early_[7] << 8) |
                                     (uint8_t)early_[8];
                if (len > (1u << 20)) {
                    why = "a wrong early payload";
                    return false;
                }
                if (early_.size() < 9 + len) return true;
                plain = early_.substr(9 + len);
            } else {
                plain = early_;
            }
            early_.clear();
            early_done_ = true;
        }
        std::string reply;
        if (!h2_.feed(plain.data(), plain.size(), reply, why)) return false;
        if (!reply.empty() && !send(reply, why)) return false;
        return true;
    }

    H2 &h2() { return h2_; }
    double last_rx() const { return last_rx_; }
    void close() { s_.close(); }

private:
    bool try_connect(const std::string &host, int port, bool tls, const uint8_t machine[32], const uint8_t control_pub[32],
                     std::string &why) {
        noise_ = Noise();
        h2_ = H2();
        early_.clear();
        early_done_ = false;
        host_ = host;
        if (!s_.open(host, port, tls, 15, why)) return false;
        const std::string init = noise_.initiation(machine, control_pub, kCapVersion);
        const std::string req = "POST /ts2021 HTTP/1.1\r\nHost: " + host +
                                "\r\nUpgrade: tailscale-control-protocol\r\nConnection: upgrade\r\nX-Tailscale-Handshake: " +
                                base64(init) + "\r\nUser-Agent: TokenStation5\r\n\r\n";
        if (!s_.write(req.data(), req.size(), why)) return false;
        std::string in;
        size_t he;
        const double until = now_s() + 15;
        while ((he = in.find("\r\n\r\n")) == std::string::npos || in.size() < he + 4 + 3) {
            char buf[4096];
            const int n = s_.read(buf, sizeof buf, 500, why);
            if (n == -2 && now_s() < until) continue;
            if (n <= 0) {
                if (why.empty()) why = "no answer";
                return false;
            }
            in.append(buf, (size_t)n);
            if (in.size() > 65536) break;
            // An answer that isn't 101 has no Noise after it.
            if (he == std::string::npos && (he = in.find("\r\n\r\n")) != std::string::npos && in.find(" 101") > in.find("\r\n")) break;
        }
        if (in.compare(0, 5, "HTTP/") != 0 || in.find(" 101") == std::string::npos || in.find(" 101") > in.find("\r\n")) {
            why = "the server didn't take Tailscale's protocol: " + in.substr(0, in.find("\r\n"));
            return false;
        }
        std::string rest = in.substr(he + 4);
        // The Noise answer: 51 bytes, or an error's 3-byte header and text.
        while (true) {
            size_t want = 51;
            if (rest.size() >= 3 && (uint8_t)rest[0] == 3) want = 3 + (((uint8_t)rest[1] << 8) | (uint8_t)rest[2]);
            if (rest.size() >= want) break;
            char buf[4096];
            const int n = s_.read(buf, sizeof buf, 500, why);
            if (n == -2 && now_s() < until) continue;
            if (n <= 0) {
                if (why.empty()) why = "no answer to the handshake";
                return false;
            }
            rest.append(buf, (size_t)n);
        }
        const size_t used = (uint8_t)rest[0] == 3 ? rest.size() : 51;
        if (!noise_.finish(rest.substr(0, used), why)) return false;
        last_rx_ = now_s();
        if (!send(h2_.start(), why)) return false;
        if (rest.size() > used) {
            // What came after the handshake is Noise already.
            std::string plain;
            if (!noise_.open(rest.data() + used, rest.size() - used, plain, why)) return false;
            early_ = plain;
        }
        return true;
    }

    bool send(const std::string &plain, std::string &why) {
        const std::string rec = noise_.seal(plain.data(), plain.size());
        return s_.write(rec.data(), rec.size(), why);
    }

    ts5::Stream s_;
    Noise noise_;
    H2 h2_;
    std::string host_, early_;
    bool early_done_ = false;
    double last_rx_ = 0;
};

Json str(const std::string &s) { return Json::string(s); }
Json num(long long v) {
    Json j;
    j.type = Json::NUM;
    j.n = (double)v;
    j.s = std::to_string(v);
    return j;
}
Json boolean(bool v) {
    Json j;
    j.type = Json::BOOL;
    j.b = v;
    return j;
}

// A device in the tailnet, as the control server last described it.
struct Node {
    std::string key;  // "nodekey:..."
    std::string name, address;
    std::string disco;  // "discokey:..."
    std::vector<std::string> endpoints;
};

}  // namespace

Client::~Client() {
    quit_ = true;
    if (control_.joinable()) control_.join();
    if (relay_.joinable()) relay_.join();
    if (direct_.joinable()) direct_.join();
    if (udp_ >= 0) close(udp_);
    wipe(node_priv_, sizeof node_priv_);
    wipe(disco_priv_, sizeof disco_priv_);
}

void Client::apply(const Config &c) {
    std::lock_guard<std::mutex> lk(mu_);
    cfg_ = c;
    cfg_.control_url = trim_url(cfg_.control_url.empty() ? "https://controlplane.tailscale.com" : cfg_.control_url);
    cfg_gen_++;
    if (!c.on) {
        status_ = Status();
        map_.address.clear();
        map_.peers.clear();
        map_.version++;
    }
    if (c.on && !started_) {
        started_ = true;
        control_ = std::thread([this] { control_loop(); });
        relay_ = std::thread([this] { relay_loop(); });
        direct_ = std::thread([this] { direct_loop(); });
    }
}

void Client::sign_out() {
    std::lock_guard<std::mutex> lk(mu_);
    sign_out_ = true;
    cfg_gen_++;
}

Status Client::status() {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

uint64_t Client::netmap_version() {
    std::lock_guard<std::mutex> lk(mu_);
    return map_.version;
}

Netmap Client::netmap() {
    std::lock_guard<std::mutex> lk(mu_);
    return map_;
}

void Client::send(const uint8_t to[32], const void *data, size_t len) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = paths_.find(std::string((const char *)to, 32));
    if (it != paths_.end()) {
        Path &pa = it->second;
        const double now = now_s();
        pa.wanted = now;
        if (udp_ >= 0 && pa.best && now - pa.best_at < kTrustDirect) {
            const sockaddr_in a = addr_of(pa.best);
            if (sendto(udp_, data, len, MSG_DONTWAIT, (const sockaddr *)&a, sizeof a) == (ssize_t)len) return;
        }
    }
    if (outbox_.size() > 1024) outbox_.pop_front();  // the relay is gone: keep the newest
    outbox_.emplace_back(std::string((const char *)to, 32), std::string((const char *)data, len));
}

bool Client::receive(uint8_t from[32], std::string &data) {
    std::lock_guard<std::mutex> lk(mu_);
    if (inbox_.empty()) return false;
    memcpy(from, inbox_.front().first.data(), 32);
    data.swap(inbox_.front().second);
    inbox_.pop_front();
    return true;
}

bool Client::direct(const uint8_t key[32]) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = paths_.find(std::string((const char *)key, 32));
    return it != paths_.end() && it->second.best && now_s() - it->second.best_at < kTrustDirect;
}

void Client::set_state(Status::State s, const std::string &problem) {
    std::lock_guard<std::mutex> lk(mu_);
    status_.state = s;
    status_.problem = problem;
    if (s != Status::NeedsSignIn) status_.sign_in_url.clear();
}

void Client::control_loop() {
    double backoff = 1;
    // The sign-in page last shown, and the device key it's for: a new
    // connection waits on the same page, so the code on the TV doesn't change
    // under someone signing in.
    std::string login_url, login_key;
    while (!quit_) {
        Config cfg;
        uint64_t gen;
        bool sign_out;
        {
            std::lock_guard<std::mutex> lk(mu_);
            cfg = cfg_;
            gen = cfg_gen_;
            sign_out = sign_out_;
        }
        auto changed = [&] {
            std::lock_guard<std::mutex> lk(mu_);
            return quit_ || cfg_gen_ != gen;
        };
        if (!cfg.on && !sign_out) {
            for (int i = 0; i < 10 && !changed(); i++) usleep(50 * 1000);
            continue;
        }
        Keys keys;
        load_keys(cfg.state_file, keys);
        if (!keys.have || keys.server != cfg.control_url) {
            if (sign_out) {
                std::lock_guard<std::mutex> lk(mu_);
                sign_out_ = false;
                continue;
            }
            if (!new_keys(keys, cfg.control_url) || !save_keys(cfg.state_file, keys)) {
                set_state(Status::Problem, "can't make or keep the PS5's keys");
                for (int i = 0; i < 100 && !changed(); i++) usleep(50 * 1000);
                continue;
            }
            log("new keys for %s", cfg.control_url.c_str());
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            memcpy(node_priv_, keys.node, 32);
            memcpy(map_.node_priv, keys.node, 32);
            have_node_key_ = true;
            memcpy(disco_priv_, keys.disco, 32);
            have_disco_ = true;
            if (status_.state == Status::Off || status_.state == Status::Problem) status_.state = Status::Starting;
            status_.account = keys.account;
        }
        uint8_t node_pub[32], disco_pub[32];
        public_key(node_pub, keys.node);
        public_key(disco_pub, keys.disco);
        const std::string node_key = key_text("nodekey:", node_pub), disco_key = key_text("discokey:", disco_pub);

        auto fail = [&](const std::string &why) {
            log("%s", why.c_str());
            set_state(Status::Problem, why);
            const double until = now_s() + backoff;
            backoff = backoff * 2 > 60 ? 60 : backoff * 2;
            while (now_s() < until && !changed()) usleep(100 * 1000);
        };

        // The control server's key, then the connection.
        Url u;
        if (!parse_url(cfg.control_url, u)) {
            fail("the control server address " + cfg.control_url + " isn't a web address");
            continue;
        }
        ts5::HttpResponse r;
        std::string why;
        if (!ts5::http_get(cfg.control_url + "/key?v=" + std::to_string(kCapVersion), {}, r, why, 64 << 10, 15) ||
            r.status != 200) {
            fail("can't reach the control server: " + (why.empty() ? "HTTP " + std::to_string(r.status) : why));
            continue;
        }
        Json kj;
        uint8_t control_pub[32];
        if (!Json::parse(r.body, kj) || !parse_key(kj["publicKey"].str(), "mkey:", control_pub)) {
            fail("the control server sent no key it can use");
            continue;
        }
        Control c;
        if (!c.connect(u, keys.machine, control_pub, why)) {
            fail("can't connect to the control server: " + why);
            continue;
        }
        log("connected to %s", cfg.control_url.c_str());

        auto hostinfo = [&]() {
            Json h = Json::object();
            h.set("IPNVersion", str("1.86.5-ts5-" + cfg.version));
            h.set("BackendLogID", str(keys.log_id));
            h.set("OS", str("freebsd"));
            h.set("OSVersion", str("PlayStation 5"));
            h.set("DeviceModel", str("PlayStation 5"));
            h.set("App", str("TokenStation5"));
            h.set("Hostname", str(cfg.hostname));
            int home;
            {
                std::lock_guard<std::mutex> lk(mu_);
                home = home_region_;
            }
            Json ni = Json::object();
            ni.set("PreferredDERP", num(home));
            h.set("NetInfo", ni);
            return h;
        };
        // Waits for a whole response to stream id; false when the
        // connection broke or the settings changed.
        auto wait_for = [&](int id, std::string &body, int &status) -> bool {
            while (!changed()) {
                H2::Response *resp = c.h2().response(id);
                if (resp && resp->done) {
                    status = resp->status;
                    body = resp->body;
                    c.h2().forget(id);
                    return !resp->reset;
                }
                if (!c.pump(1000, why)) return false;
            }
            why = "";
            return false;
        };

        // Sign-in: the register request, answered at once when the PS5 is
        // already a device; else with a page to approve it on, asked again
        // with that page until someone has.
        Json reg = Json::object();
        reg.set("Version", num(kCapVersion));
        reg.set("NodeKey", str(node_key));
        reg.set("OldNodeKey", str(key_text("nodekey:", (const uint8_t *)std::string(32, '\0').data())));
        reg.set("Hostinfo", hostinfo());
        if (sign_out) reg.set("Expiry", str("1970-01-01T00:02:03Z"));
        else if (!cfg.auth_key.empty()) {
            Json a = Json::object();
            a.set("AuthKey", str(cfg.auth_key));
            reg.set("Auth", a);
        }
        bool authorized = false;
        std::string followup;
        if (!sign_out && !login_url.empty() && login_key == node_key) {
            followup = login_url;
            std::lock_guard<std::mutex> lk(mu_);
            status_.state = Status::NeedsSignIn;
            status_.sign_in_url = login_url;
            status_.problem.clear();
        }
        while (!authorized) {
            if (!followup.empty()) reg.set("Followup", str(followup));
            int id = 0, status = 0;
            std::string body;
            if (!c.post("/machine/register", reg.dump(), node_key, id, why) || !wait_for(id, body, status)) break;
            Json rr;
            if (status != 200 || !Json::parse(body, rr)) {
                why = "the control server refused the sign-in (HTTP " + std::to_string(status) + "): " + body.substr(0, 200);
                break;
            }
            if (!rr["Error"].str().empty()) {
                why = rr["Error"].str();
                login_url.clear();  // the page may be no good any more
                break;
            }
            if (sign_out) {
                log("signed out");
                Keys fresh;
                new_keys(fresh, cfg.control_url);
                memcpy(fresh.machine, keys.machine, 32);
                save_keys(cfg.state_file, fresh);
                std::lock_guard<std::mutex> lk(mu_);
                sign_out_ = false;
                status_ = Status();
                map_.address.clear();
                map_.peers.clear();
                map_.version++;
                break;
            }
            if (rr["NodeKeyExpired"].boolean(false)) {
                // A new device key, and the sign-in again.
                log("the PS5's key expired; making a new one");
                new_private_key(keys.node);
                save_keys(cfg.state_file, keys);
                why = "the PS5's key expired";
                break;
            }
            const std::string login = rr["Login"]["LoginName"].str();
            if (!login.empty() && login != keys.account) {
                keys.account = login;
                save_keys(cfg.state_file, keys);
            }
            const std::string url = rr["AuthURL"].str();
            if (rr["MachineAuthorized"].boolean(false) && url.empty()) {
                authorized = true;
                login_url.clear();
                break;
            }
            if (!url.empty()) {
                if (url != login_url) log("waiting for the sign-in at %s", url.c_str());
                login_url = url;
                login_key = node_key;
                std::lock_guard<std::mutex> lk(mu_);
                status_.state = Status::NeedsSignIn;
                status_.sign_in_url = url;
                status_.problem.clear();
                followup = url;
            } else if (followup.empty()) {
                // Signed in, but an admin has to approve the device.
                std::lock_guard<std::mutex> lk(mu_);
                status_.state = Status::NeedsSignIn;
                status_.problem = "waiting for the tailnet's admin to approve the PS5";
                followup = "";
                usleep(5 * 1000 * 1000);
            }
        }
        if (sign_out) continue;
        if (!authorized) {
            if (!changed()) fail(why.empty() ? "the control server connection ended" : why);
            continue;
        }
        backoff = 1;
        {
            std::lock_guard<std::mutex> lk(mu_);
            status_.account = keys.account;
            if (status_.state != Status::Connected) status_.state = Status::Connecting;
            status_.problem.clear();
            status_.sign_in_url.clear();
        }
        log("signed in%s%s", keys.account.empty() ? "" : " as ", keys.account.c_str());

        // The map: one long response, a message at a time (4 bytes of
        // length, little-endian, then JSON), the first with everything.
        auto map_request = [&](bool stream) {
            Json m = Json::object();
            m.set("Version", num(kCapVersion));
            m.set("KeepAlive", boolean(stream));
            m.set("NodeKey", str(node_key));
            m.set("DiscoKey", str(disco_key));
            m.set("Stream", boolean(stream));
            m.set("OmitPeers", boolean(!stream));
            m.set("Compress", str(""));
            m.set("Hostinfo", hostinfo());
            // Where devices may reach the PS5 straight (the direct thread's).
            Json eps, types;
            eps.type = types.type = Json::ARR;
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (const std::string &e : endpoints_) {
                    eps.a.push_back(str(e));
                    types.a.push_back(num(1));  // on its own network
                }
            }
            m.set("Endpoints", eps);
            m.set("EndpointTypes", types);
            return m.dump();
        };
        int map_id = 0;
        if (!c.post("/machine/map", map_request(true), node_key, map_id, why)) {
            fail("the map request failed: " + why);
            continue;
        }
        int reported_home;
        uint64_t reported_endpoints;
        {
            std::lock_guard<std::mutex> lk(mu_);
            reported_home = home_region_;
            reported_endpoints = endpoints_gen_;
        }
        std::map<std::string, Node> nodes;  // by node ID
        std::string self_name, self_address;
        int update_id = 0;
        bool ok = true;
        while (ok && !changed()) {
            if (!c.pump(1000, why)) {
                ok = false;
                break;
            }
            if (now_s() - c.last_rx() > kControlSilence) {
                why = "the control server went quiet";
                ok = false;
                break;
            }
            if (update_id) {
                H2::Response *ur = c.h2().response(update_id);
                if (ur && ur->done) {
                    if (ur->status != 200) log("telling the control server the relay failed (HTTP %d)", ur->status);
                    c.h2().forget(update_id);
                    update_id = 0;
                }
            }
            // The relay chosen since, or new direct addresses: tell the
            // control server, so the other devices use them.
            int home;
            uint64_t eps;
            {
                std::lock_guard<std::mutex> lk(mu_);
                home = home_region_;
                eps = endpoints_gen_;
            }
            if ((home != reported_home || eps != reported_endpoints) && !update_id && !c.h2().going_away()) {
                if (!c.post("/machine/map", map_request(false), node_key, update_id, why)) {
                    ok = false;
                    break;
                }
                reported_home = home;
                reported_endpoints = eps;
            }
            H2::Response *mr = c.h2().response(map_id);
            if (!mr) {
                why = "the map request vanished";
                ok = false;
                break;
            }
            if (mr->done && mr->body.size() < 4) {
                why = mr->status != 200 ? "the map request failed (HTTP " + std::to_string(mr->status) + "): " + mr->body.substr(0, 200)
                                        : "the control server ended the map";
                ok = false;
                break;
            }
            bool changed_map = false;
            while (mr->body.size() >= 4) {
                const uint8_t *b = (const uint8_t *)mr->body.data();
                const size_t len = (size_t)b[0] | ((size_t)b[1] << 8) | ((size_t)b[2] << 16) | ((size_t)b[3] << 24);
                if (mr->body.size() < 4 + len) break;
                Json m;
                const bool parsed = Json::parse(mr->body.substr(4, len), m);
                mr->body.erase(0, 4 + len);
                if (!parsed) {
                    log("a map message that isn't JSON (%zu bytes)", len);
                    continue;
                }
                if (m["KeepAlive"].boolean(false)) continue;
                auto take_node = [](const Json &n, Node &out) {
                    out.key = n["Key"].str();
                    std::string name = n["Name"].str();
                    out.name = name.substr(0, name.find('.'));
                    for (const Json &a : n["Addresses"].a) {
                        const std::string s = a.str();
                        if (s.find(':') == std::string::npos) {
                            out.address = s.substr(0, s.find('/'));
                            break;
                        }
                    }
                    out.disco = n["DiscoKey"].str();
                    out.endpoints.clear();
                    for (const Json &e : n["Endpoints"].a) out.endpoints.push_back(e.str());
                };
                const Json &self = m["Node"];
                if (self.type == Json::OBJ) {
                    Node me;
                    take_node(self, me);
                    self_name = self["Name"].str();
                    while (!self_name.empty() && self_name.back() == '.') self_name.pop_back();
                    self_address = me.address;
                    changed_map = true;
                }
                const Json &dm = m["DERPMap"];
                if (dm.type == Json::OBJ) {
                    std::vector<Region> regions;
                    for (const auto &[k, rj] : dm["Regions"].o) {
                        if (rj.type != Json::OBJ) continue;  // null: a region taken away
                        Region rg;
                        rg.id = (int)rj["RegionID"].num(0);
                        rg.name = rj["RegionName"].str();
                        rg.code = rj["RegionCode"].str();
                        if (rj["Avoid"].boolean(false) || rg.id <= 0) continue;
                        for (const Json &nj : rj["Nodes"].a) {
                            if (nj["STUNOnly"].boolean(false)) continue;
                            const int port = (int)nj["DERPPort"].num(0);
                            rg.insecure = nj["InsecureForTests"].boolean(false);
                            rg.nodes.push_back({nj["HostName"].str(), port > 0 ? port : 443});
                        }
                        if (!rg.nodes.empty()) regions.push_back(rg);
                    }
                    std::lock_guard<std::mutex> lk(mu_);
                    regions_ = regions;
                    bool still = false;
                    for (const Region &rg : regions_) still |= rg.id == home_region_;
                    if (!still) home_region_ = 0;
                }
                if (m["Peers"].type == Json::ARR) {
                    nodes.clear();
                    for (const Json &p : m["Peers"].a) take_node(p, nodes[p["ID"].s]);
                    changed_map = true;
                }
                for (const Json &p : m["PeersChanged"].a) {
                    take_node(p, nodes[p["ID"].s]);
                    changed_map = true;
                }
                for (const Json &id : m["PeersRemoved"].a) {
                    nodes.erase(id.s);
                    changed_map = true;
                }
                for (const Json &p : m["PeersChangedPatch"].a) {
                    auto it = nodes.find(p["NodeID"].s);
                    if (it == nodes.end()) continue;
                    if (!p["Key"].str().empty()) it->second.key = p["Key"].str();
                    if (!p["DiscoKey"].str().empty()) it->second.disco = p["DiscoKey"].str();
                    if (p["Endpoints"].type == Json::ARR) {
                        it->second.endpoints.clear();
                        for (const Json &e : p["Endpoints"].a) it->second.endpoints.push_back(e.str());
                    }
                    changed_map = true;
                }
            }
            if (changed_map) {
                std::vector<Peer> peers;
                for (const auto &[id, n] : nodes) {
                    Peer p;
                    if (!parse_key(n.key, "nodekey:", p.key) || n.address.empty()) continue;
                    p.name = n.name;
                    p.address = n.address;
                    parse_key(n.disco, "discokey:", p.disco);
                    p.endpoints = n.endpoints;
                    peers.push_back(p);
                }
                std::lock_guard<std::mutex> lk(mu_);
                // The direct paths follow the devices.
                std::map<std::string, Path> paths;
                for (const Peer &p : peers) {
                    const std::string k((const char *)p.key, 32);
                    Path pa;
                    auto old = paths_.find(k);
                    if (old != paths_.end()) pa = old->second;
                    pa.disco.assign((const char *)p.disco, 32);
                    std::vector<uint64_t> eps;
                    for (const std::string &e : p.endpoints) {
                        uint64_t a;
                        if (parse_endpoint(e, a)) eps.push_back(a);
                    }
                    // Ones learned from its call-me-maybe stay too.
                    for (uint64_t a : pa.endpoints)
                        if (std::find(eps.begin(), eps.end(), a) == eps.end() && eps.size() < 16) eps.push_back(a);
                    pa.endpoints = eps;
                    paths[k] = pa;
                }
                paths_.swap(paths);
                for (auto it = by_addr_.begin(); it != by_addr_.end();)
                    it = paths_.count(it->second) ? std::next(it) : by_addr_.erase(it);
                map_.address = self_address;
                map_.peers = peers;
                map_.version++;
                status_.name = self_name;
                status_.address = self_address;
                status_.devices = (int)peers.size();
            }
        }
        c.close();
        if (!changed()) fail(why.empty() ? "the control server connection ended" : why);
    }
}

void Client::relay_loop() {
    double backoff = 1;
    while (!quit_) {
        bool on;
        uint64_t gen;
        std::vector<Region> regions;
        int home;
        uint8_t priv[32];
        {
            std::lock_guard<std::mutex> lk(mu_);
            on = cfg_.on && have_node_key_ && !map_.address.empty();
            gen = cfg_gen_;
            regions = regions_;
            home = home_region_;
            memcpy(priv, node_priv_, 32);
        }
        if (!on || regions.empty()) {
            usleep(200 * 1000);
            continue;
        }
        auto changed = [&] {
            std::lock_guard<std::mutex> lk(mu_);
            return quit_ || cfg_gen_ != gen || !cfg_.on || memcmp(priv, node_priv_, 32) != 0 ||
                   (home != 0 && home_region_ != home);
        };
        // The nearest relay: the quickest to connect to.
        if (home == 0) {
            double best = 1e9;
            for (const Region &rg : regions) {
                ts5::Stream probe;
                std::string why;
                const double t = now_s();
                if (!probe.open(rg.nodes[0].first, rg.nodes[0].second, false, 2, why)) continue;
                const double took = now_s() - t;
                if (took < best) {
                    best = took;
                    home = rg.id;
                }
            }
            if (home == 0) {
                log("can't reach any relay");
                set_state(Status::Problem, "can't reach any of Tailscale's relays");
                usleep(10 * 1000 * 1000);
                continue;
            }
            std::lock_guard<std::mutex> lk(mu_);
            home_region_ = home;
        }
        const Region *rg = nullptr;
        for (const Region &r : regions)
            if (r.id == home) rg = &r;
        if (!rg) continue;
        {
            std::lock_guard<std::mutex> lk(mu_);
            status_.relay = rg->name.empty() ? rg->code : rg->name;
        }
        auto fail = [&](const std::string &why) {
            log("relay %s: %s", rg->nodes[0].first.c_str(), why.c_str());
            const double until = now_s() + backoff;
            backoff = backoff * 2 > 30 ? 30 : backoff * 2;
            while (now_s() < until && !changed()) usleep(100 * 1000);
        };

        ts5::Stream s;
        std::string why;
        const std::string host = rg->nodes[0].first;
        // At most TLS 1.2: the relays' Ed25519 certificate (net.cpp).
        if (!s.open(host, rg->nodes[0].second, !rg->insecure, 10, why, true)) {
            fail(why);
            continue;
        }
        const std::string req = "GET /derp HTTP/1.1\r\nHost: " + host +
                                "\r\nUpgrade: DERP\r\nConnection: Upgrade\r\nUser-Agent: TokenStation5\r\n\r\n";
        if (!s.write(req.data(), req.size(), why)) {
            fail(why);
            continue;
        }
        std::string in;
        size_t he;
        const double until = now_s() + 10;
        bool bad = false;
        while ((he = in.find("\r\n\r\n")) == std::string::npos) {
            char buf[4096];
            const int n = s.read(buf, sizeof buf, 500, why);
            if (n == -2 && now_s() < until) continue;
            if (n <= 0 || in.size() > 65536) {
                bad = true;
                break;
            }
            in.append(buf, (size_t)n);
        }
        if (bad || in.find(" 101") == std::string::npos || in.find(" 101") > in.find("\r\n")) {
            fail("the relay didn't take the connection: " + (bad ? why : in.substr(0, in.find("\r\n"))));
            continue;
        }
        in.erase(0, he + 4);

        // Frames: 1 byte of type, 4 of length (big-endian), the payload.
        auto frame = [](uint8_t type, const std::string &payload) {
            std::string f(5, '\0');
            f[0] = (char)type;
            const uint32_t n = (uint32_t)payload.size();
            f[1] = (char)(n >> 24);
            f[2] = (char)(n >> 16);
            f[3] = (char)(n >> 8);
            f[4] = (char)n;
            return f + payload;
        };
        bool hello = false, ok = true;
        uint8_t server_key[32];
        double last_rx = now_s();
        std::string out;
        while (ok && !changed()) {
            // What the tunnel has for the relay.
            {
                std::lock_guard<std::mutex> lk(mu_);
                while (hello && !outbox_.empty() && out.size() < (256u << 10)) {
                    out += frame(0x04, outbox_.front().first + outbox_.front().second);  // SendPacket
                    outbox_.pop_front();
                }
            }
            if (!out.empty()) {
                if (!s.write(out.data(), out.size(), why)) {
                    ok = false;
                    break;
                }
                out.clear();
            }
            // Whole frames first (the relay's first may have come with its
            // HTTP answer), then whatever else has arrived.
            while (in.size() >= 5) {
                const uint8_t *h = (const uint8_t *)in.data();
                const size_t len = ((size_t)h[1] << 24) | ((size_t)h[2] << 16) | ((size_t)h[3] << 8) | h[4];
                if (len > (1u << 20)) {
                    why = "a relay frame too big";
                    ok = false;
                    break;
                }
                if (in.size() < 5 + len) break;
                const uint8_t type = h[0];
                const std::string p = in.substr(5, len);
                in.erase(0, 5 + len);
                switch (type) {
                    case 0x01: {  // ServerKey: magic, the relay's key. Then the PS5's hello.
                        if (p.size() < 40 || p.compare(0, 8, "DERP\xf0\x9f\x94\x91") != 0) {
                            why = "not a Tailscale relay";
                            ok = false;
                            break;
                        }
                        memcpy(server_key, p.data() + 8, 32);
                        uint8_t pub[32];
                        public_key(pub, priv);
                        const std::string sealed = box_seal(priv, server_key, "{\"version\":2,\"CanAckPings\":true}");
                        if (sealed.empty()) {
                            why = "the relay's key is no good";
                            ok = false;
                            break;
                        }
                        out += frame(0x02, std::string((const char *)pub, 32) + sealed);  // ClientInfo
                        out += frame(0x07, std::string(1, '\x01'));                      // NotePreferred: home relay
                        hello = true;
                        backoff = 1;
                        log("on relay %s (%s)", rg->name.c_str(), host.c_str());
                        {
                            std::lock_guard<std::mutex> lk(mu_);
                            if (status_.state == Status::Connecting || status_.state == Status::Problem) {
                                status_.state = Status::Connected;
                                status_.problem.clear();
                            }
                        }
                        break;
                    }
                    case 0x05:  // RecvPacket: who from, the packet
                        if (p.size() > 32) {
                            std::lock_guard<std::mutex> lk(mu_);
                            const std::string from = p.substr(0, 32), packet = p.substr(32);
                            if (is_disco(packet)) {
                                disco_in(packet, &from, 0);
                                break;
                            }
                            if (inbox_.size() > 1024) inbox_.pop_front();
                            inbox_.emplace_back(from, packet);
                        }
                        break;
                    case 0x12:  // Ping: answered with its 8 bytes
                        out += frame(0x13, p);
                        break;
                    case 0x14:  // Health: the relay's own problem, "" when it's over
                        if (!p.empty()) log("relay: %s", p.c_str());
                        break;
                    case 0x15:  // Restarting
                        why = "the relay is restarting";
                        ok = false;
                        break;
                    default:  // ServerInfo, KeepAlive, PeerGone, PeerPresent...
                        break;
                }
                if (!ok) break;
            }
            if (!ok || !out.empty()) continue;
            char buf[16384];
            const int n = s.read(buf, sizeof buf, 5, why);
            if (n == -2) {
                if (now_s() - last_rx > 150) {
                    why = "the relay went quiet";
                    ok = false;
                }
                continue;
            }
            if (n <= 0) {
                if (n == 0) why = "the relay closed the connection";
                ok = false;
                break;
            }
            last_rx = now_s();
            in.append(buf, (size_t)n);
        }
        s.close();
        if (!ok && !changed()) fail(why);
    }
}

std::string Client::disco_seal(const uint8_t to_disco[32], const std::string &plain) {
    uint8_t pub[32];
    public_key(pub, disco_priv_);
    const std::string sealed = box_seal(disco_priv_, to_disco, plain);
    if (sealed.empty()) return "";
    return std::string(kDiscoMagic, kDiscoMagicLen) + std::string((const char *)pub, 32) + sealed;
}

void Client::ping_locked(const std::string &node_key, double now) {
    auto it = paths_.find(node_key);
    if (it == paths_.end() || udp_ < 0) return;
    Path &pa = it->second;
    pa.pinged = now;
    for (auto p = pa.pings.begin(); p != pa.pings.end();) p = now - p->second.second > 10 ? pa.pings.erase(p) : std::next(p);
    // The path in use, else every address it may have.
    std::vector<uint64_t> to;
    if (pa.best && now - pa.best_at < kTrustDirect) to.push_back(pa.best);
    else to = pa.endpoints;
    uint8_t node_pub[32];
    public_key(node_pub, node_priv_);
    for (uint64_t a : to) {
        uint8_t rnd[32];
        if (!new_private_key(rnd)) return;
        const std::string txid((const char *)rnd, 12);
        std::string plain;
        plain += (char)kDiscoPing;
        plain += '\0';
        plain += txid;
        plain.append((const char *)node_pub, 32);
        const std::string packet = disco_seal((const uint8_t *)pa.disco.data(), plain);
        if (packet.empty()) return;
        const sockaddr_in sa = addr_of(a);
        sendto(udp_, packet.data(), packet.size(), MSG_DONTWAIT, (const sockaddr *)&sa, sizeof sa);
        pa.pings[txid] = {a, now};
    }
}

void Client::disco_in(const std::string &packet, const std::string *relay_from, uint64_t src) {
    const std::string sender = packet.substr(kDiscoMagicLen, 32);
    std::string node;
    for (const auto &[k, pa] : paths_)
        if (pa.disco == sender) {
            node = k;
            break;
        }
    if (node.empty() || (relay_from && *relay_from != node)) return;
    std::string plain;
    if (!box_open(disco_priv_, (const uint8_t *)sender.data(), packet.substr(kDiscoHeader), plain) || plain.size() < 2)
        return;
    Path &pa = paths_[node];
    const double now = now_s();
    switch ((uint8_t)plain[0]) {
        case kDiscoPing: {
            if (plain.size() < 2 + 12) return;
            // The pong: its TxID, and where the ping came from as the PS5 saw
            // it (an IPv4 address as IPv6, the port).
            std::string pong;
            pong += (char)kDiscoPong;
            pong += '\0';
            pong += plain.substr(2, 12);
            uint32_t ip;
            uint16_t port;
            if (relay_from) {
                ip = 0x7f030328;  // 127.3.3.40: "through relay N", as Tailscale writes it
                port = (uint16_t)home_region_;
            } else {
                ip = (uint32_t)(src >> 16);
                port = (uint16_t)src;
            }
            const uint8_t mapped[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, (uint8_t)(ip >> 24), (uint8_t)(ip >> 16),
                                        (uint8_t)(ip >> 8), (uint8_t)ip, (uint8_t)(port >> 8), (uint8_t)port};
            pong.append((const char *)mapped, sizeof mapped);
            const std::string reply = disco_seal((const uint8_t *)pa.disco.data(), pong);
            if (reply.empty()) return;
            if (relay_from) {
                if (outbox_.size() > 1024) outbox_.pop_front();
                outbox_.emplace_back(node, reply);
            } else if (udp_ >= 0) {
                const sockaddr_in sa = addr_of(src);
                sendto(udp_, reply.data(), reply.size(), MSG_DONTWAIT, (const sockaddr *)&sa, sizeof sa);
                // It reached the PS5 from there: that's its way in.
                if (pa.best != src) log("%s: pinged from %s", hex((const uint8_t *)node.data(), 4).c_str(), addr_text(src).c_str());
                by_addr_[src] = node;
                pa.best = src;
                pa.best_at = now;
            }
            break;
        }
        case kDiscoPong: {
            if (plain.size() < 2 + 12 || relay_from) return;
            auto it = pa.pings.find(plain.substr(2, 12));
            if (it == pa.pings.end() || it->second.first != src) return;
            if (pa.best != src || now - pa.best_at >= kTrustDirect)
                log("direct to %s at %s (%.0f ms)", hex((const uint8_t *)node.data(), 4).c_str(), addr_text(src).c_str(),
                    (now - it->second.second) * 1000);
            pa.pings.erase(it);
            by_addr_[src] = node;
            pa.best = src;
            pa.best_at = now;
            break;
        }
        case kDiscoCallMeMaybe: {
            // Its addresses: ping them, so both sides open the way.
            for (size_t i = 2; i + 18 <= plain.size(); i += 18) {
                const uint8_t *e = (const uint8_t *)plain.data() + i;
                static const uint8_t v4[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
                if (memcmp(e, v4, 12) != 0) continue;
                const uint64_t a = ((uint64_t)e[12] << 40) | ((uint64_t)e[13] << 32) | ((uint64_t)e[14] << 24) |
                                   ((uint64_t)e[15] << 16) | ((uint64_t)e[16] << 8) | e[17];
                if (std::find(pa.endpoints.begin(), pa.endpoints.end(), a) == pa.endpoints.end() && pa.endpoints.size() < 16)
                    pa.endpoints.push_back(a);
            }
            ping_locked(node, now);
            break;
        }
        default:
            break;
    }
}

void Client::direct_loop() {
    double last_ip_check = 0, last_tick = 0, retry_at = 0;
    uint16_t port = 0;
    while (!quit_) {
        bool on;
        {
            std::lock_guard<std::mutex> lk(mu_);
            on = cfg_.on && have_disco_ && !map_.address.empty();
            if (!on && udp_ >= 0) {
                close(udp_);
                udp_ = -1;
                endpoints_.clear();
                endpoints_gen_++;
                for (auto &[k, pa] : paths_) pa.best = 0;
                by_addr_.clear();
            }
        }
        if (!on) {
            usleep(200 * 1000);
            continue;
        }
        double now = now_s();
        int fd;
        {
            std::lock_guard<std::mutex> lk(mu_);
            fd = udp_;
        }
        if (fd < 0) {
            if (now < retry_at) {
                usleep(200 * 1000);
                continue;
            }
            fd = socket(AF_INET, SOCK_DGRAM, 0);
            port = 0;
            for (uint16_t p : kDirectPorts) {
                sockaddr_in a{};
                a.sin_family = AF_INET;
                a.sin_port = htons(p);
                if (fd >= 0 && bind(fd, (const sockaddr *)&a, sizeof a) == 0) {
                    sockaddr_in me{};
                    socklen_t len = sizeof me;
                    getsockname(fd, (sockaddr *)&me, &len);
                    port = ntohs(me.sin_port);
                    break;
                }
            }
            if (port == 0) {
                log("no UDP port for direct paths: only the relay");
                if (fd >= 0) close(fd);
                retry_at = now + 60;
                continue;
            }
            // The PS5 ignores O_NONBLOCK from fcntl; FIONBIO takes.
            int one = 1;
            ioctl(fd, FIONBIO, &one);
            log("direct paths on UDP port %u", port);
            std::lock_guard<std::mutex> lk(mu_);
            udp_ = fd;
            last_ip_check = 0;
        }
        // The PS5's own address, now and then: the one devices at home use.
        if (now - last_ip_check > 30) {
            last_ip_check = now;
            const uint32_t ip = local_ip();
            std::vector<std::string> eps;
            if (ip) {
                sockaddr_in a{};
                a.sin_addr.s_addr = ip;
                a.sin_port = htons(port);
                eps.push_back(addr_text(addr_key(a)));
            }
            std::lock_guard<std::mutex> lk(mu_);
            if (eps != endpoints_) {
                endpoints_ = eps;
                endpoints_gen_++;
                if (!eps.empty()) log("direct address %s", eps[0].c_str());
            }
        }
        // What came in.
        bool any = false;
        for (int i = 0; i < 64; i++) {
            char buf[4096];
            sockaddr_in from{};
            socklen_t flen = sizeof from;
            const ssize_t n = recvfrom(fd, buf, sizeof buf, MSG_DONTWAIT, (sockaddr *)&from, &flen);
            if (n <= 0) break;
            any = true;
            const std::string packet(buf, (size_t)n);
            const uint64_t src = addr_key(from);
            std::lock_guard<std::mutex> lk(mu_);
            if (is_disco(packet)) {
                disco_in(packet, nullptr, src);
                continue;
            }
            auto it = by_addr_.find(src);
            if (it == by_addr_.end()) continue;  // from no device the PS5 knows
            Path &pa = paths_[it->second];
            if (pa.best == src) pa.best_at = now;
            if (inbox_.size() > 1024) inbox_.pop_front();
            inbox_.emplace_back(it->second, packet);
        }
        // Devices in use get pinged, to find or keep a direct path.
        now = now_s();
        if (now - last_tick > 0.25) {
            last_tick = now;
            std::lock_guard<std::mutex> lk(mu_);
            std::vector<std::string> due;
            for (const auto &[k, pa] : paths_)
                if (now - pa.wanted < 10 && now - pa.pinged >= kPingEvery && !pa.endpoints.empty()) due.push_back(k);
            for (const std::string &k : due) ping_locked(k, now);
        }
        if (!any) usleep(1000);
    }
}

}  // namespace ts
