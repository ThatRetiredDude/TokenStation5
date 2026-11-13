#include "tailnet_serve.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

#include "acme.h"
#include "https_front.h"
#include "json.h"
#include "log.h"
#include "tailnet/tailscale.h"
#include "tailnet/wg.h"

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif

namespace ts5 {

namespace {

constexpr const char *kDefaultServer = "https://controlplane.tailscale.com";
constexpr const char *kDefaultName = "tokenstation5";

const char *state_name(ts::Status::State s) {
    switch (s) {
    case ts::Status::Off: return "off";
    case ts::Status::Starting: return "starting";
    case ts::Status::NeedsSignIn: return "sign_in";
    case ts::Status::Connecting: return "connecting";
    case ts::Status::Connected: return "connected";
    case ts::Status::Problem: return "problem";
    }
    return "off";
}

// A name the tailnet takes as it is: 1 to 63 of a-z, 0-9 and '-', not
// starting or ending with '-'.
bool good_name(const std::string &n) {
    if (n.empty() || n.size() > 63 || n.front() == '-' || n.back() == '-') return false;
    for (char c : n)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return false;
    return true;
}

std::string env_or(const char *name, const std::string &def) {
    const char *v = getenv(name);
    return v && *v ? std::string(v) : def;
}

Json jbool(bool v) {
    Json j;
    j.type = Json::BOOL;
    j.b = v;
    return j;
}

Json jnum(int v) {
    Json j;
    j.type = Json::NUM;
    j.n = v;
    j.s = std::to_string(v);
    return j;
}

// Whether something listens on TCP port 22 here (this machine's SSH), from
// /proc/net/tcp and tcp6: no connection made, so nothing in sshd's log.
bool ssh_listening() {
    for (const char *path : {"/proc/net/tcp", "/proc/net/tcp6"}) {
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char line[512];
        bool found = false;
        while (!found && fgets(line, sizeof line, f)) {
            // "  0: 00000000:0016 00000000:0000 0A ...": local address:port, remote, state (0A = listening).
            char local[128], remote[128];
            unsigned st = 0;
            if (sscanf(line, " %*d: %127s %127s %x", local, remote, &st) != 3) continue;
            const char *colon = strrchr(local, ':');
            found = st == 0x0A && colon && strtoul(colon + 1, nullptr, 16) == 22;
        }
        fclose(f);
        if (found) return true;
    }
    return false;
}

bool from_here(const std::string &client) {
    return client == "127.0.0.1" || client == "::1" || client == "::ffff:127.0.0.1";
}

std::string read_file(const std::string &path) {
    std::string s;
    if (FILE *f = fopen(path.c_str(), "rb")) {
        char b[4096];
        size_t n;
        while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
        fclose(f);
    }
    return s;
}

// A file only its owner reads, written whole or not at all.
bool write_secret(const std::string &path, const std::string &text) {
    const std::string tmp = path + ".new";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    fchmod(fd, 0600);
    const bool ok = write(fd, text.data(), text.size()) == (ssize_t)text.size() && fsync(fd) == 0;
    close(fd);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

std::string utc_day(long long t) {
    const time_t tt = (time_t)t;
    struct tm tm {};
    gmtime_r(&tt, &tm);
    char b[32];
    strftime(b, sizeof b, "%Y-%m-%d", &tm);
    return b;
}

}  // namespace

std::string tailnet_error(const std::string &why) { return "{\"error\":" + json_escape(why) + "}"; }

struct Tailnet::Impl {
    ts::Client client;
    wg::Tunnel tunnel;
    bool attached = false;
    // HTTPS (under Tailnet::mu_): the front's port on 127.0.0.1, whether it
    // has a certificate (the tunnel's 443 opens then), and what to show.
    HttpsFront front;
    int front_port = 0;
    bool https_live = false;
    std::string https_state = "off", https_name, https_problem;
    long long https_until = 0;
    std::thread https_thread;
    std::condition_variable https_cv;
    bool https_kick = false, https_quit = false;
};

Tailnet::Tailnet() : d_(new Impl) { d_->tunnel.use_tailscale(&d_->client); }

Tailnet::~Tailnet() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        d_->https_quit = true;
    }
    d_->https_cv.notify_all();
    // (A certificate being fetched finishes first: at most a few minutes.)
    if (d_->https_thread.joinable()) d_->https_thread.join();
}

void Tailnet::start(const std::string &dir, int api_port) {
    std::lock_guard<std::mutex> lk(mu_);
    dir_ = dir;
    api_port_ = api_port;
    name_ = env_or("TS5_TS_NAME", kDefaultName);
    server_ = env_or("TS5_TS_SERVER", kDefaultServer);
    if (FILE *f = fopen((dir_ + "/tailscale.conf").c_str(), "r")) {
        char line[512];
        while (fgets(line, sizeof line, f)) {
            std::string l = line;
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r' || l.back() == ' ')) l.pop_back();
            const size_t eq = l.find('=');
            if (l.empty() || l[0] == '#' || eq == std::string::npos) continue;
            const std::string k = l.substr(0, eq), v = l.substr(eq + 1);
            if (k == "on") on_ = v == "1";
            else if (k == "name" && good_name(v)) name_ = v;
            else if (k == "server" && !v.empty()) server_ = v;
            else if (k == "ssh") ssh_ = v == "1";
            else if (k == "terminal") terminal_ = v == "1";
        }
        fclose(f);
    }
    if (on_) log_line("ts5: tailscale: on, as %s%s", name_.c_str(), ssh_ ? ", with SSH" : "");
    // HTTPS's front, on a free port here; the tunnel's 443 opens once it has
    // a certificate (https_loop).
    // Tests: TS5_HTTPS_TEST=CERT:KEY:PORT:DEVICE runs the front at once with
    // that certificate on 127.0.0.1:PORT, every connection to it named DEVICE
    // (as the tunnel names a tailnet device's), and gets no certificate.
    const std::string test = env_or("TS5_HTTPS_TEST", "");
    if (!test.empty()) {
        std::vector<std::string> f;
        for (size_t at = 0;;) {
            const size_t c = test.find(':', at);
            f.push_back(test.substr(at, c == std::string::npos ? std::string::npos : c - at));
            if (c == std::string::npos) break;
            at = c + 1;
        }
        std::string why;
        const std::string device = f.size() > 3 ? f[3] : "";
        if (f.size() == 4 &&
            (d_->front_port = d_->front.start(
                 api_port_, [device](uint16_t) { return device; },
                 [this](uint16_t port, const std::string &name) { d_->tunnel.note_client(port, name); }, why,
                 atoi(f[2].c_str()))) &&
            d_->front.set_certificate(read_file(f[0]), read_file(f[1]), why)) {
            d_->https_live = true;
            d_->https_state = "on";
            log_line("ts5: https: test front on 127.0.0.1:%d", d_->front_port);
        } else {
            log_line("ts5: https: TS5_HTTPS_TEST: %s", why.empty() ? "CERT:KEY:PORT:DEVICE" : why.c_str());
        }
    } else if (env_or("TS5_TS_HTTPS", "1") != "0") {
        std::string why;
        d_->front_port = d_->front.start(
            api_port_, [this](uint16_t port) { return d_->tunnel.client_for(port); },
            [this](uint16_t port, const std::string &name) { d_->tunnel.note_client(port, name); }, why);
        if (!d_->front_port) log_line("ts5: https: %s", why.c_str());
        else d_->https_thread = std::thread([this] { https_loop(); });
    }
    apply_locked(true);
}

int https_test_main(int argc, char **argv) {
    // ts5 https-test cert DOMAIN DIR: a certificate from TS5_ACME_DIRECTORY
    // (never Let's Encrypt's own), the DNS record written to
    // TS5_ACME_TXT_FILE as "NAME VALUE" for the test's stand-in to read.
    if (argc != 5 || std::string(argv[2]) != "cert") {
        fprintf(stderr, "usage: ts5 https-test cert DOMAIN DIR (tests: TS5_ACME_DIRECTORY, TS5_ACME_TXT_FILE)\n");
        return 2;
    }
    const char *dir = getenv("TS5_ACME_DIRECTORY"), *txt = getenv("TS5_ACME_TXT_FILE");
    if (!dir || !*dir || !txt || !*txt) {
        fprintf(stderr, "ts5: https-test needs TS5_ACME_DIRECTORY and TS5_ACME_TXT_FILE\n");
        return 2;
    }
    const std::string domain = argv[3], out = argv[4];
    AcmeOptions o;
    o.directory = dir;
    o.dns_wait_s = 0;
    o.account_key = read_file(out + "/account.key");
    std::string cert, key, why;
    const std::string txt_path = txt;
    const bool ok = acme_certificate(
        o, domain,
        [&](const std::string &name, const std::string &value, std::string &w) {
            if (!write_secret(txt_path, name + " " + value + "\n")) {
                w = "can't write " + txt_path;
                return false;
            }
            return true;
        },
        cert, key, why);
    if (!write_secret(out + "/account.key", o.account_key)) fprintf(stderr, "ts5: can't write the account key\n");
    if (!ok) {
        fprintf(stderr, "ts5: https: %s\n", why.c_str());
        return 1;
    }
    bool names_ok = false;
    const long long until = cert_expiry(cert, domain, &names_ok);
    if (!write_secret(out + "/cert.pem", cert) || !write_secret(out + "/key.pem", key)) {
        fprintf(stderr, "ts5: can't write in %s\n", out.c_str());
        return 1;
    }
    printf("until %lld names_ok %d\n", until, names_ok ? 1 : 0);
    return 0;
}

void Tailnet::https_set(const std::string &state, const std::string &name, long long until, const std::string &problem) {
    std::lock_guard<std::mutex> lk(mu_);
    d_->https_state = state;
    d_->https_name = name;
    d_->https_until = until;
    d_->https_problem = problem;
}

// Gets the tailnet name's certificate, keeps it, renews it, and opens the
// tunnel's port 443 once the front has one.
void Tailnet::https_loop() {
    const std::string crt_path = dir_ + "/tailscale-https.crt", key_path = dir_ + "/tailscale-https.key",
                      acct_path = dir_ + "/tailscale-acme.key";
    constexpr long long kRenew = 30LL * 86400;
    std::string loaded;  // the name the front's certificate is for
    long long loaded_until = 0, next_try = 0;
    int fails = 0;
    while (true) {
        bool on;
        {
            std::unique_lock<std::mutex> lk(mu_);
            d_->https_cv.wait_for(lk, std::chrono::seconds(10), [this] { return d_->https_kick || d_->https_quit; });
            if (d_->https_quit) return;
            d_->https_kick = false;
            on = on_;
        }
        if (!on) {
            https_set("off");
            continue;
        }
        const ts::Status st = d_->client.status();
        std::string host = st.name;
        while (!host.empty() && host.back() == '.') host.pop_back();
        if (st.state != ts::Status::Connected || host.empty()) {
            if (loaded.empty()) https_set("waiting");
            continue;
        }
        if (st.cert_domains.empty()) {
            // The tailnet's admin hasn't turned on HTTPS Certificates (or MagicDNS).
            https_set("admin", host);
            continue;
        }
        std::string domain = st.cert_domains[0];
        for (const std::string &c : st.cert_domains)
            if (c == host) domain = c;
        const long long now = (long long)time(nullptr);
        if (loaded == domain && loaded_until - now > kRenew) {
            https_set("on", domain, loaded_until);
            continue;
        }
        // The one kept from before, while it is good.
        const std::string crt = read_file(crt_path), key = read_file(key_path);
        bool names_ok = false;
        const long long until = crt.empty() ? 0 : cert_expiry(crt, domain, &names_ok);
        if (names_ok && until > now + 3600 && (loaded != domain || loaded_until != until)) {
            std::string why;
            if (d_->front.set_certificate(crt, key, why)) {
                loaded = domain;
                loaded_until = until;
                std::lock_guard<std::mutex> lk(mu_);
                if (!d_->https_live) {
                    d_->https_live = true;
                    apply_locked(false);
                }
                log_line("ts5: https: on, as https://%s/ (its certificate until %s)", domain.c_str(), utc_day(until).c_str());
            } else {
                log_line("ts5: https: the kept certificate can't be used: %s", why.c_str());
            }
        }
        if (loaded == domain && loaded_until - now > kRenew) {
            https_set("on", domain, loaded_until);
            continue;
        }
        if (now < next_try) continue;
        https_set(loaded == domain ? "on" : "getting", domain, loaded == domain ? loaded_until : 0);
        AcmeOptions o;
        o.directory = env_or("TS5_ACME_DIRECTORY", o.directory);
        o.dns_wait_s = atoi(env_or("TS5_ACME_DNS_WAIT", "60").c_str());
        o.account_key = read_file(acct_path);
        const std::string had_account = o.account_key;
        log_line("ts5: https: %s a certificate for %s from %s", loaded == domain ? "renewing" : "getting", domain.c_str(),
                 o.directory.c_str());
        std::string cert_pem, key_pem, why;
        const bool ok = acme_certificate(
            o, domain,
            [this](const std::string &name, const std::string &value, std::string &w) {
                return d_->client.set_dns(name, "TXT", value, w);
            },
            cert_pem, key_pem, why);
        if (o.account_key != had_account && !write_secret(acct_path, o.account_key))
            log_line("ts5: https: can't write %s", acct_path.c_str());
        long long got_until = 0;
        if (ok) {
            got_until = cert_expiry(cert_pem, domain, &names_ok);
            if (!got_until || !names_ok) why = "the certificate that came isn't for " + domain;
            else if (!write_secret(key_path, key_pem) || !write_secret(crt_path, cert_pem))
                log_line("ts5: https: can't keep the certificate in %s (it is used until ts5 serve stops)", dir_.c_str());
        }
        if (got_until && names_ok && d_->front.set_certificate(cert_pem, key_pem, why)) {
            loaded = domain;
            loaded_until = got_until;
            fails = 0;
            next_try = 0;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!d_->https_live) {
                    d_->https_live = true;
                    apply_locked(false);
                }
            }
            log_line("ts5: https: on, as https://%s/ (a new certificate, until %s)", domain.c_str(), utc_day(got_until).c_str());
            https_set("on", domain, got_until);
            continue;
        }
        // Let's Encrypt allows 5 failed checks an hour for a name: twice in 15
        // minutes, then once an hour.
        fails++;
        const int wait_min = fails <= 2 ? 15 : 60;
        next_try = now + wait_min * 60;
        log_line("ts5: https: no certificate for %s: %s (trying again in %d min)", domain.c_str(), why.c_str(), wait_min);
        if (loaded == domain) https_set("on", domain, loaded_until, why);
        else https_set("problem", domain, 0, why);
    }
}

void Tailnet::apply_locked(bool client) {
    // The client only when its settings changed: applying it starts its
    // connections over, which an SSH switch has no need of.
    if (client) {
        ts::Config tc;
        tc.on = on_;
        tc.control_url = server_;
        tc.hostname = name_;
        tc.auth_key = env_or("TS5_TS_AUTH_KEY", "");
        tc.state_file = dir_ + "/tailscale.tsv";
        tc.version = TS5_VERSION;
        d_->client.apply(tc);
    }
    wg::Config wc;  // WireGuard's own server stays off: the tunnel carries the tailnet only
    wc.api_port = api_port_;
    wc.tailscale = on_;
    wc.ssh = on_ && ssh_;
    wc.https_port = d_->https_live ? d_->front_port : 0;
    std::string err;
    if (!d_->tunnel.apply(wc, err)) log_line("ts5: tailscale: %s", err.c_str());
    d_->https_kick = true;
    d_->https_cv.notify_all();
}

bool Tailnet::save_locked(std::string &err) {
    const std::string path = dir_ + "/tailscale.conf", tmp = path + ".new";
    mkdir(dir_.c_str(), 0700);
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        err = "can't write " + path;
        return false;
    }
    const std::string text = "# ts5 tailscale settings (ts5-tui's Access page writes them)\non=" +
                             std::string(on_ ? "1" : "0") + "\nname=" + name_ + "\nserver=" + server_ +
                             "\nssh=" + std::string(ssh_ ? "1" : "0") + "\nterminal=" + std::string(terminal_ ? "1" : "0") +
                             "\n";
    const bool ok = write(fd, text.data(), text.size()) == (ssize_t)text.size() && fsync(fd) == 0;
    close(fd);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        err = "can't write " + path;
        return false;
    }
    return true;
}

void Tailnet::attach(Server &s) {
    d_->attached = true;
    // Connections through the tunnel reach the API from 127.0.0.1: each is
    // named by the device behind it (its 100.x address), which needs a key.
    s.set_client_name([this](const std::string &ip, int port) {
        if (ip == "127.0.0.1") {
            const std::string device = d_->tunnel.client_for((uint16_t)port);
            if (!device.empty()) return device;
        }
        return ip;
    });
    s.set_routes([this](const std::string &method, const std::string &path, const std::string &body,
                        const std::string &client, int &code, std::string &reply) {
        return route(method, path, body, client, code, reply);
    });
}

std::string Tailnet::status_json() {
    bool on, ssh, terminal, https_live;
    std::string name, server, https_state, https_name, https_problem;
    long long https_until;
    int port;
    {
        std::lock_guard<std::mutex> lk(mu_);
        on = on_;
        ssh = ssh_;
        terminal = terminal_;
        name = name_;
        server = server_;
        port = api_port_;
        https_live = d_->https_live;
        https_state = d_->front_port ? d_->https_state : "off";
        https_name = d_->https_name;
        https_problem = d_->https_problem;
        https_until = d_->https_until;
    }
    const ts::Status st = d_->client.status();
    Json j = Json::object();
    j.set("on", jbool(on));
    // Just turned on, before the client's thread has begun: starting.
    j.set("state", Json::string(!on ? "off" : st.state == ts::Status::Off ? "starting" : state_name(st.state)));
    j.set("hostname", Json::string(name));
    j.set("server", Json::string(server));
    if (!st.sign_in_url.empty()) j.set("sign_in_url", Json::string(st.sign_in_url));
    if (!st.problem.empty()) j.set("problem", Json::string(st.problem));
    if (!st.name.empty()) j.set("name", Json::string(st.name));
    if (!st.address.empty()) j.set("address", Json::string(st.address));
    if (!st.account.empty()) j.set("account", Json::string(st.account));
    if (!st.relay.empty()) j.set("relay", Json::string(st.relay));
    j.set("devices", jnum(st.devices));
    j.set("port", jnum(port));
    // SSH through the tailnet: on, and whether this machine has an SSH
    // server for it to reach.
    j.set("ssh", jbool(ssh));
    j.set("ssh_server", jbool(ssh_listening()));
    // The browser terminal (web_serve.h): ts5-tui in a web page, for the
    // tailnet's devices with a key.
    j.set("terminal", jbool(terminal));
    if (st.state == ts::Status::Connected) {
        // The address by number: a name needs the tailnet's MagicDNS (and
        // the phone's "Use Tailscale DNS"), which not every tailnet has.
        std::string host = st.name;
        while (!host.empty() && host.back() == '.') host.pop_back();
        const std::string p = ":" + std::to_string(port) + "/v1";
        j.set("url", Json::string("http://" + (st.address.empty() ? host : st.address) + p));
        if (!host.empty() && !st.address.empty()) j.set("name_url", Json::string("http://" + host + p));
        // HTTPS by its name (the certificate's), port 443: what a phone's
        // browser treats as a secure page.
        if (https_live && https_state == "on" && !https_name.empty())
            j.set("https_url", Json::string("https://" + https_name + "/"));
    }
    {
        Json h = Json::object();
        h.set("state", Json::string(on ? https_state : "off"));
        if (!https_name.empty()) h.set("name", Json::string(https_name));
        if (https_until) {
            Json u;
            u.type = Json::NUM;
            u.n = (double)https_until;
            u.s = std::to_string(https_until);
            h.set("until", u);
        }
        if (!https_problem.empty()) h.set("problem", Json::string(https_problem));
        j.set("https", h);
    }
    Json peers;
    peers.type = Json::ARR;
    // Off (just signed out, say): the tunnel may not have let its devices
    // go yet, and they aren't this machine's any more.
    for (const wg::PeerState &ps : on ? d_->tunnel.tailnet_peers() : std::vector<wg::PeerState>{}) {
        Json o = Json::object();
        o.set("name", Json::string(ps.name));
        o.set("address", Json::string(ps.address));
        o.set("up", jbool(ps.up));
        o.set("path", Json::string(ps.from));
        peers.a.push_back(o);
    }
    j.set("peers", peers);
    return j.dump();
}

bool Tailnet::route(const std::string &method, const std::string &path, const std::string &body,
                    const std::string &client, int &code, std::string &reply) {
    if (path != "/ts5/tailnet") return false;
    // Only this machine: not another one on the network, not a device on
    // the tailnet itself (those come as 100.x, set_client_name above).
    if (!from_here(client)) {
        code = 403;
        reply = tailnet_error("only ts5 on this machine can change its Tailscale (ts5-tui's Access page)");
        return true;
    }
    if (method == "GET") {
        code = 200;
        reply = status_json();
        return true;
    }
    if (method != "POST") {
        code = 400;
        reply = tailnet_error("GET or POST");
        return true;
    }
    Json j;
    if (!Json::parse(body, j) || j.type != Json::OBJ) {
        code = 400;
        reply = tailnet_error(
            "the body is a JSON object: {\"on\": true}, {\"name\": \"NAME\"}, {\"ssh\": true} or {\"sign_out\": true}");
        return true;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (dir_.empty()) {
            code = 400;
            reply = tailnet_error("this ts5 serve has no keys file (TS5_KEYS), so it keeps no tailnet");
            return true;
        }
        const bool was_on = on_;
        const std::string was_name = name_, was_server = server_;
        if (const Json &n = j["name"]; n.type == Json::STR) {
            std::string v = n.s;
            for (char &c : v)
                if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
            if (!good_name(v)) {
                code = 400;
                reply = tailnet_error("a name is 1 to 63 letters, digits and '-' (not first or last)");
                return true;
            }
            name_ = v;
        }
        if (const Json &s = j["server"]; s.type == Json::STR) server_ = s.s.empty() ? kDefaultServer : s.s;
        const bool sign_out = j["sign_out"].boolean(false);
        if (sign_out) on_ = false;
        else if (j["on"].type == Json::BOOL) on_ = j["on"].b;
        if (j["ssh"].type == Json::BOOL) ssh_ = j["ssh"].b;
        if (j["terminal"].type == Json::BOOL && j["terminal"].b != terminal_) {
            terminal_ = j["terminal"].b;
            log_line("ts5: tailscale: the browser terminal %s", terminal_ ? "on" : "off");
        }
        // Asked to be on again while on: a fresh try (after a problem, say).
        const bool client = sign_out || on_ != was_on || name_ != was_name || server_ != was_server ||
                            (on_ && j["on"].type == Json::BOOL);
        std::string err;
        if (!save_locked(err)) log_line("ts5: tailscale: %s", err.c_str());
        if (client)
            log_line("ts5: tailscale: %s%s, as %s%s", sign_out ? "signing out, " : "", on_ ? "on" : "off",
                     name_.c_str(), on_ && ssh_ ? ", with SSH" : "");
        else if (!on_ && j["ssh"].type == Json::BOOL)
            log_line("ts5: tailscale: SSH %s for when Tailscale is on", ssh_ ? "on" : "off");
        apply_locked(client);
        if (sign_out) d_->client.sign_out();
    }
    code = 200;
    reply = status_json();
    return true;
}

}  // namespace ts5
