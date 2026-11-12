#include "tailnet_serve.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

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

bool from_here(const std::string &client) {
    return client == "127.0.0.1" || client == "::1" || client == "::ffff:127.0.0.1";
}

}  // namespace

std::string tailnet_error(const std::string &why) { return "{\"error\":" + json_escape(why) + "}"; }

struct Tailnet::Impl {
    ts::Client client;
    wg::Tunnel tunnel;
    bool attached = false;
};

Tailnet::Tailnet() : d_(new Impl) { d_->tunnel.use_tailscale(&d_->client); }

Tailnet::~Tailnet() = default;

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
        }
        fclose(f);
    }
    if (on_) log_line("ts5: tailscale: on, as %s", name_.c_str());
    apply_locked();
}

void Tailnet::apply_locked() {
    ts::Config tc;
    tc.on = on_;
    tc.control_url = server_;
    tc.hostname = name_;
    tc.auth_key = env_or("TS5_TS_AUTH_KEY", "");
    tc.state_file = dir_ + "/tailscale.tsv";
    tc.version = TS5_VERSION;
    d_->client.apply(tc);
    wg::Config wc;  // WireGuard's own server stays off: the tunnel carries the tailnet only
    wc.api_port = api_port_;
    wc.tailscale = on_;
    std::string err;
    if (!d_->tunnel.apply(wc, err)) log_line("ts5: tailscale: %s", err.c_str());
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
                             std::string(on_ ? "1" : "0") + "\nname=" + name_ + "\nserver=" + server_ + "\n";
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
    bool on;
    std::string name, server;
    int port;
    {
        std::lock_guard<std::mutex> lk(mu_);
        on = on_;
        name = name_;
        server = server_;
        port = api_port_;
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
    if (st.state == ts::Status::Connected) {
        // The address by number: a name needs the tailnet's MagicDNS (and
        // the phone's "Use Tailscale DNS"), which not every tailnet has.
        std::string host = st.name;
        while (!host.empty() && host.back() == '.') host.pop_back();
        const std::string p = ":" + std::to_string(port) + "/v1";
        j.set("url", Json::string("http://" + (st.address.empty() ? host : st.address) + p));
        if (!host.empty() && !st.address.empty()) j.set("name_url", Json::string("http://" + host + p));
    }
    Json peers;
    peers.type = Json::ARR;
    for (const wg::PeerState &ps : d_->tunnel.tailnet_peers()) {
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
        reply = tailnet_error("the body is a JSON object: {\"on\": true}, {\"name\": \"NAME\"} or {\"sign_out\": true}");
        return true;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (dir_.empty()) {
            code = 400;
            reply = tailnet_error("this ts5 serve has no keys file (TS5_KEYS), so it keeps no tailnet");
            return true;
        }
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
        std::string err;
        if (!save_locked(err)) log_line("ts5: tailscale: %s", err.c_str());
        log_line("ts5: tailscale: %s%s, as %s", sign_out ? "signing out, " : "", on_ ? "on" : "off", name_.c_str());
        apply_locked();
        if (sign_out) d_->client.sign_out();
    }
    code = 200;
    reply = status_json();
    return true;
}

}  // namespace ts5
