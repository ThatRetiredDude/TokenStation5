#include "wg_devices.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "net.h"

namespace app {

namespace {

constexpr const char *kHeader = "# ts5 wireguard v1";
constexpr const char *kColumns = "#id\tname\tpublic_key\taddress\ton\tadded";
constexpr double kRequestFor = 600;  // a request waits this long for the TV

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> out;
    size_t at = 0;
    for (;;) {
        size_t t = line.find('\t', at);
        out.push_back(line.substr(at, t == std::string::npos ? std::string::npos : t - at));
        if (t == std::string::npos) break;
        at = t + 1;
    }
    return out;
}

// A name for one line of the file, and for the TV: printable, at most 40 characters.
std::string clean(const std::string &s) {
    std::string out;
    for (unsigned char c : s) {
        if (out.size() >= 40) break;
        if (c >= 0x20 && c != 0x7f) out += (char)c;
        else if (c == '\t' || c == '\n' || c == '\r') out += ' ';
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    size_t b = out.find_first_not_of(' ');
    return b == std::string::npos ? std::string() : out.substr(b);
}

// A WireGuard public key: 44 characters of base64 for 32 bytes, ending "=".
bool is_key(const std::string &k) {
    if (k.size() != 44 || k[43] != '=') return false;
    for (size_t i = 0; i < 43; i++) {
        const char c = k[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' || c == '/'))
            return false;
    }
    return true;
}

double steady_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

void WgStore::load(const std::string &path) {
    std::lock_guard<std::mutex> lk(mu_);
    path_ = path;
    devices_.clear();
    private_key_.clear();
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return;
    char buf[1024];
    std::vector<std::string> names;
    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '@') {
            const std::vector<std::string> v = split_tabs(line.substr(1));
            if (v.size() == 2 && v[0] == "private_key") private_key_ = v[1];
            continue;
        }
        if (line[0] == '#') {
            if (line.size() > 1 && line[1] != ' ') names = split_tabs(line.substr(1));
            continue;
        }
        const std::vector<std::string> v = split_tabs(line);
        WgDevice d;
        for (size_t i = 0; i < v.size() && i < names.size(); i++) {
            if (names[i] == "id") d.id = v[i];
            else if (names[i] == "name") d.name = v[i];
            else if (names[i] == "public_key") d.public_key = v[i];
            else if (names[i] == "address") d.address = v[i];
            else if (names[i] == "on") d.on = v[i] != "0";
            else if (names[i] == "added") d.added = strtoll(v[i].c_str(), nullptr, 10);
        }
        if (d.id.empty() || !is_key(d.public_key) || d.address.rfind("10.55.0.", 0) != 0) {
            printf("ts5-title: wireguard: a line of %s isn't a device; skipped\n", path.c_str());
            continue;
        }
        devices_.push_back(d);
    }
    fclose(f);
    if (!private_key_.empty() && wg::public_key_of(private_key_).empty()) {
        printf("ts5-title: wireguard: %s has no usable key; a new one is made when WireGuard starts\n", path.c_str());
        private_key_.clear();
    }
    printf("ts5-title: wireguard: %zu devices\n", devices_.size());
}

bool WgStore::save_locked(std::string &err) {
    // Written next to it, then moved over it, as keys.tsv is.
    const std::string tmp = path_ + ".new";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) {
        err = "can't write " + tmp + ": " + strerror(errno);
        return false;
    }
    bool ok = fprintf(f, "%s\n", kHeader) >= 0;
    if (!private_key_.empty()) ok = ok && fprintf(f, "@private_key\t%s\n", private_key_.c_str()) >= 0;
    ok = ok && fprintf(f, "%s\n", kColumns) >= 0;
    for (const WgDevice &d : devices_)
        ok = ok && fprintf(f, "%s\t%s\t%s\t%s\t%d\t%lld\n", d.id.c_str(), clean(d.name).c_str(), d.public_key.c_str(),
                           d.address.c_str(), d.on ? 1 : 0, (long long)d.added) >= 0;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path_.c_str()) != 0) {
        err = "can't write " + path_ + ": " + strerror(errno);
        ::remove(tmp.c_str());
        return false;
    }
    return true;
}

std::string WgStore::public_key(std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (private_key_.empty()) {
        std::string priv, pub;
        if (!wg::make_keys(priv, pub)) {
            err = "no random numbers";
            return "";
        }
        private_key_ = priv;
        if (!save_locked(err)) {
            private_key_.clear();
            return "";
        }
        printf("ts5-title: wireguard: made this PS5's key pair, public key %s\n", pub.c_str());
        return pub;
    }
    return wg::public_key_of(private_key_);
}

bool WgStore::has_key() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !private_key_.empty();
}

bool WgStore::new_key(std::string &err) {
    std::string priv, pub;
    if (!wg::make_keys(priv, pub)) {
        err = "no random numbers";
        return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    const std::string old = private_key_;
    private_key_ = priv;
    if (!save_locked(err)) {
        private_key_ = old;
        return false;
    }
    printf("ts5-title: wireguard: a new key pair for this PS5, public key %s\n", pub.c_str());
    return true;
}

std::vector<WgDevice> WgStore::devices() const {
    std::lock_guard<std::mutex> lk(mu_);
    return devices_;
}

bool WgStore::set_on(const std::string &id, bool on, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    for (WgDevice &d : devices_)
        if (d.id == id) {
            const bool was = d.on;
            d.on = on;
            if (save_locked(err)) {
                printf("ts5-title: wireguard: %s (%s) turned %s\n", d.name.c_str(), d.address.c_str(), on ? "on" : "off");
                return true;
            }
            d.on = was;
            return false;
        }
    err = "no device " + id;
    return false;
}

bool WgStore::remove(const std::string &id, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < devices_.size(); i++)
        if (devices_[i].id == id) {
            const WgDevice gone = devices_[i];
            devices_.erase(devices_.begin() + (long)i);
            if (save_locked(err)) {
                printf("ts5-title: wireguard: removed %s (%s)\n", gone.name.c_str(), gone.address.c_str());
                return true;
            }
            devices_.insert(devices_.begin() + (long)i, gone);
            return false;
        }
    err = "no device " + id;
    return false;
}

std::string WgStore::free_address_locked() const {
    for (int n = 2; n < 255; n++) {
        const std::string a = "10.55.0." + std::to_string(n);
        bool used = has_request_ && request_.address == a;
        for (const WgDevice &d : devices_) used = used || d.address == a;
        if (!used) return a;
    }
    return "";
}

bool WgStore::ask(const std::string &name, const std::string &public_key, const std::string &from, WgRequest &out,
                  std::string &err) {
    std::string key;
    for (char c : public_key)
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') key += c;
    const std::string nm = clean(name);
    if (nm.empty()) {
        err = "Give the device a name, such as Laptop.";
        return false;
    }
    if (!is_key(key) || wg::public_key_of(key).empty()) {
        err = "That isn't a WireGuard public key. It is 44 characters long and ends with =, as the WireGuard app "
              "shows it under Public key.";
        return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (has_request_ && steady_s() - request_.at > kRequestFor) has_request_ = false;
    if (has_request_) {
        err = "Another device is waiting for the TV. Approve or refuse it there first.";
        return false;
    }
    if (!private_key_.empty() && key == wg::public_key_of(private_key_)) {
        err = "That is the PS5's own public key. Paste the key your device's WireGuard app shows.";
        return false;
    }
    if ((int)devices_.size() >= wg::kMaxPeers) {
        err = "This PS5 already has " + std::to_string(wg::kMaxPeers) + " devices. Remove one in Settings first.";
        return false;
    }
    for (const WgDevice &d : devices_)
        if (d.public_key == key) {
            err = "That key is already " + d.name + "'s, at " + d.address + ".";
            return false;
        }
    const std::string address = free_address_locked();
    if (address.empty()) {
        err = "No addresses left. Remove a device in Settings first.";
        return false;
    }
    request_ = {nm, key, address, from, steady_s()};
    has_request_ = true;
    out = request_;
    printf("ts5-title: wireguard: %s asks to join as %s, from %s\n", nm.c_str(), address.c_str(), from.c_str());
    return true;
}

bool WgStore::waiting(WgRequest &out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (has_request_ && steady_s() - request_.at > kRequestFor) {
        printf("ts5-title: wireguard: %s's request lapsed\n", request_.name.c_str());
        has_request_ = false;
    }
    if (has_request_) out = request_;
    return has_request_;
}

WgStore::State WgStore::state_of(const std::string &public_key) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const WgDevice &d : devices_)
        if (d.public_key == public_key) return d.on ? State::In : State::Off;
    if (has_request_ && steady_s() - request_.at > kRequestFor) has_request_ = false;
    if (has_request_ && request_.public_key == public_key) return State::Waiting;
    return State::None;
}

bool WgStore::answer(bool yes, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!has_request_) return true;
    const WgRequest r = request_;
    has_request_ = false;
    if (!yes) {
        printf("ts5-title: wireguard: refused %s\n", r.name.c_str());
        return true;
    }
    unsigned char b[4];
    if (!ts5::random_bytes(b, sizeof b)) {
        err = "no random numbers";
        return false;
    }
    char id[9];
    snprintf(id, sizeof id, "%02x%02x%02x%02x", b[0], b[1], b[2], b[3]);
    WgDevice d;
    d.id = id;
    d.name = r.name;
    d.public_key = r.public_key;
    d.address = r.address;
    d.added = (int64_t)time(nullptr);
    devices_.push_back(d);
    if (!save_locked(err)) {
        devices_.pop_back();
        return false;
    }
    printf("ts5-title: wireguard: added %s as %s\n", d.name.c_str(), d.address.c_str());
    return true;
}

wg::Config WgStore::config(bool on, int port, int api_port) const {
    std::lock_guard<std::mutex> lk(mu_);
    wg::Config c;
    c.on = on && !private_key_.empty();
    c.private_key = private_key_;
    c.port = port;
    c.address = kAddress;
    c.api_port = api_port;
    for (const WgDevice &d : devices_)
        if (d.on) c.peers.push_back({d.name, d.public_key, d.address, "", 0});
    return c;
}

}  // namespace app
