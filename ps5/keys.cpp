#include "keys.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "net.h"

namespace app {

namespace {

constexpr const char *kHeader = "# ts5 keys v1";
constexpr const char *kColumns = "#id\tname\tsha256\ton\tcreated\tused";
// Crockford's base32 in lower case: no i, l, o or u, so a key read off the
// TV is never typed wrong between 1 and l or 0 and o.
constexpr const char *kAlphabet = "0123456789abcdefghjkmnpqrstvwxyz";

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

// A name for one line of the file: tabs and line ends become spaces.
std::string clean(const std::string &s) {
    std::string out;
    for (char c : s) out += (c == '\t' || c == '\n' || c == '\r') ? ' ' : c;
    return out;
}

// Equal without stopping at the first difference.
bool same(const std::string &a, const std::string &b) {
    if (a.size() != b.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < a.size(); i++) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}

}  // namespace

void KeyStore::load(const std::string &path) {
    std::lock_guard<std::mutex> lk(mu_);
    path_ = path;
    keys_.clear();
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return;
    char buf[1024];
    std::vector<std::string> names;
    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.size() > 1 && line[1] != ' ') names = split_tabs(line.substr(1));
            continue;
        }
        const std::vector<std::string> v = split_tabs(line);
        ApiKey k;
        for (size_t i = 0; i < v.size() && i < names.size(); i++) {
            if (names[i] == "id") k.id = v[i];
            else if (names[i] == "name") k.name = v[i];
            else if (names[i] == "sha256") k.sha256 = v[i];
            else if (names[i] == "on") k.on = v[i] != "0";
            else if (names[i] == "created") k.created = strtoll(v[i].c_str(), nullptr, 10);
            else if (names[i] == "used") k.used = strtoll(v[i].c_str(), nullptr, 10);
        }
        if (k.id.empty() || k.sha256.size() != 64) {
            printf("ts5-title: keys: a line of %s isn't a key; skipped\n", path.c_str());
            continue;
        }
        keys_.push_back(k);
    }
    fclose(f);
    printf("ts5-title: keys: %zu\n", keys_.size());
}

std::vector<ApiKey> KeyStore::list() const {
    std::lock_guard<std::mutex> lk(mu_);
    return keys_;
}

bool KeyStore::save_locked(std::string &err) {
    // Written next to it, then moved over it, as ts5.env is.
    const std::string tmp = path_ + ".new";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) {
        err = "can't write " + tmp + ": " + strerror(errno);
        return false;
    }
    bool ok = fprintf(f, "%s\n%s\n", kHeader, kColumns) >= 0;
    for (const ApiKey &k : keys_)
        ok = ok && fprintf(f, "%s\t%s\t%s\t%d\t%lld\t%lld\n", k.id.c_str(), clean(k.name).c_str(), k.sha256.c_str(),
                           k.on ? 1 : 0, (long long)k.created, (long long)k.used) >= 0;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path_.c_str()) != 0) {
        err = "can't write " + path_ + ": " + strerror(errno);
        ::remove(tmp.c_str());
        return false;
    }
    used_dirty_ = false;
    return true;
}

std::string KeyStore::create(const std::string &name, std::string &id, std::string &err) {
    // 20 characters of 5 bits from 13 random bytes (100 bits used), and 4
    // bytes for the id.
    unsigned char r[17];
    if (!ts5::random_bytes(r, sizeof r)) {
        err = "no random numbers";
        return "";
    }
    std::string key = "ts5";
    uint32_t acc = 0;
    int bits = 0, n = 0;
    for (int i = 0; i < 13 && n < 20; i++) {
        acc = (acc << 8) | r[i];
        bits += 8;
        while (bits >= 5 && n < 20) {
            if (n % 5 == 0) key += '-';
            key += kAlphabet[(acc >> (bits - 5)) & 31];
            bits -= 5;
            n++;
        }
    }
    char idbuf[9];
    snprintf(idbuf, sizeof idbuf, "%02x%02x%02x%02x", r[13], r[14], r[15], r[16]);
    ApiKey k;
    k.id = idbuf;
    k.name = clean(name);
    k.sha256 = ts5::sha256_hex(key);
    k.created = (int64_t)time(nullptr);
    if (k.sha256.size() != 64) {
        err = "no SHA-256 in this build";
        return "";
    }
    std::lock_guard<std::mutex> lk(mu_);
    keys_.push_back(k);
    if (!save_locked(err)) {
        keys_.pop_back();
        return "";
    }
    id = k.id;
    printf("ts5-title: keys: made %s (%s)\n", k.id.c_str(), k.name.c_str());
    return key;
}

bool KeyStore::set_on(const std::string &id, bool on, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    for (ApiKey &k : keys_)
        if (k.id == id) {
            const bool was = k.on;
            k.on = on;
            if (save_locked(err)) {
                printf("ts5-title: keys: %s turned %s\n", id.c_str(), on ? "on" : "off");
                return true;
            }
            k.on = was;
            return false;
        }
    err = "no key " + id;
    return false;
}

bool KeyStore::remove(const std::string &id, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < keys_.size(); i++)
        if (keys_[i].id == id) {
            const ApiKey gone = keys_[i];
            keys_.erase(keys_.begin() + (long)i);
            if (save_locked(err)) {
                printf("ts5-title: keys: removed %s\n", id.c_str());
                return true;
            }
            keys_.insert(keys_.begin() + (long)i, gone);
            return false;
        }
    err = "no key " + id;
    return false;
}

bool KeyStore::authorize(const std::string &key, const std::string &client, std::string &name) {
    name.clear();
    if (!key.empty()) {
        const std::string h = ts5::sha256_hex(key);
        std::lock_guard<std::mutex> lk(mu_);
        for (ApiKey &k : keys_)
            if (k.on && same(k.sha256, h)) {
                name = k.name;
                const int64_t now = (int64_t)time(nullptr);
                if (now - k.used >= 60) {
                    k.used = now;
                    used_dirty_ = true;
                }
                return true;
            }
    }
    // The title's own Chat page comes from the console itself.
    return !required_ || client == "127.0.0.1";
}

void KeyStore::save_used(int64_t now) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!used_dirty_ || now - used_saved_ < 60) return;
    used_saved_ = now;
    std::string err;
    if (!save_locked(err)) printf("ts5-title: keys: %s\n", err.c_str());
}

}  // namespace app
