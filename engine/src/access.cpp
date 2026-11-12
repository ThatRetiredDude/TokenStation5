#include "access.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.h"
#include "net.h"

namespace ts5 {

namespace {

constexpr const char *kHeader = "# ts5 keys v1";
constexpr const char *kRequired = "# required=";
constexpr const char *kColumns = "#id\tname\tsha256\ton\tcreated\tused";
// Crockford's base32 in lower case, as the TV app's keys: no i, l, o or u.
constexpr const char *kAlphabet = "0123456789abcdefghjkmnpqrstvwxyz";

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> out;
    size_t at = 0;
    for (;;) {
        const size_t t = line.find('\t', at);
        out.push_back(line.substr(at, t == std::string::npos ? std::string::npos : t - at));
        if (t == std::string::npos) break;
        at = t + 1;
    }
    return out;
}

// A name for one line of the file: tabs, line ends and other control
// characters become spaces, several spaces one, and it is at most 60 bytes.
std::string clean(const std::string &s) {
    std::string out;
    for (unsigned char c : s) {
        const char x = c < 0x20 || c == 0x7f ? ' ' : (char)c;
        if (x != ' ' || out.empty() || out.back() != ' ') out += x;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    size_t from = 0;
    while (from < out.size() && out[from] == ' ') from++;
    out.erase(0, from);
    if (out.size() > 60) {
        out.resize(60);
        while (!out.empty() && ((unsigned char)out.back() & 0xC0) == 0x80) out.pop_back();
        if (!out.empty() && ((unsigned char)out.back() & 0x80)) out.pop_back();
    }
    return out;
}

// Equal without stopping at the first difference.
bool same(const std::string &a, const std::string &b) {
    if (a.size() != b.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < a.size(); i++) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}

double mono_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A file written beside its place and moved over it. mode: for a new file.
bool write_file(const std::string &path, const std::string &text, mode_t mode, std::string &err) {
    struct stat st {};
    if (stat(path.c_str(), &st) == 0) mode = st.st_mode & 0777;
    const std::string tmp = path + ".new";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
    if (fd < 0) {
        err = "can't write " + tmp + ": " + strerror(errno);
        return false;
    }
    fchmod(fd, mode);
    size_t at = 0;
    bool ok = true;
    while (ok && at < text.size()) {
        const ssize_t n = write(fd, text.data() + at, text.size() - at);
        if (n <= 0) ok = false;
        else at += (size_t)n;
    }
    ok = close(fd) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        err = "can't write " + path + ": " + strerror(errno);
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

void make_parents(const std::string &path) {
    for (size_t i = 1; i < path.size(); i++)
        if (path[i] == '/') mkdir(path.substr(0, i).c_str(), 0755);
}

std::string when(int64_t t) {
    if (t <= 0) return "never";
    char b[32];
    const time_t tt = (time_t)t;
    struct tm tm {};
    localtime_r(&tt, &tm);
    strftime(b, sizeof b, "%Y-%m-%d %H:%M", &tm);
    return b;
}

}  // namespace

std::string default_keys_path() {
    if (const char *k = getenv("TS5_KEYS"); k && *k) return k;
    if (const char *d = getenv("TS5_DATA_DIR"); d && *d) return std::string(d) + "/access/keys.tsv";
    if (const char *x = getenv("XDG_DATA_HOME"); x && *x == '/') return std::string(x) + "/tokenstation/access/keys.tsv";
    const char *h = getenv("HOME");
    return std::string(h && *h ? h : ".") + "/.local/share/tokenstation/access/keys.tsv";
}

bool KeyFile::load(std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    return load_locked(err);
}

bool KeyFile::load_locked(std::string &err) {
    keys_.clear();
    required_ = false;
    seen_mtime_ns_ = seen_size_ = -1;
    struct stat st {};
    if (stat(path_.c_str(), &st) != 0) {
        if (errno == ENOENT || errno == ENOTDIR) return true;
        err = "can't read " + path_ + ": " + strerror(errno);
        return false;
    }
    FILE *f = fopen(path_.c_str(), "r");
    if (!f) {
        err = "can't read " + path_ + ": " + strerror(errno);
        return false;
    }
    seen_mtime_ns_ = (int64_t)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
    seen_size_ = (int64_t)st.st_size;
    char buf[1024];
    std::vector<std::string> names = split_tabs(std::string(kColumns).substr(1));
    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.compare(0, strlen(kRequired), kRequired) == 0) required_ = line.substr(strlen(kRequired)) != "0";
            else if (line.size() > 1 && line[1] != ' ') names = split_tabs(line.substr(1));
            continue;
        }
        const std::vector<std::string> v = split_tabs(line);
        AccessKey k;
        for (size_t i = 0; i < v.size() && i < names.size(); i++) {
            if (names[i] == "id") k.id = v[i];
            else if (names[i] == "name") k.name = v[i];
            else if (names[i] == "sha256") k.sha256 = v[i];
            else if (names[i] == "on") k.on = v[i] != "0";
            else if (names[i] == "created") k.created = strtoll(v[i].c_str(), nullptr, 10);
            else if (names[i] == "used") k.used = strtoll(v[i].c_str(), nullptr, 10);
        }
        if (k.id.empty() || k.sha256.size() != 64) continue;  // not a key
        keys_.push_back(k);
    }
    fclose(f);
    // When each was last used, as the server wrote it down.
    if (FILE *u = fopen((path_ + ".used").c_str(), "r")) {
        while (fgets(buf, sizeof buf, u)) {
            const std::vector<std::string> v = split_tabs(buf);
            if (v.size() < 2) continue;
            const int64_t t = strtoll(v[1].c_str(), nullptr, 10);
            for (AccessKey &k : keys_)
                if (k.id == v[0] && t > k.used) k.used = t;
        }
        fclose(u);
    }
    return true;
}

bool KeyFile::save_locked(std::string &err) {
    std::string text = std::string(kHeader) + "\n";
    if (required_) text += std::string(kRequired) + "1\n";
    text += std::string(kColumns) + "\n";
    for (const AccessKey &k : keys_)
        text += k.id + "\t" + clean(k.name) + "\t" + k.sha256 + "\t" + (k.on ? "1" : "0") + "\t" + std::to_string(k.created) +
                "\t" + std::to_string(k.used) + "\n";
    make_parents(path_);
    // (Its hashes are no keys, but they are nobody else's business: the
    // owner and the group, which is the service's when the file is its.)
    if (!write_file(path_, text, 0640, err)) return false;
    struct stat st {};
    if (stat(path_.c_str(), &st) == 0) {
        seen_mtime_ns_ = (int64_t)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
        seen_size_ = (int64_t)st.st_size;
    }
    return true;
}

std::vector<AccessKey> KeyFile::keys() const {
    std::lock_guard<std::mutex> lk(mu_);
    return keys_;
}

bool KeyFile::required() const {
    std::lock_guard<std::mutex> lk(mu_);
    return required_;
}

int KeyFile::find_locked(const std::string &which, std::string &err) const {
    int by_name = -1, names = 0;
    for (size_t i = 0; i < keys_.size(); i++) {
        if (keys_[i].id == which) return (int)i;
        if (keys_[i].name == which) by_name = (int)i, names++;
    }
    if (names == 1) return by_name;
    err = names > 1 ? "more than one key is named " + which + ": give its id (ts5 keys list)" : "there is no key " + which;
    return -1;
}

std::string KeyFile::create(const std::string &name, std::string &id, std::string &err) {
    const std::string n = clean(name);
    if (n.empty()) {
        err = "a key needs a name, such as the app or the machine it is for";
        return "";
    }
    // 20 characters of 5 bits from 13 random bytes (100 bits used), and 4
    // bytes for the id.
    unsigned char r[17];
    if (!random_bytes(r, sizeof r)) {
        err = "the system gave no random numbers";
        return "";
    }
    std::string key = "ts5";
    uint32_t acc = 0;
    int bits = 0, made = 0;
    for (int i = 0; i < 13 && made < 20; i++) {
        acc = (acc << 8) | r[i];
        bits += 8;
        while (bits >= 5 && made < 20) {
            if (made % 5 == 0) key += '-';
            key += kAlphabet[(acc >> (bits - 5)) & 31];
            bits -= 5;
            made++;
        }
    }
    char idbuf[9];
    snprintf(idbuf, sizeof idbuf, "%02x%02x%02x%02x", r[13], r[14], r[15], r[16]);
    AccessKey k;
    k.id = idbuf;
    k.name = n;
    k.sha256 = sha256_hex(key);
    k.created = (int64_t)time(nullptr);
    if (k.sha256.size() != 64) {
        err = "this build has no SHA-256";
        return "";
    }
    std::lock_guard<std::mutex> lk(mu_);
    keys_.push_back(k);
    if (!save_locked(err)) {
        keys_.pop_back();
        return "";
    }
    id = k.id;
    return key;
}

bool KeyFile::set_on(const std::string &which, bool on, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    const int i = find_locked(which, err);
    if (i < 0) return false;
    const bool was = keys_[(size_t)i].on;
    keys_[(size_t)i].on = on;
    if (save_locked(err)) return true;
    keys_[(size_t)i].on = was;
    return false;
}

bool KeyFile::remove(const std::string &which, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    const int i = find_locked(which, err);
    if (i < 0) return false;
    const AccessKey gone = keys_[(size_t)i];
    keys_.erase(keys_.begin() + i);
    if (save_locked(err)) return true;
    keys_.insert(keys_.begin() + i, gone);
    return false;
}

bool KeyFile::set_required(bool required, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    const bool was = required_;
    required_ = required;
    if (save_locked(err)) return true;
    required_ = was;
    return false;
}

void KeyFile::note_used_locked(AccessKey &k, int64_t now) {
    if (now - k.used < 60) return;
    k.used = now;
    std::string text;
    for (const AccessKey &x : keys_)
        if (x.used > 0) text += x.id + "\t" + std::to_string(x.used) + "\n";
    std::string err;
    write_file(path_ + ".used", text, 0640, err);  // (a folder it can't write in: the times are not kept)
}

bool KeyFile::authorize(const std::string &key, const std::string &client, std::string &name) {
    std::lock_guard<std::mutex> lk(mu_);
    // The file as it is now: looked at once a second at most.
    if (const double now = mono_s(); now - looked_ >= 1) {
        looked_ = now;
        struct stat st {};
        const bool there = stat(path_.c_str(), &st) == 0;
        const int64_t mtime = there ? (int64_t)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec : -1;
        const int64_t size = there ? (int64_t)st.st_size : -1;
        if (mtime != seen_mtime_ns_ || size != seen_size_) {
            std::string err;
            load_locked(err);
        }
    }
    name.clear();
    if (!key.empty()) {
        const std::string h = sha256_hex(key);
        for (AccessKey &k : keys_)
            if (k.on && same(k.sha256, h)) {
                name = k.name;
                note_used_locked(k, (int64_t)time(nullptr));
                return true;
            }
    }
    const bool here = client == "::1" || client.compare(0, 4, "127.") == 0 || client == "::ffff:127.0.0.1";
    // A device on a tailnet reaches this machine from anywhere: it always
    // needs a key, as on the TV app.
    return !on_tailnet(client) && (here || !required_);
}

bool on_tailnet(const std::string &client) {
    int a = 0, b = 0;
    return sscanf(client.c_str(), "%d.%d.", &a, &b) == 2 && a == 100 && b >= 64 && b < 128;
}

int keys_main(int argc, char **argv) {
    std::string path = default_keys_path();
    bool json = false;
    std::vector<std::string> a;
    for (int i = 2; i < argc; i++) {
        const std::string s = argv[i];
        if (s == "--file" && i + 1 < argc) path = argv[++i];
        else if (s == "--json") json = true;
        else a.push_back(s);
    }
    const std::string cmd = a.empty() ? "list" : a[0];
    const auto fail = [&](const std::string &why) {
        if (json) printf("{\"error\":%s}\n", json_escape(why).c_str());
        else fprintf(stderr, "ts5 keys: %s\n", why.c_str());
        return 1;
    };
    const auto usage = [&] {
        return fail("ts5 keys [--file FILE] [--json] list | new NAME | on KEY | off KEY | remove KEY | require on|off "
                    "(KEY: a key's id or name; FILE: " + default_keys_path() + ")");
    };
    KeyFile f(path);
    std::string err;
    if (!f.load(err)) return fail(err);
    if (cmd == "new") {
        if (a.size() < 2) return usage();
        std::string name = a[1];
        for (size_t i = 2; i < a.size(); i++) name += " " + a[i];
        std::string id;
        const std::string key = f.create(name, id, err);
        if (key.empty()) return fail(err);
        if (json) {
            printf("{\"id\":%s,\"name\":%s,\"key\":%s}\n", json_escape(id).c_str(), json_escape(clean(name)).c_str(),
                   json_escape(key).c_str());
        } else {
            printf("%s\n", key.c_str());
            fprintf(stderr,
                    "That is the key for \"%s\". It is shown this once: %s keeps its SHA-256, not the key.\n"
                    "An app sends it as its API key (Authorization: Bearer ...).\n",
                    clean(name).c_str(), path.c_str());
        }
        return 0;
    }
    if (cmd == "on" || cmd == "off" || cmd == "remove") {
        if (a.size() != 2) return usage();
        if (!(cmd == "remove" ? f.remove(a[1], err) : f.set_on(a[1], cmd == "on", err))) return fail(err);
    } else if (cmd == "require") {
        if (a.size() != 2 || (a[1] != "on" && a[1] != "off")) return usage();
        if (!f.set_required(a[1] == "on", err)) return fail(err);
    } else if (cmd != "list" || a.size() > 1) {
        return usage();
    }
    const std::vector<AccessKey> keys = f.keys();
    if (json) {
        // (Whether this user may change the file: the file, or the folder a new one would be made in.)
        const std::string dir = path.find('/') == std::string::npos ? "." : path.substr(0, path.rfind('/'));
        const bool there = access(path.c_str(), F_OK) == 0;
        const bool writable = access(dir.c_str(), W_OK) == 0 || (!there && access(dir.c_str(), F_OK) != 0);
        std::string out = "{\"file\":" + json_escape(path) + ",\"required\":" + (f.required() ? "true" : "false") +
                          ",\"writable\":" + (writable ? "true" : "false") + ",\"keys\":[";
        for (size_t i = 0; i < keys.size(); i++) {
            const AccessKey &k = keys[i];
            out += std::string(i ? "," : "") + "{\"id\":" + json_escape(k.id) + ",\"name\":" + json_escape(k.name) +
                   ",\"on\":" + (k.on ? "true" : "false") + ",\"created\":" + std::to_string(k.created) + ",\"used\":" +
                   std::to_string(k.used) + "}";
        }
        printf("%s]}\n", out.c_str());
        return 0;
    }
    printf("Keys in %s\nOther machines need a key: %s\n", path.c_str(),
           f.required() ? "yes" : "no (ts5 keys require on)");
    if (keys.empty()) {
        printf("\nNo keys yet. Make one with: ts5 keys new NAME\n");
        return 0;
    }
    size_t w = 4;
    for (const AccessKey &k : keys) w = std::max(w, k.name.size());
    printf("\n%-8s  %-*s  %-3s  %-16s  %s\n", "ID", (int)w, "NAME", "ON", "MADE", "LAST USED");
    for (const AccessKey &k : keys)
        printf("%-8s  %-*s  %-3s  %-16s  %s\n", k.id.c_str(), (int)w, k.name.c_str(), k.on ? "yes" : "off",
               when(k.created).c_str(), when(k.used).c_str());
    return 0;
}

}  // namespace ts5
