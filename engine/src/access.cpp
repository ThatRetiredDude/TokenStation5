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
#include <dirent.h>

#include <algorithm>

#include "json.h"
#include "net.h"

namespace ts5 {

namespace {

constexpr const char *kHeader = "# ts5 keys v1";
constexpr const char *kRequired = "# required=";
constexpr const char *kColumns = "#id\tname\tsha256\ton\tcreated\tused\tuser";
constexpr const char *kUsersHeader = "# ts5 users v1";
constexpr const char *kUsersColumns = "#id\tname\ton\tweb_search\tapp_tools\ttokens_day\tmax_reply\tcreated\towner_reads";
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

std::string dir_of(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash);
}

// Today on this machine's clock: "2026-11-14".
std::string local_day() {
    char b[16];
    const time_t t = time(nullptr);
    struct tm tm {};
    localtime_r(&t, &tm);
    strftime(b, sizeof b, "%Y-%m-%d", &tm);
    return b;
}

int64_t mtime_ns(const struct stat &st) { return (int64_t)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec; }

bool parse_count(const std::string &v, int64_t &out) {
    if (v.empty() || v.size() > 12) return false;
    for (char c : v)
        if (c < '0' || c > '9') return false;
    out = strtoll(v.c_str(), nullptr, 10);
    return true;
}

}  // namespace

// (An app's own tools stay on in both: an editor pointed at the address
// runs them itself, on its own machine, and needs them to work.)
bool apply_preset(AccessUser &u, const std::string &preset) {
    if (preset == "chat") u.web_search = false;
    else if (preset == "chat-web") u.web_search = true;
    else return false;
    return true;
}

std::string preset_of(const AccessUser &u) {
    if (u.owner()) return "owner";
    return u.web_search ? "chat-web" : "chat";
}

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
            else if (names[i] == "user" && !v[i].empty()) k.user = v[i];
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
    // The users, beside it.
    users_.clear();
    users_mtime_ns_ = users_size_ = -1;
    const std::string upath = dir_of(path_) + "/users.tsv";
    if (stat(upath.c_str(), &st) == 0) {
        users_mtime_ns_ = mtime_ns(st);
        users_size_ = (int64_t)st.st_size;
    }
    if (FILE *uf = fopen(upath.c_str(), "r")) {
        std::vector<std::string> cols = split_tabs(std::string(kUsersColumns).substr(1));
        while (fgets(buf, sizeof buf, uf)) {
            std::string line(buf);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (line.size() > 1 && line[1] != ' ') cols = split_tabs(line.substr(1));
                continue;
            }
            const std::vector<std::string> v = split_tabs(line);
            AccessUser u;
            for (size_t i = 0; i < v.size() && i < cols.size(); i++) {
                const std::string &c = cols[i];
                if (c == "id") u.id = v[i];
                else if (c == "name") u.name = v[i];
                else if (c == "on") u.on = v[i] != "0";
                else if (c == "web_search") u.web_search = v[i] != "0";
                else if (c == "app_tools") u.app_tools = v[i] != "0";
                else if (c == "tokens_day") u.tokens_day = strtoll(v[i].c_str(), nullptr, 10);
                else if (c == "max_reply") u.max_reply = strtoll(v[i].c_str(), nullptr, 10);
                else if (c == "created") u.created = strtoll(v[i].c_str(), nullptr, 10);
                else if (c == "owner_reads") u.owner_reads = v[i] == "1";
            }
            if (u.id.size() != 8 || u.owner()) continue;  // not a user
            users_.push_back(u);
        }
        fclose(uf);
    }
    load_today_locked();
    return true;
}

void KeyFile::load_today_locked() {
    today_.clear();
    today_date_ = local_day();
    FILE *f = fopen((dir_of(path_) + "/users.today").c_str(), "r");
    if (!f) return;
    char buf[256];
    bool same_day = false;
    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        const std::vector<std::string> v = split_tabs(line);
        if (v.size() < 2) continue;
        if (v[0] == "date") same_day = v[1] == today_date_;
        else if (same_day) today_[v[0]] = strtoll(v[1].c_str(), nullptr, 10);
    }
    fclose(f);
}

bool KeyFile::save_users_locked(std::string &err) {
    std::string text = std::string(kUsersHeader) + "\n" + kUsersColumns + "\n";
    for (const AccessUser &u : users_)
        text += u.id + "\t" + clean(u.name) + "\t" + (u.on ? "1" : "0") + "\t" + (u.web_search ? "1" : "0") + "\t" +
                (u.app_tools ? "1" : "0") + "\t" + std::to_string(u.tokens_day) + "\t" + std::to_string(u.max_reply) +
                "\t" + std::to_string(u.created) + "\t" + (u.owner_reads ? "1" : "0") + "\n";
    const std::string upath = dir_of(path_) + "/users.tsv";
    make_parents(upath);
    if (!write_file(upath, text, 0640, err)) return false;
    struct stat st {};
    if (stat(upath.c_str(), &st) == 0) {
        users_mtime_ns_ = mtime_ns(st);
        users_size_ = (int64_t)st.st_size;
    }
    return true;
}

bool KeyFile::save_locked(std::string &err) {
    std::string text = std::string(kHeader) + "\n";
    if (required_) text += std::string(kRequired) + "1\n";
    // The user column only once a key is another user's: until then the
    // file is line for line what it was before users.
    bool users = false;
    for (const AccessKey &k : keys_) users |= k.user != "owner";
    text += (users ? std::string(kColumns) : std::string(kColumns).substr(0, std::string(kColumns).rfind('\t'))) + "\n";
    for (const AccessKey &k : keys_)
        text += k.id + "\t" + clean(k.name) + "\t" + k.sha256 + "\t" + (k.on ? "1" : "0") + "\t" + std::to_string(k.created) +
                "\t" + std::to_string(k.used) + (users ? "\t" + k.user : std::string()) + "\n";
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

std::string KeyFile::create(const std::string &name, std::string &id, std::string &err, const std::string &user) {
    const std::string n = clean(name);
    if (n.empty()) {
        err = "a key needs a name, such as the app or the machine it is for";
        return "";
    }
    std::string owner_id = "owner";
    if (!user.empty() && user != "owner") {
        std::lock_guard<std::mutex> lk(mu_);
        const int u = find_user_locked(user, err);
        if (u < 0) return "";
        owner_id = users_[(size_t)u].id;
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
    k.user = owner_id;
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

// The files as they are now: looked at once a second at most.
bool KeyFile::fresh_locked() {
    const double now = mono_s();
    if (now - looked_ < 1) return false;
    looked_ = now;
    struct stat st {};
    const bool there = stat(path_.c_str(), &st) == 0;
    const int64_t mtime = there ? mtime_ns(st) : -1, size = there ? (int64_t)st.st_size : -1;
    const bool uthere = stat((dir_of(path_) + "/users.tsv").c_str(), &st) == 0;
    const int64_t umtime = uthere ? mtime_ns(st) : -1, usize = uthere ? (int64_t)st.st_size : -1;
    if (mtime != seen_mtime_ns_ || size != seen_size_ || umtime != users_mtime_ns_ || usize != users_size_) {
        // (What was counted today is the server's own: kept over the reload.)
        const std::map<std::string, int64_t> today = today_;
        const std::string date = today_date_;
        std::string err;
        load_locked(err);
        if (date == today_date_) today_ = today;
        return true;
    }
    return false;
}

const AccessUser *KeyFile::user_locked(const std::string &id) const {
    for (const AccessUser &u : users_)
        if (u.id == id) return &u;
    return nullptr;
}

bool KeyFile::authorize(const std::string &key, const std::string &client, std::string &name) {
    Server::Grant g;
    const bool ok = admit(key, client, g);
    name = g.key;
    return ok;
}

bool KeyFile::admit(const std::string &key, const std::string &client, Server::Grant &g) {
    std::lock_guard<std::mutex> lk(mu_);
    fresh_locked();
    g = Server::Grant();
    g.user = "owner";
    g.user_name = "Owner";
    if (!key.empty()) {
        const std::string h = sha256_hex(key);
        for (AccessKey &k : keys_)
            if (k.on && same(k.sha256, h)) {
                if (k.user != "owner") {
                    // Another user's key: only while that user is there and on.
                    const AccessUser *u = user_locked(k.user);
                    if (!u || !u->on) return false;
                    g.owner = false;
                    g.user = u->id;
                    g.user_name = u->name;
                    g.web_search = u->web_search;
                    g.app_tools = u->app_tools;
                    g.max_reply = u->max_reply;
                    g.tokens_day = u->tokens_day;
                    g.owner_reads = u->owner_reads;
                    if (today_date_ != local_day()) today_.clear(), today_date_ = local_day();
                    const auto t = today_.find(u->id);
                    if (u->tokens_day > 0 && t != today_.end() && t->second >= u->tokens_day)
                        g.over = "Today's tokens for " + u->name + " are used up (" + std::to_string(u->tokens_day) +
                                 " a day). More tomorrow, or the owner raises the limit on the Access page.";
                }
                g.key = k.name;
                note_used_locked(k, (int64_t)time(nullptr));
                return true;
            }
    }
    const bool here = client == "::1" || client.compare(0, 4, "127.") == 0 || client == "::ffff:127.0.0.1";
    // A device on a tailnet reaches this machine from anywhere: it always
    // needs a key, as on the TV app.
    return !on_tailnet(client) && (here || !required_);
}

std::vector<AccessUser> KeyFile::users() const {
    std::lock_guard<std::mutex> lk(mu_);
    AccessUser owner;
    owner.id = "owner";
    owner.name = "Owner";
    std::vector<AccessUser> out{owner};
    out.insert(out.end(), users_.begin(), users_.end());
    for (AccessUser &u : out) {
        const auto t = today_.find(u.id);
        u.today = t == today_.end() ? 0 : t->second;
    }
    return out;
}

int KeyFile::find_user_locked(const std::string &which, std::string &err) const {
    int by_name = -1, names = 0;
    for (size_t i = 0; i < users_.size(); i++) {
        if (users_[i].id == which) return (int)i;
        if (users_[i].name == which) by_name = (int)i, names++;
    }
    if (names == 1) return by_name;
    err = which == "owner" || which == "Owner" ? "the owner is always there, with everything: it can't be changed"
          : names > 1                            ? "more than one user is named " + which + ": give its id (ts5 users list)"
                                                 : "there is no user " + which;
    return -1;
}

std::string KeyFile::add_user(const std::string &name, const std::string &preset, std::string &err) {
    AccessUser u;
    u.name = clean(name);
    if (u.name.empty()) {
        err = "a user needs a name, such as the person's";
        return "";
    }
    if (u.name == "owner" || u.name == "Owner") {
        err = "Owner is the owner's own user";
        return "";
    }
    if (!apply_preset(u, preset.empty() ? "chat" : preset)) {
        err = "a preset is chat (chat only) or chat-web (chat and web search)";
        return "";
    }
    unsigned char r[4];
    if (!random_bytes(r, sizeof r)) {
        err = "the system gave no random numbers";
        return "";
    }
    char idbuf[9];
    snprintf(idbuf, sizeof idbuf, "%02x%02x%02x%02x", r[0], r[1], r[2], r[3]);
    u.id = idbuf;
    u.created = (int64_t)time(nullptr);
    std::lock_guard<std::mutex> lk(mu_);
    for (const AccessUser &x : users_)
        if (x.name == u.name) {
            err = "there is a user named " + u.name + " already";
            return "";
        }
    users_.push_back(u);
    if (!save_users_locked(err)) {
        users_.pop_back();
        return "";
    }
    // With a user other than the owner, every other machine needs a key:
    // without one it would come in as the owner.
    if (!required_) {
        required_ = true;
        std::string e2;
        if (!save_locked(e2)) required_ = false;
    }
    return u.id;
}

bool KeyFile::set_user(const std::string &which, const std::vector<std::string> &changes, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    const int i = find_user_locked(which, err);
    if (i < 0) return false;
    AccessUser u = users_[(size_t)i];
    for (const std::string &c : changes) {
        const size_t eq = c.find('=');
        const std::string k = c.substr(0, eq), v = eq == std::string::npos ? "" : c.substr(eq + 1);
        const bool flag = v == "1" || v == "on" || v == "yes" || v == "true";
        const bool good_flag = flag || v == "0" || v == "off" || v == "no" || v == "false";
        int64_t n = 0;
        if (eq == std::string::npos) {
            err = "a change is NAME=VALUE, such as web_search=off: " + c;
            return false;
        }
        if (k == "name") {
            const std::string nn = clean(v);
            if (nn.empty() || nn == "owner" || nn == "Owner") {
                err = "not a name for a user: " + v;
                return false;
            }
            for (size_t j = 0; j < users_.size(); j++)
                if ((int)j != i && users_[j].name == nn) {
                    err = "there is a user named " + nn + " already";
                    return false;
                }
            u.name = nn;
        } else if (k == "preset") {
            if (!apply_preset(u, v)) {
                err = "a preset is chat or chat-web";
                return false;
            }
        } else if ((k == "on" || k == "web_search" || k == "app_tools" || k == "owner_reads") && good_flag) {
            (k == "on" ? u.on : k == "web_search" ? u.web_search : k == "app_tools" ? u.app_tools : u.owner_reads) = flag;
        } else if ((k == "tokens_day" || k == "max_reply") && parse_count(v, n)) {
            (k == "tokens_day" ? u.tokens_day : u.max_reply) = n;
        } else {
            err = "not a change ts5 knows: " + c +
                  " (name=NAME, preset=chat|chat-web, on=on|off, web_search=on|off, app_tools=on|off, "
                  "owner_reads=on|off, tokens_day=N, max_reply=N; 0 is no limit)";
            return false;
        }
    }
    const AccessUser was = users_[(size_t)i];
    users_[(size_t)i] = u;
    if (save_users_locked(err)) return true;
    users_[(size_t)i] = was;
    return false;
}

bool KeyFile::remove_user(const std::string &which, bool forget, std::string &err) {
    std::lock_guard<std::mutex> lk(mu_);
    const int i = find_user_locked(which, err);
    if (i < 0) return false;
    const AccessUser gone = users_[(size_t)i];
    const std::vector<AccessKey> keys = keys_;
    keys_.erase(std::remove_if(keys_.begin(), keys_.end(), [&](const AccessKey &k) { return k.user == gone.id; }),
                keys_.end());
    if (!save_locked(err)) {
        keys_ = keys;
        return false;
    }
    users_.erase(users_.begin() + i);
    if (!save_users_locked(err)) {
        users_.insert(users_.begin() + i, gone);
        return false;
    }
    if (forget) {
        const std::string dir = user_dir(gone.id) + "/chats";
        if (DIR *d = opendir(dir.c_str())) {
            while (dirent *e = readdir(d))
                if (e->d_name[0] != '.') unlink((dir + "/" + e->d_name).c_str());
            closedir(d);
        }
        rmdir(dir.c_str());
        rmdir(user_dir(gone.id).c_str());
    }
    return true;
}

void KeyFile::count(const std::string &user, int64_t tokens) {
    if (user.empty() || tokens <= 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (today_date_ != local_day()) today_.clear(), today_date_ = local_day();
    today_[user] += tokens;
    std::string text = "date\t" + today_date_ + "\n";
    for (const auto &t : today_) text += t.first + "\t" + std::to_string(t.second) + "\n";
    std::string err;
    write_file(dir_of(path_) + "/users.today", text, 0640, err);  // (a folder it can't write in: counted in memory only)
}

std::string KeyFile::user_dir(const std::string &user) const {
    // ~/.local/share/tokenstation/access/keys.tsv: ~/.local/share/tokenstation/users/ID.
    const std::string dir = dir_of(path_);
    const size_t slash = dir.rfind('/');
    const std::string base = slash != std::string::npos && dir.substr(slash + 1) == "access" ? dir_of(dir) : dir;
    return base + "/users/" + user;
}

bool on_tailnet(const std::string &client) {
    int a = 0, b = 0;
    return sscanf(client.c_str(), "%d.%d.", &a, &b) == 2 && a == 100 && b >= 64 && b < 128;
}

int keys_main(int argc, char **argv) {
    std::string path = default_keys_path(), user;
    bool json = false;
    std::vector<std::string> a;
    for (int i = 2; i < argc; i++) {
        const std::string s = argv[i];
        if (s == "--file" && i + 1 < argc) path = argv[++i];
        else if (s == "--user" && i + 1 < argc) user = argv[++i];
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
        return fail("ts5 keys [--file FILE] [--json] list | new NAME [--user USER] | on KEY | off KEY | remove KEY | require on|off "
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
        const std::string key = f.create(name, id, err, user);
        if (key.empty()) return fail(err);
        std::string uid = "owner";
        for (const AccessKey &k : f.keys())
            if (k.id == id) uid = k.user;
        if (json) {
            printf("{\"id\":%s,\"name\":%s,\"key\":%s,\"user\":%s}\n", json_escape(id).c_str(),
                   json_escape(clean(name)).c_str(), json_escape(key).c_str(), json_escape(uid).c_str());
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
                   std::to_string(k.used) + ",\"user\":" + json_escape(k.user) + "}";
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

namespace {

std::string user_json(const AccessUser &u, const std::vector<AccessKey> &keys) {
    std::string out = "{\"id\":" + json_escape(u.id) + ",\"name\":" + json_escape(u.name) + ",\"owner\":" +
                      (u.owner() ? "true" : "false") + ",\"on\":" + (u.on ? "true" : "false") +
                      ",\"preset\":" + json_escape(preset_of(u)) + ",\"web_search\":" + (u.web_search ? "true" : "false") +
                      ",\"app_tools\":" + (u.app_tools ? "true" : "false") +
                      ",\"owner_reads\":" + (u.owner_reads ? "true" : "false") + ",\"terminal\":" +
                      (u.owner() ? "true" : "false") + ",\"tokens_day\":" + std::to_string(u.tokens_day) +
                      ",\"max_reply\":" + std::to_string(u.max_reply) + ",\"today\":" + std::to_string(u.today) +
                      ",\"created\":" + std::to_string(u.created) + ",\"keys\":[";
    bool first = true;
    for (const AccessKey &k : keys) {
        if (k.user != u.id) continue;
        out += std::string(first ? "" : ",") + "{\"id\":" + json_escape(k.id) + ",\"name\":" + json_escape(k.name) +
               ",\"on\":" + (k.on ? "true" : "false") + ",\"created\":" + std::to_string(k.created) +
               ",\"used\":" + std::to_string(k.used) + "}";
        first = false;
    }
    return out + "]}";
}

std::string limit_text(int64_t n) { return n > 0 ? std::to_string(n) : "no limit"; }

}  // namespace

int users_main(int argc, char **argv) {
    std::string path = default_keys_path(), preset;
    bool json = false, forget = false, owner_reads = false;
    std::vector<std::string> a;
    for (int i = 2; i < argc; i++) {
        const std::string s = argv[i];
        if (s == "--file" && i + 1 < argc) path = argv[++i];
        else if (s == "--preset" && i + 1 < argc) preset = argv[++i];
        else if (s == "--json") json = true;
        else if (s == "--forget") forget = true;
        else if (s == "--owner-reads") owner_reads = true;
        else a.push_back(s);
    }
    const std::string cmd = a.empty() ? "list" : a[0];
    const auto fail = [&](const std::string &why) {
        if (json) printf("{\"error\":%s}\n", json_escape(why).c_str());
        else fprintf(stderr, "ts5 users: %s\n", why.c_str());
        return 1;
    };
    const auto usage = [&] {
        return fail("ts5 users [--file FILE] [--json] list | new NAME [--preset chat|chat-web] [--owner-reads] | "
                    "set USER CHANGE... | remove USER [--forget] (USER: a user's id or name; CHANGE: name=NAME, "
                    "preset=P, on=on|off, web_search=on|off, app_tools=on|off, owner_reads=on|off, tokens_day=N, max_reply=N; "
                    "FILE: the keys' file, " + default_keys_path() + ")");
    };
    KeyFile f(path);
    std::string err;
    if (!f.load(err)) return fail(err);
    std::string made;
    if (cmd == "new") {
        if (a.size() < 2) return usage();
        std::string name = a[1];
        for (size_t i = 2; i < a.size(); i++) name += " " + a[i];
        made = f.add_user(name, preset, err);
        if (made.empty()) return fail(err);
        if (owner_reads && !f.set_user(made, {"owner_reads=1"}, err)) return fail(err);
        if (!json)
            fprintf(stderr, "Made the user \"%s\" (%s). Other machines now need a key. A key for them: ts5 keys new "
                            "NAME --user %s\n",
                    clean(name).c_str(), made.c_str(), made.c_str());
    } else if (cmd == "set") {
        if (a.size() < 3) return usage();
        if (!f.set_user(a[1], std::vector<std::string>(a.begin() + 2, a.end()), err)) return fail(err);
    } else if (cmd == "remove") {
        if (a.size() != 2) return usage();
        if (!f.remove_user(a[1], forget, err)) return fail(err);
    } else if (cmd != "list" || a.size() > 1) {
        return usage();
    }
    const std::vector<AccessUser> users = f.users();
    const std::vector<AccessKey> keys = f.keys();
    if (json) {
        const std::string dir = path.find('/') == std::string::npos ? "." : path.substr(0, path.rfind('/'));
        const bool there = access(path.c_str(), F_OK) == 0;
        const bool writable = access(dir.c_str(), W_OK) == 0 || (!there && access(dir.c_str(), F_OK) != 0);
        std::string out = "{\"file\":" + json_escape(path) + ",\"required\":" + (f.required() ? "true" : "false") +
                          ",\"writable\":" + (writable ? "true" : "false") +
                          (made.empty() ? std::string() : ",\"made\":" + json_escape(made)) + ",\"users\":[";
        for (size_t i = 0; i < users.size(); i++) out += (i ? "," : "") + user_json(users[i], keys);
        printf("%s]}\n", out.c_str());
        return 0;
    }
    printf("Users of %s\nOther machines need a key: %s\n", path.c_str(), f.required() ? "yes" : "no (ts5 keys require on)");
    for (const AccessUser &u : users) {
        size_t n = 0;
        for (const AccessKey &k : keys) n += k.user == u.id;
        printf("\n%s%s (%s)%s\n", u.name.c_str(), u.owner() ? "" : (" [" + u.id + "]").c_str(),
               u.owner() ? "everything, the browser terminal too" : preset_of(u).c_str(), u.on ? "" : ", paused");
        if (!u.owner())
            printf("  web search %s · app tools %s · tokens a day %s (today %lld) · longest reply %s · the owner reads its chats: %s\n",
                   u.web_search ? "on" : "off", u.app_tools ? "on" : "off", limit_text(u.tokens_day).c_str(),
                   (long long)u.today, limit_text(u.max_reply).c_str(), u.owner_reads ? "yes" : "no");
        printf("  %zu %s", n, n == 1 ? "key" : "keys");
        bool first = true;
        for (const AccessKey &k : keys)
            if (k.user == u.id) printf("%s %s%s", first ? ":" : ",", k.name.c_str(), k.on ? "" : " (off)"), first = false;
        printf("\n");
    }
    return 0;
}

}  // namespace ts5
