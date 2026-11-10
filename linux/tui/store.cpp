// ts5-tui's saved chats (store.h).
#include "store.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "json.h"

namespace tui {

std::string data_dir() {
    if (const char *e = getenv("TS5_DATA_DIR"); e && *e) return e;
    if (const char *x = getenv("XDG_DATA_HOME"); x && *x == '/') return std::string(x) + "/tokenstation";
    return home_dir() + "/.local/share/tokenstation";
}

bool make_dirs(const std::string &path) {
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') {
            const std::string p = path.substr(0, i);
            if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
    return true;
}

namespace {

const char *kTvChats = "/var/lib/tokenstation/chats";

std::string my_chats() { return data_dir() + "/chats"; }

bool slurp(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[65536];
    size_t n;
    out.clear();
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f);
    return true;
}

std::string lower(std::string s) {
    for (auto &ch : s) ch = (char)tolower((unsigned char)ch);
    return s;
}

// As the TV app names chats: "c", the time in milliseconds and a digit.
std::string new_id() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    static int n = 0;
    return "c" + std::to_string((long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000) + std::to_string(n++ % 10);
}

// The TV app's chat_title(): the first message on one line, cut at about 60
// characters.
std::string title_of(const std::string &first) {
    std::string t;
    for (char ch : first) {
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        if (ch == ' ' && (t.empty() || t.back() == ' ')) continue;
        t += ch;
    }
    while (!t.empty() && t.back() == ' ') t.pop_back();
    size_t chars = 0, cut = 0;
    for (size_t i = 0; i < t.size(); i++) {
        if (((unsigned char)t[i] & 0xC0) == 0x80) continue;
        if (chars++ == 60) {
            cut = i;
            break;
        }
    }
    if (cut == 0) return t.empty() ? "A chat" : t;
    t = t.substr(0, cut);
    const size_t sp = t.rfind(' ');
    if (sp != std::string::npos && sp > cut * 2 / 3) t = t.substr(0, sp);
    return t + "\xE2\x80\xA6";
}

bool read_chat(const std::string &dir, const std::string &id, SavedChat &s, ts5::Json *turns, std::string *all) {
    std::string text;
    ts5::Json j;
    if (!slurp(dir + "/" + id + ".json", text) || !ts5::Json::parse(text, j) || j.type != ts5::Json::OBJ) return false;
    s.id = id;
    s.dir = dir;
    s.title = j["title"].str("A chat");
    s.created = (int64_t)j["created"].num(0);
    s.updated = (int64_t)j["updated"].num((double)s.created);
    s.messages = (int)j["turns"].a.size();
    if (all) {
        *all = lower(s.title);
        for (auto &t : j["turns"].a) *all += "\n" + lower(t["text"].str());
    }
    if (turns) *turns = j["turns"];
    return true;
}

void scan(const std::string &dir, bool mine, const std::vector<std::string> &words, std::vector<SavedChat> &out) {
    DIR *d = opendir(dir.c_str());
    if (!d) return;
    std::vector<std::string> ids;
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 5 && n[0] != '.' && n.compare(n.size() - 5, 5, ".json") == 0) ids.push_back(n.substr(0, n.size() - 5));
    }
    closedir(d);
    for (auto &id : ids) {
        SavedChat s;
        std::string all;
        if (!read_chat(dir, id, s, nullptr, words.empty() ? nullptr : &all)) continue;
        bool every = true;
        for (auto &w : words) every = every && all.find(w) != std::string::npos;
        if (!every) continue;
        s.mine = mine;
        out.push_back(s);
    }
}

}  // namespace

std::vector<SavedChat> list_chats(const std::string &query) {
    std::vector<std::string> words;
    {
        const std::string q = lower(query);
        size_t i = 0;
        while (i < q.size()) {
            const size_t a = q.find_first_not_of(" \t", i);
            if (a == std::string::npos) break;
            const size_t b = std::min(q.find_first_of(" \t", a), q.size());
            words.push_back(q.substr(a, b - a));
            i = b;
        }
    }
    std::vector<SavedChat> out;
    scan(my_chats(), true, words, out);
    char a[4096], b[4096];
    const bool same = realpath(my_chats().c_str(), a) && realpath(kTvChats, b) && strcmp(a, b) == 0;
    if (!same && access(kTvChats, R_OK | X_OK) == 0) scan(kTvChats, false, words, out);
    std::sort(out.begin(), out.end(), [](const SavedChat &x, const SavedChat &y) {
        return x.updated != y.updated ? x.updated > y.updated : x.id > y.id;
    });
    return out;
}

bool save_chat(Chat &c, std::string &why) {
    std::string first, body;
    int n = 0;
    {
        std::lock_guard<std::mutex> l(c.mu);
        for (const Msg &m : c.msgs) {
            if (m.role != "user" && m.role != "assistant") continue;
            if (m.role == "assistant" && !m.done) continue;
            const bool user = m.role == "user";
            if (user && first.empty()) first = m.content;
            body += std::string(n++ ? ",\n" : "\n") + "{\"user\":" + (user ? "true" : "false") +
                    ",\"text\":" + ts5::json_escape(m.content);
            if (!user) {
                const char *end = !m.error.empty() ? "failed" : m.stopped ? "stopped" : "done";
                body += ",\"model\":" + ts5::json_escape(m.model) + fmt(",\"tps\":%.2f", m.gen_tps) +
                        ",\"tokens\":" + std::to_string(m.gen_tokens) + ",\"end\":\"" + end + "\"";
                if (!m.error.empty()) body += ",\"error\":" + ts5::json_escape(m.error);
                if (!m.reasoning.empty()) body += ",\"reasoning\":" + ts5::json_escape(m.reasoning);
            }
            body += "}";
        }
    }
    if (first.empty()) return true;  // nothing of yours yet
    const std::string dir = my_chats();
    if (!make_dirs(dir)) {
        why = "can't make " + dir + ": " + strerror(errno);
        return false;
    }
    const int64_t now = (int64_t)time(nullptr);
    int64_t created = now;
    if (c.saved_id.empty()) c.saved_id = new_id();
    else {
        SavedChat old;
        if (read_chat(dir, c.saved_id, old, nullptr, nullptr)) created = old.created;
    }
    const std::string text = "{\"v\":1,\"title\":" + ts5::json_escape(title_of(first)) +
                             ",\"created\":" + std::to_string(created) + ",\"updated\":" + std::to_string(now) +
                             ",\"turns\":[" + body + "\n]}\n";
    const std::string path = dir + "/" + c.saved_id + ".json", tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    bool ok = f && fwrite(text.data(), 1, text.size(), f) == text.size();
    if (f) ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        why = "can't write " + path + ": " + strerror(errno);
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

bool load_chat(Chat &c, const SavedChat &s, std::string &why) {
    SavedChat got;
    ts5::Json turns;
    if (!read_chat(s.dir, s.id, got, &turns, nullptr)) {
        why = "can't read " + s.dir + "/" + s.id + ".json";
        return false;
    }
    chat_new(c);
    std::lock_guard<std::mutex> l(c.mu);
    for (const ts5::Json &t : turns.a) {
        Msg m;
        m.done = true;
        m.content = t["text"].str();
        if (t["user"].boolean(false)) {
            m.role = "user";
            c.history.push_back(m.content);
        } else {
            m.role = "assistant";
            m.model = t["model"].str();
            m.gen_tps = t["tps"].num(0);
            m.gen_tokens = (int)t["tokens"].num(0);
            m.reasoning = t["reasoning"].str();
            const std::string end = t["end"].str("done");
            m.stopped = end == "stopped";
            if (end == "failed") m.error = t["error"].str("it failed");
        }
        c.msgs.push_back(m);
    }
    // Answers go on in ts5-tui's folder: a TV chat becomes a copy of its own.
    c.saved_id = s.mine ? s.id : "";
    c.scroll = 0;
    return true;
}

std::string when_text(int64_t t) {
    const time_t now = time(nullptr), tt = (time_t)t;
    const long ago = (long)(now - tt);
    if (ago < 60) return "just now";
    if (ago < 3600) return fmt("%ld min ago", ago / 60);
    tm a, b;
    localtime_r(&tt, &a);
    localtime_r(&now, &b);
    char buf[64];
    if (a.tm_year == b.tm_year && a.tm_yday == b.tm_yday) strftime(buf, sizeof buf, "today %H:%M", &a);
    else if (a.tm_year == b.tm_year && a.tm_yday == b.tm_yday - 1) strftime(buf, sizeof buf, "yesterday %H:%M", &a);
    else if (a.tm_year == b.tm_year) strftime(buf, sizeof buf, "%e %b %H:%M", &a);
    else strftime(buf, sizeof buf, "%e %b %Y", &a);
    std::string r = buf;
    while (!r.empty() && r[0] == ' ') r.erase(0, 1);
    return r;
}

}  // namespace tui
