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
#include <set>

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
        for (size_t i = 0; i < c.msgs.size(); i++) {
            const Msg &m = c.msgs[i];
            if (m.role == "summary") {
                // The model's summary of the chat before it ("kept": how
                // many of the turns before it the model still gets as they
                // were). The turns themselves stay in the file.
                if (!m.done || m.stopped || !m.error.empty() || m.content.empty()) continue;
                int kept = 0;
                for (size_t k = (size_t)std::max(0, m.keep_from); k < i; k++)
                    kept += c.msgs[k].role == "user" || c.msgs[k].role == "tool" ||
                            (c.msgs[k].role == "assistant" && c.msgs[k].done);
                body += std::string(n++ ? ",\n" : "\n") + "{\"user\":false,\"summary\":true,\"text\":" + ts5::json_escape(m.content) +
                        ",\"model\":" + ts5::json_escape(m.model) + ",\"kept\":" + std::to_string(kept) +
                        ",\"before\":" + std::to_string(m.sum_before) + ",\"after\":" + std::to_string(m.sum_after) +
                        ",\"why\":" + std::to_string(m.sum_why) + "}";
                continue;
            }
            if (m.role == "tool") {
                // What a tool call gave back (ts5-tui's own fields; the TV
                // app reads it as a reply's text).
                body += std::string(n++ ? ",\n" : "\n") + "{\"user\":false,\"tool\":" + ts5::json_escape(m.tool) +
                        ",\"call\":" + ts5::json_escape(m.call_id) + ",\"head\":" + ts5::json_escape(m.head) +
                        ",\"ok\":" + (m.tool_ok ? "true" : "false") + (m.left_out ? ",\"left_out\":true" : "") +
                        ",\"text\":" + ts5::json_escape(m.content) + "}";
                continue;
            }
            if (m.role != "user" && m.role != "assistant") continue;
            if (m.role == "assistant" && !m.done) continue;
            const bool user = m.role == "user";
            if (user && first.empty()) first = m.content;
            body += std::string(n++ ? ",\n" : "\n") + "{\"user\":" + (user ? "true" : "false") +
                    ",\"text\":" + ts5::json_escape(m.content);
            if (!user && !m.calls.empty()) {
                body += ",\"calls\":[";
                for (size_t k = 0; k < m.calls.size(); k++)
                    body += std::string(k ? "," : "") + "{\"id\":" + ts5::json_escape(m.calls[k].id) + ",\"name\":" +
                            ts5::json_escape(m.calls[k].name) + ",\"args\":" + ts5::json_escape(m.calls[k].args) + "}";
                body += "]";
            }
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
    {
        std::lock_guard<std::mutex> l(c.mu);
        for (const ts5::Json &t : turns.a) {
            Msg m;
            m.done = true;
            m.content = t["text"].str();
            if (t["summary"].boolean(false)) {
                m.role = "summary";
                m.model = t["model"].str();
                m.sum_before = (int)t["before"].num(0);
                m.sum_after = (int)t["after"].num(0);
                m.sum_why = (int)t["why"].num(0);
                // The turns kept beside it: that many of the ones before it.
                int kept = (int)t["kept"].num(0);
                size_t from = c.msgs.size();
                while (kept > 0 && from > 0) {
                    from--;
                    if (c.msgs[from].role != "summary") kept--;
                }
                m.keep_from = (int)from;
            } else if (t["user"].boolean(false)) {
                m.role = "user";
                c.history.push_back(m.content);
            } else if (!t["tool"].str().empty()) {
                m.role = "tool";
                m.tool = t["tool"].str();
                m.call_id = t["call"].str();
                m.head = t["head"].str();
                m.tool_ok = t["ok"].boolean(true);
                m.left_out = t["left_out"].boolean(false);
            } else {
                m.role = "assistant";
                for (const ts5::Json &u : t["calls"].a) m.calls.push_back({u["id"].str(), u["name"].str(), u["args"].str()});
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
    }
    chat_count(c);
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

// ---- usage and keys ----

#include <map>

#include "usage.h"

namespace tui {

namespace {

// Folders that may hold the app's files: ts5-tui's data folder, then the TV
// service's.
std::vector<std::string> app_dirs() {
    std::vector<std::string> d = {data_dir(), "/var/lib/tokenstation"};
    char a[4096], b[4096];
    if (realpath(d[0].c_str(), a) && realpath(d[1].c_str(), b) && strcmp(a, b) == 0) d.pop_back();
    return d;
}

struct Sums {
    int64_t requests = 0, failed = 0, prompt = 0, gen = 0, gen_ms = 0;
    void add(const ts5::UsageRecord &r) {
        requests++;
        if (r.status >= 400 || !r.why.empty()) failed++;
        prompt += r.prompt;
        gen += r.gen;
        gen_ms += r.gen_ms;
    }
};

std::string count(int64_t n) {
    if (n >= 10000000) return fmt("%.0fM", n / 1e6);
    if (n >= 1000000) return fmt("%.1fM", n / 1e6);
    if (n >= 10000) return fmt("%.0fk", n / 1e3);
    return std::to_string((long long)n);
}

std::string row(const std::string &name, const Sums &s) {
    std::string r = "| " + name + " | " + count(s.requests) + (s.failed ? fmt(" (%lld failed)", (long long)s.failed) : "") +
                    " | " + count(s.prompt) + " | " + count(s.gen) + " | ";
    r += s.gen_ms > 0 ? fmt("%.1f", s.gen * 1000.0 / s.gen_ms) : std::string("—");
    return r + " |\n";
}

void read_usage(const std::string &path, std::vector<ts5::UsageRecord> &out) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return;
    ts5::UsageColumns cols = ts5::usage_columns();
    char *line = nullptr;
    size_t cap = 0;
    ssize_t n;
    while ((n = getline(&line, &cap, f)) > 0) {
        const std::string l(line, (size_t)n);
        if (l.rfind("#\t", 0) == 0 || (l.size() > 1 && l[0] == '#' && l.find('\t') != std::string::npos)) {
            cols = ts5::usage_columns(l);
            continue;
        }
        ts5::UsageRecord r;
        if (ts5::parse_usage_line(cols, l, r)) out.push_back(r);
    }
    free(line);
    fclose(f);
}

}  // namespace

std::string usage_report() {
    std::vector<std::string> files;
    if (const char *e = getenv("TS5_USAGE_LOG"); e && *e) files.push_back(e);
    for (auto &d : app_dirs()) {
        files.push_back(d + "/usage.prev.tsv");
        files.push_back(d + "/usage.tsv");
    }
    std::vector<ts5::UsageRecord> recs;
    std::vector<std::string> read_from;
    for (auto &f : files) {
        if (access(f.c_str(), R_OK) != 0) continue;
        const size_t before = recs.size();
        read_usage(f, recs);
        if (recs.size() > before || f.find(".prev.") == std::string::npos) read_from.push_back(f);
    }
    if (read_from.empty()) return "";
    const int64_t now_ms = (int64_t)time(nullptr) * 1000;
    Sums day, week;
    std::map<std::string, Sums> by_app, by_model;
    std::set<std::string> seen;  // a line in two files (a copy) counts once
    for (auto &r : recs) {
        if (!seen.insert(r.id + "\t" + std::to_string(r.time_ms)).second) continue;
        if (r.time_ms < now_ms - 7 * 86400000LL) continue;
        week.add(r);
        const std::string app = !r.key.empty() ? r.key : r.client.empty() ? "(unknown)" : r.client;
        by_app[app].add(r);
        by_model[r.model.empty() ? "(no model)" : r.model].add(r);
        if (r.time_ms >= now_ms - 86400000LL) day.add(r);
    }
    std::string m = "**API use**, from " + read_from[0] + (read_from.size() > 1 ? fmt(" and %d more", (int)read_from.size() - 1) : "") + "\n\n";
    if (week.requests == 0) return m + "No requests in the last 7 days.\n";
    m += "| | Requests | Prompt tokens | Generated | tok/s |\n| --- | --- | --- | --- | --- |\n";
    m += row("Last 24 hours", day) + row("Last 7 days", week);
    auto top = [&](const std::map<std::string, Sums> &by, const char *what) {
        std::vector<std::pair<std::string, Sums>> v(by.begin(), by.end());
        std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.second.requests > b.second.requests; });
        std::string t = std::string("\n**By ") + what + "**, last 7 days\n\n| | Requests | Prompt tokens | Generated | tok/s |\n| --- | --- | --- | --- | --- |\n";
        for (size_t i = 0; i < v.size() && i < 8; i++) t += row(v[i].first, v[i].second);
        return t;
    };
    return m + top(by_app, "app (API key name, else address)") + top(by_model, "model");
}

std::string keys_report() {
    for (auto &d : app_dirs()) {
        const std::string path = d + "/keys.tsv";
        FILE *f = fopen(path.c_str(), "r");
        if (!f) {
            if (access(path.c_str(), F_OK) == 0)
                return "The API keys are in `" + path + "`, which your user can't read. They're made and turned on or off on the TV app's Settings page.";
            continue;
        }
        std::string m = "**API keys**, from `" + path + "` (only their SHA-256 is kept, so a key shows once, when it's made: on the TV app's Settings page)\n\n| Name | On | Made | Last used |\n| --- | --- | --- | --- |\n";
        char *line = nullptr;
        size_t cap = 0;
        ssize_t n;
        int keys = 0;
        while ((n = getline(&line, &cap, f)) > 0) {
            std::string l(line, (size_t)n);
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
            if (l.empty() || l[0] == '#') continue;
            std::vector<std::string> c;
            size_t at = 0;
            for (;;) {
                const size_t tab = l.find('\t', at);
                c.push_back(l.substr(at, tab == std::string::npos ? std::string::npos : tab - at));
                if (tab == std::string::npos) break;
                at = tab + 1;
            }
            // id, name, sha256, on, created, used
            if (c.size() < 6) continue;
            const int64_t made = atoll(c[4].c_str()), used = atoll(c[5].c_str());
            m += "| " + c[1] + " | " + (c[3] == "1" ? "yes" : "off") + " | " + (made ? when_text(made) : "—") + " | " +
                 (used ? when_text(used) : "never") + " |\n";
            keys++;
        }
        free(line);
        fclose(f);
        if (!keys) return "No API keys yet (`" + path + "`). They're made on the TV app's Settings page.";
        return m;
    }
    return "";
}

}  // namespace tui
