#include "chat_store.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <ctime>

#include "json.h"
#include "log.h"

namespace app {

namespace {

std::string lower(std::string s) {
    for (auto &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

bool read_file(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    char buf[16384];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) out.append(buf, n);
    fclose(f);
    return true;
}

const char *end_name(ChatTurn::End e) {
    return e == ChatTurn::Stopped ? "stopped" : e == ChatTurn::Failed ? "failed" : "done";
}

ChatTurn::End end_of(const std::string &s) {
    return s == "stopped" ? ChatTurn::Stopped : s == "failed" ? ChatTurn::Failed : ChatTurn::Done;
}

// "c" and the time in milliseconds: file names that sort as the chats began.
std::string new_id() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    static int n = 0;
    return "c" + std::to_string((long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000) + std::to_string(n++ % 10);
}

bool newer(const ChatSummary &a, const ChatSummary &b) {
    return a.updated != b.updated ? a.updated > b.updated : a.id > b.id;
}

}  // namespace

std::string chat_title(const std::string &first_message) {
    std::string t;
    for (char c : first_message) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (c == ' ' && (t.empty() || t.back() == ' ')) continue;
        t += c;
    }
    while (!t.empty() && t.back() == ' ') t.pop_back();
    constexpr size_t kMax = 60;
    size_t chars = 0, cut = 0;
    for (size_t i = 0; i < t.size(); i++) {
        if (((unsigned char)t[i] & 0xC0) == 0x80) continue;  // inside a character
        if (chars++ == kMax) {
            cut = i;
            break;
        }
    }
    if (cut == 0) return t.empty() ? "A chat" : t;
    t = t.substr(0, cut);
    // At a word's end where there is one close by.
    const size_t sp = t.rfind(' ');
    if (sp != std::string::npos && sp > cut * 2 / 3) t = t.substr(0, sp);
    return t + "\xE2\x80\xA6";
}

ChatStore::ChatStore(const std::string &dir) : dir_(dir) { mkdir(dir_.c_str(), 0777); }

std::string ChatStore::path_of(const std::string &id) const { return dir_ + "/" + id + ".json"; }

bool ChatStore::read(const std::string &id, ChatSummary &s, std::vector<ChatTurn> *turns, std::string *all_text) {
    std::string text;
    ts5::Json j;
    if (!read_file(path_of(id), text) || !ts5::Json::parse(text, j) || j.type != ts5::Json::OBJ) {
        ts5::log_line("ts5-title: chats: %s.json can't be read", id.c_str());
        return false;
    }
    s.id = id;
    s.title = j["title"].str("A chat");
    s.created = (int64_t)j["created"].num(0);
    s.updated = (int64_t)j["updated"].num(0);
    s.messages = (int)j["turns"].a.size();
    if (all_text) *all_text = lower(s.title);
    for (const ts5::Json &t : j["turns"].a) {
        if (all_text) *all_text += "\n" + lower(t["text"].str());
        if (!turns) continue;
        ChatTurn c;
        c.user = t["user"].boolean(false);
        c.text = t["text"].str();
        c.model = t["model"].str();
        c.tps = t["tps"].num(0);
        c.tokens = (int)t["tokens"].num(0);
        c.end = end_of(t["end"].str());
        c.error = t["error"].str();
        turns->push_back(c);
    }
    return true;
}

void ChatStore::scan() {
    scanned_ = true;
    list_.clear();
    DIR *d = opendir(dir_.c_str());
    if (!d) return;
    std::vector<std::string> ids;
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 5 && n[0] != '.' && n.compare(n.size() - 5, 5, ".json") == 0) ids.push_back(n.substr(0, n.size() - 5));
    }
    closedir(d);
    for (const std::string &id : ids) {
        ChatSummary s;
        if (read(id, s, nullptr, nullptr)) list_.push_back(s);
    }
    std::sort(list_.begin(), list_.end(), newer);
    ts5::log_line("ts5-title: chats: %zu saved", list_.size());
}

std::vector<ChatSummary> ChatStore::list() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!scanned_) scan();
    return list_;
}

std::vector<ChatSummary> ChatStore::search(const std::string &query) {
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
    std::lock_guard<std::mutex> lk(mu_);
    if (!scanned_) scan();
    std::vector<ChatSummary> found;
    for (const ChatSummary &c : list_) {
        ChatSummary s;
        std::string all;
        if (!read(c.id, s, nullptr, &all)) continue;
        bool every = true;
        for (const std::string &w : words) every = every && all.find(w) != std::string::npos;
        if (every) found.push_back(c);
    }
    return found;
}

bool ChatStore::load(const std::string &id, std::vector<ChatTurn> &turns) {
    std::lock_guard<std::mutex> lk(mu_);
    ChatSummary s;
    turns.clear();
    return read(id, s, &turns, nullptr);
}

bool ChatStore::save(std::string &id, const std::vector<ChatTurn> &turns) {
    std::string first;
    std::string body;
    int n = 0;
    for (const ChatTurn &t : turns) {
        if (!t.user && t.end == ChatTurn::Coming) continue;
        if (t.user && first.empty()) first = t.text;
        body += std::string(n++ ? ",\n" : "\n") + "{\"user\":" + (t.user ? "true" : "false") +
                ",\"text\":" + ts5::json_escape(t.text);
        if (!t.user)
            body += ",\"model\":" + ts5::json_escape(t.model) + ",\"tps\":" + std::to_string(t.tps) +
                    ",\"tokens\":" + std::to_string(t.tokens) + ",\"end\":\"" + end_name(t.end) + "\"" +
                    (t.error.empty() ? "" : ",\"error\":" + ts5::json_escape(t.error));
        body += "}";
    }
    if (first.empty()) return true;  // nothing of yours yet: nothing to keep
    std::lock_guard<std::mutex> lk(mu_);
    if (!scanned_) scan();
    const int64_t now = (int64_t)time(nullptr);
    if (id.empty()) id = new_id();
    ChatSummary s;
    s.id = id;
    s.title = chat_title(first);
    s.created = now;
    s.updated = now;
    s.messages = n;
    for (const ChatSummary &c : list_)
        if (c.id == id) s.created = c.created;
    const std::string text = "{\"v\":1,\"title\":" + ts5::json_escape(s.title) + ",\"created\":" +
                             std::to_string(s.created) + ",\"updated\":" + std::to_string(s.updated) +
                             ",\"turns\":[" + body + "\n]}\n";
    const std::string path = path_of(id), tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    bool ok = f && fwrite(text.data(), 1, text.size(), f) == text.size();
    if (f) ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        ts5::log_line("ts5-title: chats: couldn't save %s.json", id.c_str());
        return false;
    }
    list_.erase(std::remove_if(list_.begin(), list_.end(), [&](const ChatSummary &c) { return c.id == id; }),
                list_.end());
    list_.insert(list_.begin(), s);
    version_++;
    return true;
}

bool ChatStore::remove(const std::string &id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (unlink(path_of(id).c_str()) != 0) {
        ts5::log_line("ts5-title: chats: couldn't remove %s.json", id.c_str());
        return false;
    }
    list_.erase(std::remove_if(list_.begin(), list_.end(), [&](const ChatSummary &c) { return c.id == id; }),
                list_.end());
    version_++;
    ts5::log_line("ts5-title: chats: removed %s", id.c_str());
    return true;
}

}  // namespace app
