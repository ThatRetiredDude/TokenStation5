// Feedback reports (feedback.h).
#include "feedback.h"

#include <pwd.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "net.h"

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif

namespace ts5 {

namespace {

// Where reports go: TokenStation's own feedback Worker. It only takes
// reports in; nobody can read them back through it.
const char *kFeedbackUrl = "https://ts5-feedback.tokenstation.workers.dev/v1/report";
const size_t kMaxReport = 200u << 10;  // the Worker takes up to 256 KiB
const int kLogLines = 300;
const int kChatMessages = 40;

std::string url() {
    const char *e = getenv("TS5_FEEDBACK_URL");
    return e && *e ? e : kFeedbackUrl;
}

bool slurp(const std::string &path, std::string &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream s;
    s << f.rdbuf();
    out = s.str();
    return true;
}

std::string run(const std::string &cmd) {
    std::string out;
    if (FILE *p = popen((cmd + " 2>/dev/null").c_str(), "r")) {
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
        pclose(p);
    }
    return out;
}

std::string quote(const std::string &s) {
    std::string r = "'";
    for (char c : s) r += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return r + "'";
}

std::string tail_lines(const std::string &text, int n) {
    size_t pos = text.size();
    if (pos && text[pos - 1] == '\n') pos--;
    int seen = 0;
    while (pos > 0) {
        if (text[pos - 1] == '\n' && ++seen == n) break;
        pos--;
    }
    return text.substr(pos);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool word_char(unsigned char c) { return std::isalnum(c) || c == '_' || c == '-' || c == '.'; }

// Every whole-word copy of word (letters, digits, _ - . around it don't
// count as a boundary) replaced by with.
void replace_word(std::string &s, const std::string &word, const std::string &with, bool fold_case) {
    if (word.size() < 2) return;
    const std::string hay = fold_case ? lower(s) : s, needle = fold_case ? lower(word) : word;
    std::string out;
    size_t from = 0, at;
    while ((at = hay.find(needle, from)) != std::string::npos) {
        const size_t end = at + needle.size();
        const bool left = at == 0 || !word_char((unsigned char)hay[at - 1]);
        // A trailing dot ends a sentence, not a name.
        const bool right = end == hay.size() || !word_char((unsigned char)hay[end]) ||
                           (hay[end] == '.' && (end + 1 == hay.size() || !word_char((unsigned char)hay[end + 1])));
        out.append(s, from, at - from);
        out += left && right ? with : s.substr(at, needle.size());
        from = end;
    }
    out.append(s, from, std::string::npos);
    s = out;
}

void replace_all(std::string &s, const std::string &a, const std::string &b) {
    if (a.empty()) return;
    size_t at = 0;
    while ((at = s.find(a, at)) != std::string::npos) {
        s.replace(at, a.size(), b);
        at += b.size();
    }
}

// PEM blocks: "-----BEGIN ... PRIVATE KEY-----" to its END line (or the
// end of the text when it was cut).
void drop_pem(std::string &s) {
    size_t at = 0;
    while ((at = s.find("-----BEGIN ", at)) != std::string::npos) {
        const size_t eol = s.find("-----", at + 11);
        if (eol == std::string::npos) break;
        const std::string what = s.substr(at + 11, eol - at - 11);
        if (what.find("PRIVATE KEY") == std::string::npos && what.find("OPENSSH") == std::string::npos &&
            what.find("PGP PRIVATE") == std::string::npos) {
            at = eol;
            continue;
        }
        size_t end = s.find("-----END ", eol);
        if (end != std::string::npos) end = s.find("-----", end + 9);
        end = end == std::string::npos ? s.size() : end + 5;
        s.replace(at, end - at, "<private key removed>");
        at += 21;
    }
}

bool valid_octets(const std::smatch &m, int o[4]) {
    for (int i = 0; i < 4; i++) {
        const std::string p = m[i + 1].str();
        if (p.size() > 1 && p[0] == '0') return false;  // 01.2.3.4: a version, not an address
        o[i] = atoi(p.c_str());
        if (o[i] > 255) return false;
    }
    return true;
}

std::string ipv4_kind(const int o[4]) {
    if (o[0] == 100 && o[1] >= 64 && o[1] < 128) return "<tailnet-ip>";
    if (o[0] == 10 || (o[0] == 172 && o[1] >= 16 && o[1] < 32) || (o[0] == 192 && o[1] == 168) ||
        (o[0] == 169 && o[1] == 254))
        return "<lan-ip>";
    return "<ip>";
}

std::string regex_replace_fn(const std::string &s, const std::regex &re,
                             const std::function<std::string(const std::smatch &)> &fn) {
    std::string out;
    auto begin = std::sregex_iterator(s.begin(), s.end(), re), end = std::sregex_iterator();
    size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        const std::smatch &m = *it;
        out.append(s, last, (size_t)m.position(0) - last);
        out += fn(m);
        last = (size_t)(m.position(0) + m.length(0));
    }
    out.append(s, last, std::string::npos);
    return out;
}

// A long run of letters and digits that looks random: upper and lower case
// and digits mixed (a hash in lowercase hex is left: it names a file).
bool looks_random(const std::string &t) {
    bool up = false, lo = false, dg = false;
    for (unsigned char c : t) up |= (bool)std::isupper(c), lo |= (bool)std::islower(c), dg |= (bool)std::isdigit(c);
    return up && lo && dg;
}

}  // namespace

ScrubWords feedback_scrub_words() {
    ScrubWords w;
    auto add_user = [&](const char *u) {
        if (!u || !*u || !strcmp(u, "root")) return;
        if (std::find(w.users.begin(), w.users.end(), u) == w.users.end()) w.users.push_back(u);
    };
    if (struct passwd *pw = getpwuid(getuid())) {
        add_user(pw->pw_name);
        if (pw->pw_dir) w.home = pw->pw_dir;
    }
    add_user(getenv("USER"));
    add_user(getenv("LOGNAME"));
    add_user(getenv("SUDO_USER"));
    if (const char *h = getenv("HOME"); h && *h && w.home.empty()) w.home = h;
    char host[256] = {0};
    if (gethostname(host, sizeof host - 1) == 0) w.host = host;
    if (w.host == "localhost") w.host.clear();
    return w;
}

std::string feedback_scrub(const std::string &text, const ScrubWords &w, bool keep_emails) {
    std::string s = text;
    drop_pem(s);

    // Tokens with a known shape.
    static const std::regex tokens(
        R"((ts5-[a-z0-9]{5}(?:-[a-z0-9]{5}){2,})|)"
        R"(((?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,})|)"
        R"((tskey-[A-Za-z0-9-]{8,})|)"
        R"((hf_[A-Za-z0-9]{20,})|)"
        R"((tvly-[A-Za-z0-9-]{10,})|)"
        R"(((?:sk|pk|rk)-[A-Za-z0-9_-]{16,})|)"
        R"((xox[abprs]-[A-Za-z0-9-]{10,})|)"
        R"((AKIA[0-9A-Z]{16})|)"
        R"((AIza[0-9A-Za-z_-]{30,})|)"
        R"((eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}))");
    s = regex_replace_fn(s, tokens, [](const std::smatch &m) -> std::string {
        if (m[1].matched) return "<ts5-key>";
        if (m[2].matched) return "<github-token>";
        if (m[3].matched) return "<tailscale-key>";
        return "<token>";
    });

    // "Bearer X", "Authorization: X", and name=value / "name": "value"
    // where the name says key, token, secret or password.
    static const std::regex bearer(R"(((?:Bearer|Basic|token)\s+)[A-Za-z0-9._~+/=-]{8,})", std::regex::icase);
    s = std::regex_replace(s, bearer, "$1<hidden>");
    static const std::regex auth(R"(((?:authorization|cookie|set-cookie|x-api-key)\s*:\s*)[^\r\n]+)", std::regex::icase);
    s = std::regex_replace(s, auth, "$1<hidden>");
    static const std::regex named(
        R"((["']?[A-Za-z0-9_.-]*(?:key|token|secret|passw(?:or)?d|pwd|auth|cookie|credential)[A-Za-z0-9_.-]*["']?\s*[:=]\s*)("[^"\r\n]*"|'[^'\r\n]*'|[^\s,;}&]+))",
        std::regex::icase);
    s = regex_replace_fn(s, named, [](const std::smatch &m) -> std::string {
        const std::string v = m[2].str();
        const std::string bare = v.size() >= 2 && (v[0] == '"' || v[0] == '\'') ? v.substr(1, v.size() - 2) : v;
        // Nothing there, or a yes/no: nothing to hide.
        const std::string lv = lower(bare);
        if (bare.empty() || lv == "true" || lv == "false" || lv == "on" || lv == "off" || lv == "0" || lv == "1" ||
            lv == "none" || lv == "null" || lv == "<hidden>" || (lv.size() > 1 && lv[0] == '<' && lv.back() == '>'))
            return m[0].str();
        const char q = v[0] == '"' || v[0] == '\'' ? v[0] : 0;
        return m[1].str() + (q ? std::string(1, q) + "<hidden>" + q : "<hidden>");
    });

    // A password in an address: scheme://user:pass@host.
    static const std::regex userinfo(R"(([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@)", std::regex::icase);
    s = std::regex_replace(s, userinfo, "$1<hidden>@");

    static const std::regex email(R"([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})");
    if (!keep_emails) s = std::regex_replace(s, email, "<email>");

    static const std::regex tsnet(R"([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.ts\.net)", std::regex::icase);
    s = std::regex_replace(s, tsnet, "<tailnet-name>");

    static const std::regex mac(R"(\b[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}\b)");
    s = std::regex_replace(s, mac, "<mac>");

    static const std::regex ipv4(R"(\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b)");
    {
        const std::string in = s;
        s = regex_replace_fn(in, ipv4, [&in](const std::smatch &m) -> std::string {
            const size_t at = (size_t)m.position(0), end = at + (size_t)m.length(0);
            // Part of a longer dotted run (a version such as 1.2.3.4.5).
            if ((at > 0 && in[at - 1] == '.') || (end + 1 < in.size() && in[end] == '.' && std::isdigit((unsigned char)in[end + 1])))
                return m[0].str();
            int o[4];
            if (!valid_octets(m, o)) return m[0].str();
            // 0.x.y.z is a version number (ts5 0.0.1.200); 127.x and the
            // broadcast address say nothing about anyone.
            if (o[0] == 0 || o[0] == 127 || (o[0] == 255 && o[1] == 255)) return m[0].str();
            return ipv4_kind(o);
        });
    }

    // IPv6: hex groups with colons, at least one letter or a "::", so a
    // time (10:05:23) stays.
    static const std::regex ipv6(R"((?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4})");
    s = regex_replace_fn(s, ipv6, [](const std::smatch &m) -> std::string {
        const std::string t = m[0].str();
        const int colons = (int)std::count(t.begin(), t.end(), ':');
        const bool dbl = t.find("::") != std::string::npos;
        bool letter = false;
        for (unsigned char c : t) letter |= (bool)std::isalpha(c);
        if (colons < 2 || (!dbl && colons < 3) || (!letter && !dbl) || t == "::" || t == "::1") return t;
        if (!dbl && colons < 5) return t;  // a:b:c:d is too short to be an address on its own
        const std::string lt = lower(t);
        if (lt.rfind("fd7a:115c:a1e0", 0) == 0) return "<tailnet-ip>";
        if (lt.rfind("fe80", 0) == 0 || lt.rfind("fd", 0) == 0 || lt.rfind("fc", 0) == 0) return "<lan-ip>";
        return "<ip>";
    });

    // Long random-looking runs (a key in a shape we don't know).
    static const std::regex longrun(R"([A-Za-z0-9_+/=-]{32,})");
    s = regex_replace_fn(s, longrun, [](const std::smatch &m) -> std::string {
        const std::string t = m[0].str();
        if (t.find('/') != std::string::npos && t.find_first_of("0123456789") == std::string::npos) return t;
        return looks_random(t) ? "<secret-like text removed>" : t;
    });

    // The home folder, any /home/NAME or /Users/NAME, the login names and
    // this machine's name.
    if (!w.home.empty() && w.home != "/" && w.home != "/root") replace_all(s, w.home, "~");
    static const std::regex homes(R"((/home/|/Users/)[^/\s'"]+)");
    s = std::regex_replace(s, homes, "$1<user>");
    for (const auto &u : w.users) replace_word(s, u, "<user>", false);
    if (!w.host.empty()) {
        replace_word(s, w.host, "<this-machine>", true);
        // The short name too: ps5-living.example.lan -> ps5-living.
        const size_t dot = w.host.find('.');
        if (dot != std::string::npos && dot >= 3) replace_word(s, w.host.substr(0, dot), "<this-machine>", true);
    }
    return s;
}

namespace {

struct Prep {
    std::string summary, log, journal, settings, chat;
    bool no_logs = false;
};

std::string first_line_of(const std::string &path, const std::string &key) {
    std::string t;
    if (!slurp(path, t)) return "";
    std::istringstream in(t);
    std::string l;
    while (std::getline(in, l))
        if (l.rfind(key, 0) == 0) {
            std::string v = l.substr(key.size());
            v.erase(0, v.find_first_not_of(" \t:="));
            if (v.size() >= 2 && v[0] == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
            return v;
        }
    return "";
}

std::string system_facts() {
    std::string r;
    struct utsname u {};
    if (uname(&u) == 0) r += std::string("- Kernel: ") + u.release + " (" + u.machine + ")\n";
    const std::string os = first_line_of("/etc/os-release", "PRETTY_NAME=");
    if (!os.empty()) r += "- System: " + os + "\n";
    const std::string cpu = first_line_of("/proc/cpuinfo", "model name");
    if (!cpu.empty()) r += "- Processor: " + cpu + "\n";
    const double mt = atof(first_line_of("/proc/meminfo", "MemTotal").c_str()) / 1048576.0,
                 ma = atof(first_line_of("/proc/meminfo", "MemAvailable").c_str()) / 1048576.0;
    if (mt > 0) {
        char b[96];
        snprintf(b, sizeof b, "- Memory: %.1f GiB free of %.1f GiB\n", ma, mt);
        r += b;
    }
    std::string gpu = run("lspci 2>/dev/null | grep -iE 'vga|display|3d' | head -3");
    while (!gpu.empty() && gpu.back() == '\n') gpu.pop_back();
    if (!gpu.empty()) {
        replace_all(gpu, "\n", "; ");
        r += "- Graphics: " + gpu + "\n";
    }
    std::string mesa = run("dpkg-query -W -f='${Version}' mesa-vulkan-drivers");
    if (!mesa.empty()) r += "- Mesa: " + mesa + "\n";
    std::string disk = run("df -h --output=avail,size / | tail -1 | awk '{print $1 \" free of \" $2}'");
    while (!disk.empty() && (disk.back() == '\n' || disk.back() == ' ')) disk.pop_back();
    if (!disk.empty()) r += "- Disk: " + disk + "\n";
    return r;
}

// tui.conf as name = value lines; anything named like a key, token,
// secret or password says only whether it is set.
std::string settings_text(const std::string &path) {
    std::string t;
    if (!slurp(path, t)) return "";
    std::istringstream in(t);
    std::string l, r;
    static const std::regex secret_name(R"(key|token|secret|passw|pwd|auth|cookie|credential)", std::regex::icase);
    while (std::getline(in, l)) {
        const size_t eq = l.find('=');
        if (eq != std::string::npos && std::regex_search(l.substr(0, eq), secret_name)) {
            std::string v = l.substr(eq + 1);
            v.erase(0, v.find_first_not_of(" \t"));
            r += l.substr(0, eq + 1) + (v.empty() ? "" : " <set>") + "\n";
        } else {
            r += l + "\n";
        }
    }
    return r;
}

// A saved chat (the TV app's format, chats/ID.json): its last messages.
std::string chat_text(const std::string &path) {
    std::string t;
    if (!slurp(path, t)) return "";
    Json j;
    if (!Json::parse(t, j)) return "";
    const Json &msgs = j["messages"];
    std::string r;
    const size_t from = msgs.a.size() > (size_t)kChatMessages ? msgs.a.size() - kChatMessages : 0;
    if (from) r += "(" + std::to_string(from) + " earlier messages left out)\n\n";
    for (size_t i = from; i < msgs.a.size(); i++) {
        const Json &m = msgs.a[i];
        std::string text = m["content"].str();
        if (text.size() > 4000) text = text.substr(0, 4000) + " …(cut)";
        r += "**" + m["role"].str("?") + ":** " + text + "\n\n";
    }
    return r;
}

std::string prepare(const Prep &p) {
    // The user's own words keep an email address: it is how they can be
    // answered. Everything else loses them.
    const ScrubWords w = feedback_scrub_words();
    const std::string head = feedback_scrub(
        "## What happened\n\n" + (p.summary.empty() ? std::string("(no words given)") : p.summary) + "\n\n", w, true);
    std::string r = "## This machine\n\n- ts5: " TS5_VERSION "\n";
    const std::string tui = run("ts5-tui --version");
    if (!tui.empty()) r += "- " + tail_lines(tui, 1);
    if (r.back() != '\n') r += '\n';
    r += system_facts() + "\n";
    if (!p.settings.empty()) {
        const std::string s = settings_text(p.settings);
        if (!s.empty()) r += "## Settings\n\n```\n" + s + "```\n\n";
    }
    if (!p.no_logs) {
        std::string log;
        if (!p.log.empty()) {
            std::string all;
            if (slurp(p.log, all)) log = tail_lines(all, kLogLines);
        } else if (!p.journal.empty()) {
            log = run("journalctl -u " + quote(p.journal) + " -n " + std::to_string(kLogLines) + " -o cat --no-pager");
        }
        if (!log.empty()) {
            replace_all(log, "```", "'''");
            r += "## Log (last " + std::to_string(kLogLines) + " lines)\n\n```\n" + log + (log.back() == '\n' ? "" : "\n") +
                 "```\n\n";
        }
    }
    if (!p.chat.empty()) {
        std::string c = chat_text(p.chat);
        if (!c.empty()) r += "## The chat\n\n" + c;
    }
    r = head + feedback_scrub(r, w, false);
    // Too long: the log's oldest lines go first.
    if (r.size() > kMaxReport) {
        const size_t lg = r.find("## Log");
        if (lg != std::string::npos) {
            const size_t start = r.find("```\n", lg) + 4, over = r.size() - kMaxReport + 64;
            if (start + over < r.size()) r.replace(start, over, "…(older lines left out)\n");
        }
        if (r.size() > kMaxReport) r = r.substr(0, kMaxReport) + "\n…(cut)\n";
    }
    return r;
}

void print_json_error(const std::string &why) {
    Json j = Json::object();
    j.set("error", Json::string(why));
    printf("%s\n", j.dump().c_str());
}

int send(const std::string &path) {
    std::string body;
    if (!slurp(path, body)) {
        print_json_error("can't read " + path);
        return 1;
    }
    {
        const ScrubWords w = feedback_scrub_words();
        const size_t m = body.find("\n## This machine");
        body = m == std::string::npos ? feedback_scrub(body, w, false)
                                      : feedback_scrub(body.substr(0, m), w, true) + feedback_scrub(body.substr(m), w, false);
    }
    if (body.size() > kMaxReport) body = body.substr(0, kMaxReport) + "\n…(cut)\n";
    std::string summary;
    const size_t h = body.find("## What happened\n\n");
    if (h != std::string::npos) {
        const size_t s0 = h + 18, e = body.find('\n', s0);
        summary = body.substr(s0, e == std::string::npos ? std::string::npos : e - s0);
    }
    if (summary.size() > 200) summary = summary.substr(0, 200);
    Json j = Json::object();
    j.set("app", Json::string("ts5-tui"));
    j.set("version", Json::string(TS5_VERSION));
    j.set("summary", Json::string(summary));
    j.set("body", Json::string(body));
    HttpResponse r;
    std::string err;
    if (!http_post(url(), {"Content-Type: application/json"}, j.dump(), r, err, 30)) {
        print_json_error(err.empty() ? "the feedback service couldn't be reached" : err);
        return 1;
    }
    Json a;
    if (r.status == 200 || r.status == 201) {
        if (Json::parse(r.body, a) && !a["id"].str().empty()) {
            Json o = Json::object();
            Json yes;
            yes.type = Json::BOOL;
            yes.b = true;
            o.set("ok", yes);
            o.set("id", Json::string(a["id"].str()));
            printf("%s\n", o.dump().c_str());
            return 0;
        }
    }
    std::string why = Json::parse(r.body, a) ? a["error"].str() : "";
    if (why.empty()) why = r.status == 429 ? "too many reports from here: try again in an hour"
                                           : "the feedback service answered " + std::to_string(r.status);
    print_json_error(why);
    return 1;
}

int usage() {
    fprintf(stderr,
            "usage: ts5 feedback prepare [--summary TEXT] [--log FILE | --journal UNIT] [--settings FILE]\n"
            "                            [--chat FILE] [--no-logs]\n"
            "       ts5 feedback scrub [--keep-emails] < TEXT\n"
            "       ts5 feedback send FILE\n");
    return 2;
}

}  // namespace

int feedback_main(int argc, char **argv) {
    // argv: ts5 feedback SUB ...
    if (argc < 3) return usage();
    const std::string sub = argv[2];
    if (sub == "scrub") {
        const bool keep = argc > 3 && std::string(argv[3]) == "--keep-emails";
        std::ostringstream s;
        s << std::cin.rdbuf();
        fputs(feedback_scrub(s.str(), feedback_scrub_words(), keep).c_str(), stdout);
        return 0;
    }
    if (sub == "send") return argc == 4 ? send(argv[3]) : usage();
    if (sub != "prepare") return usage();
    Prep p;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        auto val = [&](std::string &to) {
            if (i + 1 >= argc) return false;
            to = argv[++i];
            return true;
        };
        if (a == "--no-logs") p.no_logs = true;
        else if (!((a == "--summary" && val(p.summary)) || (a == "--log" && val(p.log)) ||
                   (a == "--journal" && val(p.journal)) || (a == "--settings" && val(p.settings)) ||
                   (a == "--chat" && val(p.chat))))
            return usage();
    }
    fputs(prepare(p).c_str(), stdout);
    return 0;
}

}  // namespace ts5
