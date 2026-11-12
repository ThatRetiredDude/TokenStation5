// The GitHub connector (github.h).
#include "github.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "json.h"
#include "net.h"

namespace ts5 {
namespace {

// TokenStation5's GitHub App (github.com/apps/tokenstation5). Its Client ID
// is public: the device flow needs no secret.
const char *kClientId = "Iv23lih5kLX8OesNLJ8n";
const char *kApp = "tokenstation5";

std::string env_or(const char *name, const std::string &def) {
    const char *v = getenv(name);
    return v && *v ? std::string(v) : def;
}
std::string client_id() { return env_or("TS5_GITHUB_CLIENT_ID", kClientId); }
std::string app_slug() { return env_or("TS5_GITHUB_APP", kApp); }
std::string web_base() { return env_or("TS5_GITHUB_WEB", "https://github.com"); }
std::string api_base() { return env_or("TS5_GITHUB_API", "https://api.github.com"); }
std::string install_url() { return web_base() + "/apps/" + app_slug() + "/installations/new"; }

int64_t now() { return (int64_t)time(nullptr); }

Json jnum(double v) {
    Json j;
    j.type = Json::NUM;
    j.n = v;
    char b[32];
    snprintf(b, sizeof b, "%.0f", v);
    j.s = b;
    return j;
}
Json jbool(bool v) {
    Json j;
    j.type = Json::BOOL;
    j.b = v;
    return j;
}

std::string form(const std::string &v) {
    static const char *hex = "0123456789ABCDEF";
    std::string o;
    for (unsigned char c : v) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char)c;
        else {
            o += '%';
            o += hex[c >> 4];
            o += hex[c & 15];
        }
    }
    return o;
}

// The answer's error in plain words, from GitHub's {"message": ...} or the status.
std::string why_failed(const HttpResponse &r) {
    Json j;
    std::string m;
    if (Json::parse(r.body, j) && j.type == Json::OBJ) m = j["message"].str(j["error_description"].str(j["error"].str()));
    if (m.empty()) m = r.status == 0 ? "no answer" : "HTTP " + std::to_string(r.status);
    return m;
}

// ---- The token's file ----

struct Token {
    std::string login, access, refresh;
    int64_t expires = 0, refresh_expires = 0;  // Unix seconds; 0: never
};

bool load(const std::string &path, Token &t, bool &there, std::string &err) {
    there = false;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) {
        if (errno == ENOENT || errno == ENOTDIR) return true;
        err = "can't read " + path + ": " + strerror(errno);
        return false;
    }
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    fclose(f);
    Json j;
    if (!Json::parse(text, j) || j.type != Json::OBJ) {
        err = path + " is not what ts5 wrote there: sign in again";
        return false;
    }
    t.login = j["login"].str();
    t.access = j["access_token"].str();
    t.refresh = j["refresh_token"].str();
    t.expires = (int64_t)j["expires_at"].num(0);
    t.refresh_expires = (int64_t)j["refresh_expires_at"].num(0);
    there = !t.access.empty();
    return true;
}

// The folders above path; the one it is in for its owner only.
void make_parents(const std::string &path) {
    const size_t last = path.rfind('/');
    for (size_t i = 1; i < path.size(); i++)
        if (path[i] == '/') mkdir(path.substr(0, i).c_str(), i == last ? 0700 : 0755);
}

bool save(const std::string &path, const Token &t, std::string &err) {
    make_parents(path);
    Json j = Json::object();
    j.set("login", Json::string(t.login));
    j.set("access_token", Json::string(t.access));
    j.set("expires_at", jnum((double)t.expires));
    j.set("refresh_token", Json::string(t.refresh));
    j.set("refresh_expires_at", jnum((double)t.refresh_expires));
    j.set("app", Json::string(app_slug()));
    const std::string text = j.dump(2) + "\n";
    const std::string tmp = path + ".new";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        err = "can't write " + tmp + ": " + strerror(errno);
        return false;
    }
    fchmod(fd, 0600);
    size_t at = 0;
    bool ok = true;
    while (ok && at < text.size()) {
        const ssize_t w = write(fd, text.data() + at, text.size() - at);
        if (w <= 0) ok = false;
        else at += (size_t)w;
    }
    ok = fsync(fd) == 0 && ok;
    ok = close(fd) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        err = "can't write " + path + ": " + strerror(errno);
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

// One ts5 at a time renews the token (a refresh token works once): held
// while the file is read, renewed and written.
struct Lock {
    int fd = -1;
    explicit Lock(const std::string &path) {
        make_parents(path);
        fd = open((path + ".lock").c_str(), O_RDWR | O_CREAT, 0600);
        if (fd >= 0) flock(fd, LOCK_EX);
    }
    ~Lock() {
        if (fd >= 0) close(fd);
    }
};

// The token from GitHub's answer to a sign-in or a renewal.
bool take_token(const HttpResponse &r, Token &t, std::string &error, std::string &err) {
    Json j;
    if (!Json::parse(r.body, j) || j.type != Json::OBJ) {
        err = "GitHub's answer was not understood (" + std::to_string(r.status) + ")";
        return false;
    }
    error = j["error"].str();
    if (!error.empty()) {
        err = j["error_description"].str(error);
        return false;
    }
    if (j["access_token"].str().empty()) {
        err = "GitHub's answer had no token (" + std::to_string(r.status) + ")";
        return false;
    }
    t.access = j["access_token"].str();
    t.refresh = j["refresh_token"].str();
    const double e = j["expires_in"].num(0), re = j["refresh_token_expires_in"].num(0);
    t.expires = e > 0 ? now() + (int64_t)e : 0;
    t.refresh_expires = re > 0 ? now() + (int64_t)re : 0;
    return true;
}

bool post_form(const std::string &path, const std::string &body, HttpResponse &r, std::string &err) {
    return http_request("POST", web_base() + path,
                        {"Accept: application/json", "Content-Type: application/x-www-form-urlencoded"}, body, r, err, 30);
}

// A token to call the API with, renewed when it has under 5 minutes left
// (or when force: GitHub refused it).
bool fresh_token(Token &t, bool force, std::string &err) {
    const std::string path = github_file();
    Lock lock(path);
    bool there = false;
    if (!load(path, t, there, err)) return false;
    if (!there) {
        err = "not signed in to GitHub: ts5-tui's Connectors page signs in";
        return false;
    }
    if (!force && (t.expires == 0 || now() < t.expires - 300)) return true;
    if (t.refresh.empty() || (t.refresh_expires && now() >= t.refresh_expires)) {
        err = "the GitHub sign-in ran out: sign in again (ts5-tui's Connectors page)";
        return false;
    }
    HttpResponse r;
    if (!post_form("/login/oauth/access_token",
                   "client_id=" + form(client_id()) + "&grant_type=refresh_token&refresh_token=" + form(t.refresh), r,
                   err)) {
        err = "GitHub couldn't be reached to renew the sign-in: " + err;
        return false;
    }
    Token n;
    std::string error;
    if (!take_token(r, n, error, err)) {
        if (error == "bad_refresh_token" || error == "incorrect_client_credentials" || error == "unauthorized")
            err = "GitHub ended this sign-in (" + error + "): sign in again (ts5-tui's Connectors page)";
        else
            err = "GitHub didn't renew the sign-in: " + err;
        return false;
    }
    n.login = t.login;
    if (n.refresh.empty()) {  // (an app whose tokens don't run out)
        n.refresh = t.refresh;
        n.refresh_expires = t.refresh_expires;
    }
    if (!save(path, n, err)) return false;
    t = n;
    return true;
}

bool path_ok(const std::string &p) {
    if (p.empty() || p[0] != '/' || p.compare(0, 2, "//") == 0) return false;
    for (unsigned char c : p)
        if (c <= ' ' || c == 0x7f || c == '\\' || c == '#') return false;
    return p.find("://") == std::string::npos;
}

// A call to the API as the signed-in user. A token GitHub refuses is
// renewed once and the call made again.
bool api(const std::string &method, const std::string &path, const std::string &body, HttpResponse &r,
         std::string &err) {
    if (!path_ok(path)) {
        err = "not a GitHub API path: " + path + " (one such as /user/repos)";
        return false;
    }
    Token t;
    for (int round = 0; round < 2; round++) {
        if (!fresh_token(t, round == 1, err)) return false;
        std::vector<std::string> h = {"Authorization: Bearer " + t.access, "Accept: application/vnd.github+json",
                                      "X-GitHub-Api-Version: 2022-11-28"};
        if (!body.empty()) h.push_back("Content-Type: application/json");
        r = HttpResponse();
        if (!http_request(method, api_base() + path, h, body, r, err, 60)) {
            err = "GitHub couldn't be reached: " + err;
            return false;
        }
        if (r.status != 401 || t.refresh.empty()) break;
    }
    return true;
}

// The next page's path, from the answer's Link header.
std::string next_page(const std::string &head) {
    std::string lower = head;
    for (auto &c : lower) c = (char)tolower((unsigned char)c);
    const size_t at = lower.find("\nlink:");
    if (at == std::string::npos) return "";
    const size_t end = head.find('\n', at + 1);
    const std::string line = head.substr(at + 6, end == std::string::npos ? std::string::npos : end - at - 6);
    size_t p = 0;
    while ((p = line.find('<', p)) != std::string::npos) {
        const size_t q = line.find('>', p);
        if (q == std::string::npos) break;
        const size_t semi = line.find(',', q);
        const std::string attrs = line.substr(q, semi == std::string::npos ? std::string::npos : semi - q);
        if (attrs.find("rel=\"next\"") != std::string::npos) {
            std::string url = line.substr(p + 1, q - p - 1);
            const std::string base = api_base();
            if (url.compare(0, base.size(), base) == 0) url = url.substr(base.size());
            return url;
        }
        p = q;
    }
    return "";
}

// ---- --lean ----

int b64v(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

bool unbase64(const std::string &in, std::string &out) {
    out.clear();
    unsigned v = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        const int d = b64v(c);
        if (d < 0) return false;
        v = (v << 6) | (unsigned)d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char)((v >> bits) & 0xff);
        }
    }
    return true;
}

std::string base64(const std::string &in) {
    static const char *abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const unsigned v = ((unsigned char)in[i] << 16) | ((unsigned char)in[i + 1] << 8) | (unsigned char)in[i + 2];
        out += abc[v >> 18], out += abc[(v >> 12) & 63], out += abc[(v >> 6) & 63], out += abc[v & 63];
    }
    if (i + 1 == in.size()) {
        const unsigned v = (unsigned char)in[i] << 16;
        out += abc[v >> 18], out += abc[(v >> 12) & 63], out += "==";
    } else if (i + 2 == in.size()) {
        const unsigned v = ((unsigned char)in[i] << 16) | ((unsigned char)in[i + 1] << 8);
        out += abc[v >> 18], out += abc[(v >> 12) & 63], out += abc[(v >> 6) & 63], out += '=';
    }
    return out;
}

bool utf8_text(const std::string &s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        int n = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : -1;
        if (n < 0 || (c < 0x20 && c != '\n' && c != '\r' && c != '\t')) return false;
        if (i + (size_t)n >= s.size() && n > 0) return false;
        for (int k = 1; k <= n; k++)
            if (((unsigned char)s[i + (size_t)k] >> 6) != 2) return false;
        i += (size_t)n + 1;
    }
    return true;
}

void lean(Json &j) {
    if (j.type == Json::ARR) {
        for (Json &e : j.a) lean(e);
        return;
    }
    if (j.type != Json::OBJ) return;
    // A user or an organization: its login.
    if (j["login"].type == Json::STR && !j["avatar_url"].is_null() && !j["type"].is_null()) {
        const std::string login = j["login"].str();
        j = Json::string(login);
        return;
    }
    // A file: its text.
    if (j["encoding"].str() == "base64" && j["content"].type == Json::STR) {
        std::string text;
        if (unbase64(j["content"].str(), text)) {
            Json &c = j.o["content"];
            Json &e = j.o["encoding"];
            if (utf8_text(text)) {
                c = Json::string(text);
                e = Json::string("text");
            } else {
                c = Json::string("(" + std::to_string(text.size()) + " bytes, not text)");
                e = Json::string("binary");
            }
        }
    }
    std::vector<std::string> keep;
    for (const std::string &k : j.keys) {
        const bool url = k == "url" || (k.size() > 4 && k.compare(k.size() - 4, 4, "_url") == 0 && k != "html_url");
        if (url || k == "node_id" || k == "_links" || k == "gravatar_id")
            j.o.erase(k);
        else
            keep.push_back(k);
    }
    j.keys = keep;
    for (const std::string &k : keep) lean(j.o[k]);
}

// ---- The commands ----

int sign_in() {
    const std::string path = github_file();
    HttpResponse r;
    std::string err;
    if (!post_form("/login/device/code", "client_id=" + form(client_id()), r, err)) {
        printf("error GitHub couldn't be reached: %s\n", err.c_str());
        return 1;
    }
    Json j;
    if (r.status != 200 || !Json::parse(r.body, j) || j.type != Json::OBJ || j["device_code"].str().empty()) {
        const std::string e = Json::parse(r.body, j) && j.type == Json::OBJ ? j["error"].str() : "";
        if (e == "device_flow_disabled")
            printf("error the GitHub app has its device flow off (its settings: Enable Device Flow)\n");
        else
            printf("error GitHub didn't give a code: %s\n", why_failed(r).c_str());
        return 1;
    }
    const std::string device = j["device_code"].str();
    double interval = j["interval"].num(5);
    const double expires = j["expires_in"].num(900);
    printf("code %s\nurl %s\nexpires %.0f\n", j["user_code"].str().c_str(),
           j["verification_uri"].str(web_base() + "/login/device").c_str(), expires);
    fflush(stdout);
    const int64_t until = now() + (int64_t)expires;
    Token t;
    while (true) {
        for (double w = 0; w < interval; w += 0.25) usleep(250000);
        if (now() >= until) {
            printf("expired\n");
            return 1;
        }
        HttpResponse p;
        if (!post_form("/login/oauth/access_token",
                       "client_id=" + form(client_id()) + "&device_code=" + form(device) +
                           "&grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code",
                       p, err))
            continue;  // (a dropped connection: the next round asks again)
        std::string error;
        if (take_token(p, t, error, err)) break;
        if (error == "authorization_pending") continue;
        if (error == "slow_down") {
            Json s;
            interval = Json::parse(p.body, s) && s["interval"].num(0) > 0 ? s["interval"].num(0) : interval + 5;
            continue;
        }
        if (error == "access_denied") {
            printf("denied\n");
            return 1;
        }
        if (error == "expired_token" || error == "token_expired") {
            printf("expired\n");
            return 1;
        }
        printf("error %s\n", err.c_str());
        return 1;
    }
    // Whose it is, and where the app is installed.
    {
        HttpResponse u;
        Json uj;
        if (http_request("GET", api_base() + "/user",
                         {"Authorization: Bearer " + t.access, "Accept: application/vnd.github+json",
                          "X-GitHub-Api-Version: 2022-11-28"},
                         "", u, err, 30) &&
            u.status == 200 && Json::parse(u.body, uj))
            t.login = uj["login"].str();
    }
    {
        Lock lock(path);
        if (!save(path, t, err)) {
            printf("error %s\n", err.c_str());
            return 1;
        }
    }
    printf("signed-in %s\n", t.login.empty() ? "?" : t.login.c_str());
    fflush(stdout);
    HttpResponse in;
    Json ij;
    if (api("GET", "/user/installations", "", in, err) && in.status == 200 && Json::parse(in.body, ij))
        printf("installations %.0f\n", ij["total_count"].num(0));
    return 0;
}

int status(bool repos) {
    const std::string path = github_file();
    Token t;
    bool there = false;
    std::string err;
    Json j = Json::object();
    j.set("app", Json::string(app_slug()));
    j.set("install_url", Json::string(install_url()));
    j.set("file", Json::string(path));
    {
        Lock lock(path);
        if (!load(path, t, there, err)) j.set("error", Json::string(err));
    }
    j.set("signed_in", jbool(there));
    if (there) {
        j.set("login", Json::string(t.login));
        if (t.refresh_expires) j.set("ends_in", jnum((double)(t.refresh_expires - now())));
    }
    if (there && repos) {
        HttpResponse r;
        Json ins, list;
        list.type = Json::ARR;
        if (!api("GET", "/user/installations?per_page=100", "", r, err)) {
            j.set("error", Json::string(err));
        } else if (r.status != 200 || !Json::parse(r.body, ins)) {
            j.set("error", Json::string("GitHub: " + why_failed(r)));
        } else {
            for (const Json &i : ins["installations"].a) {
                Json o = Json::object();
                o.set("account", Json::string(i["account"]["login"].str()));
                o.set("all", jbool(i["repository_selection"].str() == "all"));
                const std::string id = i["id"].s;
                Json names;
                names.type = Json::ARR;
                HttpResponse rr;
                Json rj;
                if (api("GET", "/user/installations/" + id + "/repositories?per_page=100", "", rr, err) &&
                    rr.status == 200 && Json::parse(rr.body, rj)) {
                    for (const Json &repo : rj["repositories"].a) {
                        Json n = Json::object();
                        n.set("name", Json::string(repo["full_name"].str()));
                        n.set("private", jbool(repo["private"].boolean(false)));
                        names.a.push_back(n);
                    }
                    o.set("count", jnum(rj["total_count"].num((double)names.a.size())));
                }
                o.set("repos", names);
                list.a.push_back(o);
            }
            j.set("installations", list);
        }
    }
    printf("%s\n", j.dump().c_str());
    return 0;
}

int sign_out() {
    const std::string path = github_file();
    Lock lock(path);
    if (unlink(path.c_str()) != 0 && errno != ENOENT) {
        fprintf(stderr, "ts5 github: can't remove %s: %s\n", path.c_str(), strerror(errno));
        return 1;
    }
    printf("signed out: this machine forgot its GitHub token. GitHub still lists the app under your account's "
           "Applications (%s/settings/apps/authorizations), where it can be revoked.\n",
           web_base().c_str());
    return 0;
}

int call(const std::string &method, const std::string &path, std::string body, bool want_lean) {
    if (body == "-") {
        body.clear();
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, stdin)) > 0) body.append(buf, n);
    }
    if (!body.empty()) {
        Json b;
        if (!Json::parse(body, b)) {
            printf("error the body is not JSON\n");
            return 2;
        }
        // A file written with its text (a model can't write base64): the
        // API's "content" made from "text".
        if (want_lean && b.type == Json::OBJ && path.find("/contents/") != std::string::npos &&
            b["text"].type == Json::STR && b["content"].is_null()) {
            b.set("content", Json::string(base64(b["text"].str())));
            b.o.erase("text");
            std::vector<std::string> keys;
            for (const std::string &k : b.keys)
                if (k != "text") keys.push_back(k);
            b.keys = keys;
            body = b.dump();
        }
    }
    HttpResponse r;
    std::string err;
    if (!api(method, path, body, r, err)) {
        printf("error %s\n", err.c_str());
        return 2;
    }
    printf("status %d\n", r.status);
    const std::string next = next_page(r.head);
    if (!next.empty()) printf("next %s\n", next.c_str());
    if (!want_lean) {
        printf("%s\n", r.body.c_str());
    } else {
        printf("%s\n", github_lean(r.body).c_str());
    }
    return r.status >= 200 && r.status < 300 ? 0 : 1;
}

}  // namespace

std::string github_file() {
    if (const char *f = getenv("TS5_GITHUB_FILE"); f && *f) return f;
    if (const char *d = getenv("TS5_DATA_DIR"); d && *d) return std::string(d) + "/connectors/github.json";
    if (const char *x = getenv("XDG_DATA_HOME"); x && *x == '/') return std::string(x) + "/tokenstation/connectors/github.json";
    const char *h = getenv("HOME");
    return std::string(h && *h ? h : ".") + "/.local/share/tokenstation/connectors/github.json";
}

std::string github_lean(const std::string &body) {
    Json j;
    if (!Json::parse(body, j)) return body;
    lean(j);
    return j.dump();
}

int github_main(int argc, char **argv) {
    std::vector<std::string> a;
    bool want_lean = false, repos = false;
    for (int i = 2; i < argc; i++) {
        const std::string s = argv[i];
        if (s == "--lean") want_lean = true;
        else if (s == "--repos") repos = true;
        else a.push_back(s);
    }
    const auto usage = [] {
        fprintf(stderr, "ts5 github sign-in | status [--repos] | sign-out | api METHOD PATH [BODY|-] [--lean]\n"
                        "  (the token: %s)\n",
                github_file().c_str());
        return 2;
    };
    if (a.empty()) return usage();
    if (a[0] == "sign-in" && a.size() == 1) return sign_in();
    if (a[0] == "status" && a.size() == 1) return status(repos);
    if (a[0] == "sign-out" && a.size() == 1) return sign_out();
    if (a[0] == "api" && (a.size() == 3 || a.size() == 4)) {
        std::string m = a[1];
        for (auto &c : m) c = (char)toupper((unsigned char)c);
        if (m != "GET" && m != "POST" && m != "PATCH" && m != "PUT" && m != "DELETE") return usage();
        return call(m, a[2], a.size() == 4 ? a[3] : "", want_lean);
    }
    return usage();
}

}  // namespace ts5
