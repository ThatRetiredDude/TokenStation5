// Sharing this machine over Tailscale: see tsshare.h.
#include "tsshare.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "access.h"
#include "json.h"
#include "net.h"

namespace ts5 {
namespace {

std::string api_base() {
    const char *e = getenv("TS5_TAILSCALE_API");
    return e && *e ? e : "https://api.tailscale.com";
}

std::string dir_of(const std::string &p) {
    const size_t s = p.rfind('/');
    return s == std::string::npos ? "." : s == 0 ? "/" : p.substr(0, s);
}

bool read_all(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    char buf[4096];
    size_t n;
    out.clear();
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f);
    return true;
}

// Written beside its place, 0600, and moved over it.
bool write_private(const std::string &path, const std::string &text, std::string &err) {
    const std::string tmp = path + ".new";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        err = "can't write " + tmp + ": " + strerror(errno);
        return false;
    }
    fchmod(fd, 0600);
    bool ok = true;
    for (size_t at = 0; ok && at < text.size();) {
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

void make_dirs(const std::string &dir) {
    for (size_t i = 1; i <= dir.size(); i++)
        if (i == dir.size() || dir[i] == '/') mkdir(dir.substr(0, i).c_str(), 0700);
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
    return s.substr(i);
}

std::string invites_file(const std::string &keys_path, const std::string &user) {
    return KeyFile(keys_path).user_dir(user) + "/tailscale.tsv";
}

bool save_invites(const std::string &keys_path, const std::string &user, const std::vector<ShareInvite> &v,
                  std::string &err) {
    const std::string path = invites_file(keys_path, user);
    if (v.empty()) {
        unlink(path.c_str());
        return true;
    }
    make_dirs(dir_of(path));
    std::string text = "# ts5 share v1\n#id\tway\turl\tmade\n";
    for (const ShareInvite &i : v)
        text += i.id + "\t" + i.way + "\t" + i.url + "\t" + std::to_string(i.made) + "\n";
    return write_private(path, text, err);
}

bool load_token(const std::string &keys_path, std::string &token, std::string &err) {
    if (!read_all(share_token_file(keys_path), token) || (token = trim(token)).empty()) {
        err = "no Tailscale access token yet: make one in Tailscale's admin console (Settings, Keys, Generate access "
              "token) and give it on ts5-tui's Connectors page, or: ts5 share token set";
        return false;
    }
    return true;
}

// Why Tailscale said no, in plain words.
std::string why(const HttpResponse &r) {
    Json j;
    std::string msg;
    if (Json::parse(r.body, j)) msg = j["message"].str();
    if (r.status == 401)
        return "Tailscale refused the access token (it may have run out: tokens last up to 90 days; make a new one "
               "and give it again)";
    if (r.status == 403)
        return "the access token may not do this" + (msg.empty() ? std::string() : " (" + msg + ")") +
               ": it must be made by an Owner or Admin of the tailnet";
    if (r.status == 404) return "Tailscale doesn't know it" + (msg.empty() ? std::string() : " (" + msg + ")");
    return "Tailscale answered " + std::to_string(r.status) + (msg.empty() ? std::string() : ": " + msg);
}

bool call(const std::string &method, const std::string &path, const std::string &body, HttpResponse &r,
          std::string &err, const std::string &token) {
    std::vector<std::string> h = {"Authorization: Bearer " + token, "Accept: application/json"};
    if (!body.empty()) h.push_back("Content-Type: application/json");
    r = HttpResponse();
    if (!http_request(method, api_base() + path, h, body, r, err, 30)) {
        err = "Tailscale couldn't be reached: " + err;
        return false;
    }
    return true;
}

// Only what an id in a path may hold.
bool id_ok(const std::string &s) {
    if (s.empty() || s.size() > 64) return false;
    for (unsigned char c : s)
        if (!isalnum(c) && c != '-' && c != '_') return false;
    return true;
}

bool find_user(const std::string &keys_path, const std::string &which, AccessUser &u, std::string &err) {
    KeyFile f(keys_path);
    if (!f.load(err)) return false;
    for (const AccessUser &x : f.users())
        if (x.id == which || x.name == which) {
            if (x.owner()) {
                err = "the owner is not shared with: they have the machine";
                return false;
            }
            u = x;
            return true;
        }
    err = "no such user: " + which + " (ts5 users list)";
    return false;
}

}  // namespace

std::string share_token_file(const std::string &keys_path) { return dir_of(keys_path) + "/tailscale-api.key"; }

std::vector<ShareInvite> share_invites(const std::string &keys_path, const std::string &user) {
    std::vector<ShareInvite> v;
    std::string text;
    if (!read_all(invites_file(keys_path, user), text)) return v;
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> c;
        size_t s = 0;
        for (size_t t; (t = line.find('\t', s)) != std::string::npos; s = t + 1) c.push_back(line.substr(s, t - s));
        c.push_back(line.substr(s));
        if (c.size() < 4 || !id_ok(c[0])) continue;
        v.push_back({c[0], c[1], c[2], atoll(c[3].c_str())});
    }
    return v;
}

bool share_revoke(const std::string &keys_path, const std::string &user, std::string &note, std::string &err) {
    note.clear();
    std::vector<ShareInvite> v = share_invites(keys_path, user);
    if (v.empty()) return true;
    std::string token;
    if (!load_token(keys_path, token, err)) {
        err = "couldn't take back their Tailscale invites: " + err;
        return false;
    }
    std::vector<ShareInvite> kept, joined;
    for (const ShareInvite &i : v) {
        HttpResponse r;
        std::string e;
        if (i.way == "tailnet") {
            // Taken already: the person is on the tailnet, which only the
            // admin console (Users) changes. Not yet: the invite ends.
            if (call("GET", "/api/v2/user-invites/" + i.id, "", r, e, token) && r.status == 404) {
                joined.push_back(i);
                continue;
            }
            if (!call("DELETE", "/api/v2/user-invites/" + i.id, "", r, e, token) ||
                (r.status / 100 != 2 && r.status != 404)) {
                if (e.empty()) e = why(r);
                kept.push_back(i);
                err = e;
            }
        } else if (!call("DELETE", "/api/v2/device-invites/" + i.id, "", r, e, token) ||
                   (r.status / 100 != 2 && r.status != 404)) {
            if (e.empty()) e = why(r);
            kept.push_back(i);
            err = e;
        }
    }
    // One left to try again: the joined ones stay too, so the next try
    // still says what is left for the owner.
    if (!kept.empty()) kept.insert(kept.end(), joined.begin(), joined.end());
    std::string e2;
    if (!save_invites(keys_path, user, kept, e2) && err.empty()) err = e2;
    if (!joined.empty())
        note = "They joined your tailnet: remove them in Tailscale's admin console (Users) if they should no "
               "longer reach it.";
    return kept.empty() && e2.empty();
}

int share_main(int argc, char **argv) {
    std::string path = default_keys_path();
    bool json = false;
    std::vector<std::string> a;
    for (int i = 2; i < argc; i++) {
        const std::string s = argv[i];
        if (s == "--file" && i + 1 < argc) path = argv[++i];
        else if (s == "--json") json = true;
        else a.push_back(s);
    }
    const auto fail = [&](const std::string &why_) {
        if (json) printf("{\"error\":%s}\n", json_escape(why_).c_str());
        else fprintf(stderr, "ts5 share: %s\n", why_.c_str());
        return 1;
    };
    const auto usage = [&] {
        return fail("ts5 share [--file FILE] [--json] status | token set (the token on stdin) | token clear | "
                    "new USER ps5 NODE_ID | new USER tailnet | list USER | revoke USER (ps5: this machine only, "
                    "NODE_ID its Tailscale id; tailnet: join the whole tailnet; FILE: the keys' file, " +
                    default_keys_path() + ")");
    };
    const std::string cmd = a.empty() ? "status" : a[0];
    std::string err;
    if (cmd == "status" && a.size() <= 1) {
        std::string token;
        const bool have = read_all(share_token_file(path), token) && !trim(token).empty();
        if (json) printf("{\"token\":%s}\n", have ? "true" : "false");
        else printf("Tailscale access token: %s\n", have ? "given" : "none");
        return 0;
    }
    if (cmd == "token" && a.size() == 2 && a[1] == "clear") {
        unlink(share_token_file(path).c_str());
        if (json) printf("{\"token\":false}\n");
        return 0;
    }
    if (cmd == "token" && a.size() == 2 && a[1] == "set") {
        std::string token;
        std::getline(std::cin, token);
        token = trim(token);
        if (token.empty() || token.size() > 512 || token.find_first_of(" \t\r\n") != std::string::npos)
            return fail("give the access token on stdin (one line, from Tailscale's admin console: Settings, Keys)");
        // Tried before it is kept: one that can't list the tailnet's
        // devices can't make invites either.
        HttpResponse r;
        if (!call("GET", "/api/v2/tailnet/-/devices", "", r, err, token)) return fail(err);
        if (r.status != 200) return fail(why(r));
        make_dirs(dir_of(share_token_file(path)));
        if (!write_private(share_token_file(path), token + "\n", err)) return fail(err);
        if (json) printf("{\"token\":true}\n");
        else fprintf(stderr, "Kept the Tailscale access token.\n");
        return 0;
    }
    if (cmd == "new" && a.size() >= 3) {
        AccessUser u;
        if (!find_user(path, a[1], u, err)) return fail(err);
        const std::string way = a[2];
        if (!((way == "ps5" && a.size() == 4) || (way == "tailnet" && a.size() == 3))) return usage();
        if (way == "ps5" && !id_ok(a[3])) return fail("not a Tailscale machine id: " + a[3]);
        std::string token;
        if (!load_token(path, token, err)) return fail(err);
        HttpResponse r;
        const bool ok = way == "ps5"
                            ? call("POST", "/api/v2/device/" + a[3] + "/device-invites",
                                   "[{\"multiUse\":false,\"allowExitNode\":false}]", r, err, token)
                            : call("POST", "/api/v2/tailnet/-/user-invites", "[{\"role\":\"member\"}]", r, err, token);
        if (!ok) return fail(err);
        if (r.status / 100 != 2) return fail(why(r));
        Json j;
        if (!Json::parse(r.body, j) || j.type != Json::ARR || j.a.empty())
            return fail("Tailscale's answer had no invite in it");
        const Json &got = j.a[0];
        ShareInvite inv{got["id"].type == Json::NUM ? got["id"].s : got["id"].str(), way, got["inviteUrl"].str(),
                        (long long)time(nullptr)};
        if (!id_ok(inv.id) || inv.url.compare(0, 8, "https://") != 0)
            return fail("Tailscale's answer had no invite link in it");
        std::vector<ShareInvite> v = share_invites(path, u.id);
        v.push_back(inv);
        if (!save_invites(path, u.id, v, err)) return fail(err);
        if (json)
            printf("{\"id\":%s,\"way\":%s,\"url\":%s}\n", json_escape(inv.id).c_str(), json_escape(way).c_str(),
                   json_escape(inv.url).c_str());
        else printf("%s\n", inv.url.c_str());
        return 0;
    }
    if (cmd == "list" && a.size() == 2) {
        AccessUser u;
        if (!find_user(path, a[1], u, err)) return fail(err);
        std::string out = "[";
        for (const ShareInvite &i : share_invites(path, u.id)) {
            if (out.size() > 1) out += ",";
            out += "{\"id\":" + json_escape(i.id) + ",\"way\":" + json_escape(i.way) + ",\"url\":" +
                   json_escape(i.url) + ",\"made\":" + std::to_string(i.made) + "}";
            if (!json) printf("%s  %s  %s\n", i.way.c_str(), i.id.c_str(), i.url.c_str());
        }
        if (json) printf("{\"invites\":%s]}\n", out.c_str());
        return 0;
    }
    if (cmd == "revoke" && a.size() == 2) {
        AccessUser u;
        if (!find_user(path, a[1], u, err)) return fail(err);
        std::string note;
        if (!share_revoke(path, u.id, note, err)) return fail(err);
        if (json) printf("{\"revoked\":true,\"note\":%s}\n", json_escape(note).c_str());
        else if (!note.empty()) fprintf(stderr, "%s\n", note.c_str());
        return 0;
    }
    return usage();
}

}  // namespace ts5
