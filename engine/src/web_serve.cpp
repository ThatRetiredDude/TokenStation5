#include "web_serve.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <pty.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "json.h"
#include "log.h"
#include "webpush.h"
#include "mbedtls/base64.h"
#include "mbedtls/sha1.h"

namespace ts5 {

}  // namespace ts5

// The pages' files, built in (engine/web/web_assets.S).
#define TS5_WEB_FILE(sym) extern "C" const unsigned char sym[], sym##_end[];
TS5_WEB_FILE(ts5_web_chat_html)
TS5_WEB_FILE(ts5_web_terminal_html)
TS5_WEB_FILE(ts5_web_controls_html)
TS5_WEB_FILE(ts5_web_xterm_js)
TS5_WEB_FILE(ts5_web_xterm_css)
TS5_WEB_FILE(ts5_web_addon_fit_js)
TS5_WEB_FILE(ts5_web_xterm_license)
TS5_WEB_FILE(ts5_web_addon_fit_license)
TS5_WEB_FILE(ts5_web_icon_png)
TS5_WEB_FILE(ts5_web_keep_js)
TS5_WEB_FILE(ts5_web_sw_js)
TS5_WEB_FILE(ts5_web_manifest)
TS5_WEB_FILE(ts5_web_terminal_manifest)
TS5_WEB_FILE(ts5_web_push_js)
TS5_WEB_FILE(ts5_web_splash_1320x2868)
TS5_WEB_FILE(ts5_web_splash_1206x2622)
TS5_WEB_FILE(ts5_web_splash_1260x2736)
TS5_WEB_FILE(ts5_web_splash_1290x2796)
TS5_WEB_FILE(ts5_web_splash_1179x2556)
TS5_WEB_FILE(ts5_web_splash_1284x2778)
TS5_WEB_FILE(ts5_web_splash_1170x2532)
TS5_WEB_FILE(ts5_web_splash_1125x2436)
TS5_WEB_FILE(ts5_web_splash_1242x2688)
TS5_WEB_FILE(ts5_web_splash_828x1792)
TS5_WEB_FILE(ts5_web_splash_750x1334)
#undef TS5_WEB_FILE

namespace ts5 {

std::string websocket_accept(const std::string &key) {
    const std::string s = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char sha[20];
    mbedtls_sha1((const unsigned char *)s.data(), s.size(), sha);
    unsigned char out[64];
    size_t n = 0;
    mbedtls_base64_encode(out, sizeof out, &n, sha, sizeof sha);
    return std::string((const char *)out, n);
}

namespace {

struct WebAsset {
    const char *name;
    const unsigned char *data, *end;
};
const WebAsset kWebAssets[] = {
    {"chat.html", ts5_web_chat_html, ts5_web_chat_html_end},
    {"terminal.html", ts5_web_terminal_html, ts5_web_terminal_html_end},
    {"controls.html", ts5_web_controls_html, ts5_web_controls_html_end},
    {"xterm.js", ts5_web_xterm_js, ts5_web_xterm_js_end},
    {"xterm.css", ts5_web_xterm_css, ts5_web_xterm_css_end},
    {"addon-fit.js", ts5_web_addon_fit_js, ts5_web_addon_fit_js_end},
    {"xterm-LICENSE", ts5_web_xterm_license, ts5_web_xterm_license_end},
    {"addon-fit-LICENSE", ts5_web_addon_fit_license, ts5_web_addon_fit_license_end},
    {"icon.png", ts5_web_icon_png, ts5_web_icon_png_end},
    {"keep.js", ts5_web_keep_js, ts5_web_keep_js_end},
    {"sw.js", ts5_web_sw_js, ts5_web_sw_js_end},
    {"manifest.webmanifest", ts5_web_manifest, ts5_web_manifest_end},
    {"terminal.webmanifest", ts5_web_terminal_manifest, ts5_web_terminal_manifest_end},
    {"push.js", ts5_web_push_js, ts5_web_push_js_end},
    {"splash-1320x2868.jpg", ts5_web_splash_1320x2868, ts5_web_splash_1320x2868_end},
    {"splash-1206x2622.jpg", ts5_web_splash_1206x2622, ts5_web_splash_1206x2622_end},
    {"splash-1260x2736.jpg", ts5_web_splash_1260x2736, ts5_web_splash_1260x2736_end},
    {"splash-1290x2796.jpg", ts5_web_splash_1290x2796, ts5_web_splash_1290x2796_end},
    {"splash-1179x2556.jpg", ts5_web_splash_1179x2556, ts5_web_splash_1179x2556_end},
    {"splash-1284x2778.jpg", ts5_web_splash_1284x2778, ts5_web_splash_1284x2778_end},
    {"splash-1170x2532.jpg", ts5_web_splash_1170x2532, ts5_web_splash_1170x2532_end},
    {"splash-1125x2436.jpg", ts5_web_splash_1125x2436, ts5_web_splash_1125x2436_end},
    {"splash-1242x2688.jpg", ts5_web_splash_1242x2688, ts5_web_splash_1242x2688_end},
    {"splash-828x1792.jpg", ts5_web_splash_828x1792, ts5_web_splash_828x1792_end},
    {"splash-750x1334.jpg", ts5_web_splash_750x1334, ts5_web_splash_750x1334_end},
};

const WebAsset *asset(const std::string &name) {
    for (const WebAsset &a : kWebAssets)
        if (name == a.name) return &a;
    return nullptr;
}

bool send_all(int fd, const void *p, size_t n) {
    const char *c = (const char *)p;
    while (n) {
        const ssize_t k = ::send(fd, c, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return false;
        c += k;
        n -= (size_t)k;
    }
    return true;
}

bool send_text(int fd, const std::string &s) { return send_all(fd, s.data(), s.size()); }

const char *status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 401: return "Unauthorized";
        case 500: return "Internal Server Error";
        default: return "Error";
    }
}

// A page or file: no key needed, kept out of other sites' frames, and the
// pages' scripts and connections only from here. A page, script or style: a
// browser asks again each time (no-cache) and gets 304 while the file is the
// same (its ETag), so a cached script from an older ts5 never runs with a
// newer page. A picture (the logo, the launch images): kept an hour
// (max-age=3600), as when the iPhone's home-screen icon showed the logo; with
// no-cache it drew a letter tile though it had fetched the logo whole (iOS
// 27.2, 2026-11-13 and 11-14).
void send_file(int fd, int code, const std::string &type, const std::string &body, bool page,
               const std::string &etag = "") {
    const bool picture = type.rfind("image/", 0) == 0;
    std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\nContent-Type: " + type +
                    "\r\nContent-Length: " + std::to_string(body.size()) +
                    "\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nCache-Control: " +
                    (picture ? "max-age=3600" : "no-cache") + "\r\n";
    if (!etag.empty()) h += "ETag: " + etag + "\r\n";
    if (page)
        h += "X-Frame-Options: DENY\r\n"
             "Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; "
             "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'\r\n";
    h += "Connection: close\r\n\r\n";
    send_text(fd, h + body);
}

void send_error(int fd, int code, const std::string &why) {
    Json j = Json::object();
    j.set("error", Json::string(why));
    send_file(fd, code, "application/json", j.dump(), false);
}

std::string type_of(const std::string &name) {
    const size_t dot = name.rfind('.');
    const std::string ext = dot == std::string::npos ? "" : name.substr(dot + 1);
    if (ext == "html") return "text/html; charset=utf-8";
    if (ext == "js") return "text/javascript; charset=utf-8";
    if (ext == "css") return "text/css; charset=utf-8";
    if (ext == "png") return "image/png";
    if (ext == "jpg") return "image/jpeg";
    if (ext == "webmanifest") return "application/manifest+json";
    return "text/plain; charset=utf-8";
}

// TS5_WEB_LOG=1: a line for each page, file and icon asked for (who, what,
// the answer, its size, the browser), with the time: what a phone fetches
// when a home-screen icon is added. Off unless set: the pages are many
// requests.
bool web_log_on() {
    static const bool on = [] {
        const char *v = getenv("TS5_WEB_LOG");
        return v && *v && strcmp(v, "0") != 0;
    }();
    return on;
}

void web_log(const Server::Page &p, int code, size_t bytes) {
    if (!web_log_on()) return;
    char t[16];
    const time_t now = time(nullptr);
    struct tm tm {};
    localtime_r(&now, &tm);
    strftime(t, sizeof t, "%H:%M:%S", &tm);
    std::string ua = p.header("user-agent");
    if (ua.size() > 160) ua.resize(160);
    log_line("ts5: web: %s %s %s %s%s -> %d, %zu bytes%s (%s)", t, p.client.c_str(), p.method.c_str(), p.path.c_str(),
             p.header("if-none-match").empty() ? "" : " (if changed)", code, bytes,
             p.header("cookie").find("ts5key=") != std::string::npos ? ", with the key's cookie" : "", ua.c_str());
}

// A file's ETag: FNV-1a of its bytes.
std::string etag_of(const unsigned char *d, const unsigned char *end) {
    uint64_t h = 1469598103934665603ull;
    for (; d < end; d++) h = (h ^ *d) * 1099511628211ull;
    char b[24];
    snprintf(b, sizeof b, "\"%016llx\"", (unsigned long long)h);
    return b;
}

// For working on the pages: TS5_WEB_DIR=FOLDER serves a file of the same
// name from FOLDER instead of the built-in one, read at each request, so a
// changed page is live without a new build. Only the built-in names.
bool from_web_dir(const std::string &name, std::string &out) {
    const char *dir = getenv("TS5_WEB_DIR");
    if (!dir || !*dir) return false;
    FILE *f = fopen((std::string(dir) + "/" + name).c_str(), "rb");
    if (!f) return false;
    out.clear();
    char b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) out.append(b, n);
    fclose(f);
    return true;
}

bool serve_asset(int fd, const Server::Page &p, const std::string &name, bool page) {
    const WebAsset *a = asset(name);
    if (!a) return false;
    std::string body;
    if (!from_web_dir(name, body)) body.assign((const char *)a->data, (size_t)(a->end - a->data));
    if (type_of(name).rfind("image/", 0) == 0) {
        send_file(fd, 200, type_of(name), body, page);
        web_log(p, 200, body.size());
        return true;
    }
    const std::string etag =
        etag_of((const unsigned char *)body.data(), (const unsigned char *)body.data() + body.size());
    if (p.header("if-none-match") == etag) {
        // No Content-Length: a 304 may give only the 200's own (RFC 9110,
        // 15.4.5), and a cache that took a 0 from it would keep an empty file.
        send_text(fd, "HTTP/1.1 304 Not Modified\r\nETag: " + etag +
                          "\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
        web_log(p, 304, 0);
        return true;
    }
    send_file(fd, 200, type_of(name), body, page, etag);
    web_log(p, 200, body.size());
    return true;
}

// Who may open the terminal: this machine, and the tailnet's devices
// (100.64.0.0/10, as the tunnel names its connections). Not the rest of the
// network, whatever Access says about other devices.
bool terminal_client(const std::string &client) {
    if (client == "127.0.0.1" || client == "::1") return true;
    in_addr a{};
    if (inet_pton(AF_INET, client.c_str(), &a) != 1) return false;
    return (ntohl(a.s_addr) & 0xFFC00000u) == 0x64400000u;
}

// The page's own address: a browser sends Origin with every WebSocket, and
// another site's page can't make it this one.
bool same_origin(const Server::Page &p) {
    const std::string origin = p.header("origin"), host = p.header("host");
    if (origin.empty() || host.empty()) return false;
    const size_t at = origin.find("://");
    return at != std::string::npos && origin.substr(at + 3) == host;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- WebSocket (RFC 6455), the server's side ----

struct Ws {
    int fd;
    std::string in;  // bytes read, not yet a whole frame
    std::string msg;  // a message's fragments so far
    int msg_op = 0;
};

bool ws_send(int fd, int op, const std::string &payload) {
    std::string h;
    h += (char)(0x80 | op);
    if (payload.size() < 126) h += (char)payload.size();
    else if (payload.size() < 65536) {
        h += (char)126;
        h += (char)(payload.size() >> 8);
        h += (char)(payload.size() & 0xff);
    } else {
        h += (char)127;
        for (int i = 7; i >= 0; i--) h += (char)((uint64_t)payload.size() >> (8 * i) & 0xff);
    }
    return send_text(fd, h + payload);
}

void ws_close(int fd, int code, const std::string &why) {
    std::string p;
    p += (char)(code >> 8);
    p += (char)(code & 0xff);
    p += why.substr(0, 120);
    ws_send(fd, 0x8, p);
}

// The next whole message from what has been read: 1 with op and data, 0
// when more is needed, -1 when the client broke the rules or closed.
int ws_next(Ws &w, int &op, std::string &data) {
    for (;;) {
        if (w.in.size() < 2) return 0;
        const unsigned char b0 = (unsigned char)w.in[0], b1 = (unsigned char)w.in[1];
        const bool fin = b0 & 0x80, masked = b1 & 0x80;
        const int code = b0 & 0x0f;
        uint64_t len = b1 & 0x7f;
        size_t at = 2;
        if (len == 126) {
            if (w.in.size() < 4) return 0;
            len = ((unsigned char)w.in[2] << 8) | (unsigned char)w.in[3];
            at = 4;
        } else if (len == 127) {
            if (w.in.size() < 10) return 0;
            len = 0;
            for (int i = 0; i < 8; i++) len = len << 8 | (unsigned char)w.in[2 + i];
            at = 10;
        }
        if (!masked || len > (1u << 20)) return -1;  // a browser masks every frame; big ones aren't keystrokes
        if (w.in.size() < at + 4 + len) return 0;
        const unsigned char *mask = (const unsigned char *)w.in.data() + at;
        std::string p = w.in.substr(at + 4, (size_t)len);
        for (size_t i = 0; i < p.size(); i++) p[i] = (char)(p[i] ^ mask[i & 3]);
        w.in.erase(0, at + 4 + (size_t)len);
        if (code >= 0x8) {  // control frames come whole, between a message's fragments
            if (code == 0x8) return -1;
            if (code == 0x9) ws_send(w.fd, 0xA, p);
            continue;
        }
        if (code != 0) {
            w.msg_op = code;
            w.msg.clear();
        }
        w.msg += p;
        if (w.msg.size() > (1u << 20)) return -1;
        if (!fin) continue;
        op = w.msg_op;
        data.swap(w.msg);
        w.msg.clear();
        return 1;
    }
}

// Reads what the socket has (waiting up to ms) into w.in: false once the
// client is gone.
bool ws_read(Ws &w, int ms) {
    pollfd p{w.fd, POLLIN, 0};
    const int r = poll(&p, 1, ms);
    if (r < 0) return errno == EINTR;
    if (r == 0) return true;
    char buf[16384];
    const ssize_t n = recv(w.fd, buf, sizeof buf, 0);
    if (n <= 0) return false;
    w.in.append(buf, (size_t)n);
    return true;
}

std::atomic<int> g_open{0};  // terminals open now

// A user's browser terminal opened (open) or closed: notifications for
// ts5-tui's answers go out while none is open (the push section, below).
void push_terminal(const WebOptions &o, const std::string &user, bool open);
constexpr int kMaxOpen = 4;

std::string tui_command() {
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n > 0) {
        std::string dir(self, (size_t)n);
        dir.resize(dir.rfind('/') + 1);
        const std::string tui = dir + "ts5-tui";
        if (access(tui.c_str(), X_OK) == 0) return "'" + tui + "' --tmux";
    }
    return "ts5-tui --tmux";
}

void terminal(const WebOptions &o, const Server::Page &p, int fd) {
    if (!o.terminal || !o.terminal()) {
        send_error(fd, 403, "The browser terminal is off: ts5-tui's Access page (F6) turns it on.");
        return;
    }
    if (!terminal_client(p.client)) {
        log_line("ts5: terminal: %s asked for it, refused: only the tailnet's devices may", p.client.c_str());
        send_error(fd, 403, "The browser terminal is only for your tailnet's devices (Tailscale).");
        return;
    }
    const std::string key = p.header("sec-websocket-key");
    if (key.empty() || p.header("sec-websocket-version") != "13") {
        // The page asking first whether it may (a plain GET): yes so far.
        send_file(fd, 200, "application/json", "{\"terminal\":\"on\",\"open\":\"as a WebSocket (version 13)\"}", false);
        return;
    }
    if (!same_origin(p)) {
        log_line("ts5: terminal: %s asked from another site's page (%s), refused", p.client.c_str(),
                 p.header("origin").c_str());
        send_error(fd, 403, "The terminal opens from its own page, /ts5/terminal.");
        return;
    }
    send_text(fd, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  "Sec-WebSocket-Accept: " + websocket_accept(key) + "\r\n\r\n");
    // The first message: {"key":"...","cols":N,"rows":N}, within 15 seconds.
    Ws w{fd, {}, {}, 0};
    int op = 0;
    std::string first;
    const double until = now_s() + 15;
    int got = 0;
    while ((got = ws_next(w, op, first)) == 0 && now_s() < until)
        if (!ws_read(w, 500)) return;
    Json hello;
    if (got != 1 || op != 0x1 || !Json::parse(first, hello)) {
        ws_close(fd, 1002, "the first message is the key");
        return;
    }
    std::string name;
    const std::string sent = hello["key"].str();
    Server::Grant grant;
    const bool in = o.admit ? o.admit(sent, p.client, grant) : o.authorize && o.authorize(sent, p.client, name);
    if (o.admit) name = grant.key;
    if (in && !grant.owner) {
        // Another user's key: the chat page and the API, never all of ts5-tui.
        log_line("ts5: terminal: %s: the key %s is %s's, not the owner's: refused", p.client.c_str(), name.c_str(),
                 grant.user_name.c_str());
        Json e = Json::object();
        e.set("error", Json::string("This key is " + grant.user_name +
                                    "'s: the browser terminal is the owner's alone. The chat page is at /."));
        ws_send(fd, 0x1, e.dump());
        ws_close(fd, 1008, "owner only");
        return;
    }
    if (!in) {
        log_line("ts5: terminal: %s: %s, refused", p.client.c_str(),
                 sent.empty() ? "no API key" : "an API key that isn't one of this machine's");
        Json e = Json::object();
        e.set("error", Json::string(sent.empty() ? "This needs an API key: make one on ts5-tui's Access page (F6)."
                                                 : "That API key isn't one of this machine's, or it is off."));
        ws_send(fd, 0x1, e.dump());
        ws_close(fd, 1008, "key");
        return;
    }
    if (g_open.fetch_add(1) >= kMaxOpen) {
        g_open.fetch_sub(1);
        Json e = Json::object();
        e.set("error", Json::string("Four terminals are open already: close one first."));
        ws_send(fd, 0x1, e.dump());
        ws_close(fd, 1013, "busy");
        return;
    }
    struct Done {
        ~Done() { g_open.fetch_sub(1); }
    } done;
    // Its user's: while it is open, an answer that ends in ts5-tui is seen here.
    struct Seen {
        const WebOptions &o;
        std::string user;
        ~Seen() { if (!user.empty()) push_terminal(o, user, false); }
    } seen{o, ""};
    if (o.admit && o.user_dir && !grant.user.empty()) {
        seen.user = grant.user;
        push_terminal(o, grant.user, true);
    }
    winsize ws{};
    ws.ws_col = (unsigned short)std::max(20.0, std::min(500.0, hello["cols"].num(80)));
    ws.ws_row = (unsigned short)std::max(8.0, std::min(300.0, hello["rows"].num(24)));
    std::string cmd = o.command;
    if (const char *c = getenv("TS5_TERMINAL_COMMAND"); cmd.empty() && c && *c) cmd = c;
    if (cmd.empty()) cmd = tui_command();
    int master = -1;
    const pid_t pid = forkpty(&master, nullptr, nullptr, &ws);
    if (pid < 0) {
        log_line("ts5: terminal: can't start one (errno %d)", errno);
        ws_close(fd, 1011, "no terminal");
        return;
    }
    if (pid == 0) {
        // The child: the server's secrets stay with the server.
        unsetenv("TS5_API_KEY");
        unsetenv("TS5_TS_AUTH_KEY");
        setenv("TERM", "xterm-256color", 1);
        setenv("COLORTERM", "truecolor", 1);
        setenv("TS5_BROWSER_TERMINAL", "1", 1);
        if (!getenv("LANG") && !getenv("LC_ALL")) setenv("LANG", "C.UTF-8", 1);
        signal(SIGPIPE, SIG_DFL);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *)nullptr);
        _exit(127);
    }
    const double t0 = now_s();
    log_line("ts5: terminal: %s opened one%s%s", p.client.c_str(), name.empty() ? "" : " with the key ",
             name.c_str());
    ws_send(fd, 0x1, "{\"ready\":true}");
    std::string why = "closed by the browser";
    char buf[16384];
    double checked = now_s();
    bool ended = false;
    for (;;) {
        pollfd pf[2] = {{fd, POLLIN, 0}, {master, POLLIN, 0}};
        const int r = poll(pf, 2, 1000);
        if (r < 0 && errno != EINTR) break;
        if (now_s() - checked >= 1) {
            checked = now_s();
            if (o.terminal && !o.terminal()) {
                why = "its switch turned off";
                ws_close(fd, 1001, "the browser terminal was turned off");
                break;
            }
        }
        if (r <= 0) continue;
        if (pf[1].revents & (POLLIN | POLLHUP | POLLERR)) {
            const ssize_t n = read(master, buf, sizeof buf);
            if (n <= 0) {
                why = "the program ended";
                ended = true;
                ws_close(fd, 1000, "ended");
                break;
            }
            if (!ws_send(fd, 0x2, std::string(buf, (size_t)n))) break;
        }
        if (pf[0].revents & (POLLIN | POLLHUP | POLLERR)) {
            const ssize_t n = recv(fd, buf, sizeof buf, 0);
            if (n <= 0) break;
            w.in.append(buf, (size_t)n);
            std::string data;
            int got2;
            bool bye = false;
            while ((got2 = ws_next(w, op, data)) == 1) {
                if (op == 0x2) {
                    // Keystrokes and pastes, as they are.
                    size_t off = 0;
                    while (off < data.size()) {
                        const ssize_t k = write(master, data.data() + off, data.size() - off);
                        if (k < 0 && errno == EINTR) continue;
                        if (k <= 0) break;
                        off += (size_t)k;
                    }
                } else if (op == 0x1) {
                    Json m;
                    if (Json::parse(data, m) && m["cols"].type == Json::NUM && m["rows"].type == Json::NUM) {
                        winsize z{};
                        z.ws_col = (unsigned short)std::max(20.0, std::min(500.0, m["cols"].num(80)));
                        z.ws_row = (unsigned short)std::max(8.0, std::min(300.0, m["rows"].num(24)));
                        ioctl(master, TIOCSWINSZ, &z);
                    }
                }
            }
            if (got2 < 0) bye = true;
            if (bye) {
                ws_close(fd, 1000, "bye");
                break;
            }
        }
    }
    close(master);
    // A closed terminal hangs its programs up; ts5-tui's tmux session stays.
    if (!ended) kill(-pid, SIGHUP);
    for (int i = 0; i < 30 && waitpid(pid, nullptr, WNOHANG) == 0; i++) usleep(100 * 1000);
    if (waitpid(pid, nullptr, WNOHANG) == 0) {
        kill(-pid, SIGKILL);
        waitpid(pid, nullptr, 0);
    }
    log_line("ts5: terminal: %s's session ended after %.0f s (%s)", p.client.c_str(), now_s() - t0, why.c_str());
}

// ---- each user's own chats (/ts5/me, /ts5/chats) ----

constexpr size_t kChatMax = 4u << 20;  // a chat's JSON
constexpr int kChatsMax = 1000;         // a user's chats
// All of a user's chats together: a user can't fill the machine's drive.
// (TS5_CHATS_MAX_BYTES for tests.)
size_t chats_max_bytes() {
    const char *e = getenv("TS5_CHATS_MAX_BYTES");
    return e && *e ? (size_t)strtoull(e, nullptr, 10) : (size_t)256 << 20;
}

bool good_chat_id(const std::string &id) {
    if (id.empty() || id.size() > 64) return false;
    for (char c : id)
        if (!isalnum((unsigned char)c) && c != '-' && c != '_') return false;
    return true;
}

void make_dirs(const std::string &path) {
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') mkdir(path.substr(0, i).c_str(), 0700);
}

std::string read_whole(const std::string &path) {
    std::string out;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return out;
    char b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) out.append(b, n);
    fclose(f);
    return out;
}

bool write_whole(const std::string &path, const std::string &text) {
    const std::string tmp = path + ".new";
    // The user's own: private even if a folder above it were opened up.
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE *f = fd < 0 ? nullptr : fdopen(fd, "wb");
    if (!f) {
        if (fd >= 0) close(fd);
        return false;
    }
    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    ok = fclose(f) == 0 && ok;
    if (ok && rename(tmp.c_str(), path.c_str()) == 0) return true;
    unlink(tmp.c_str());
    return false;
}

std::vector<std::string> chat_files(const std::string &dir) {
    std::vector<std::string> out;
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".json") == 0 && good_chat_id(n.substr(0, n.size() - 5)))
                out.push_back(n);
        }
        closedir(d);
    }
    return out;
}


// A chat file's mark: another whenever the file is written again (each
// write is a new file, write_whole; ts5-tui's too).
std::string chat_mark(const std::string &path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return "";
    char b[96];
    snprintf(b, sizeof b, "%llx.%lx.%llx.%llx", (unsigned long long)st.st_mtim.tv_sec, (unsigned long)st.st_mtim.tv_nsec,
             (unsigned long long)st.st_size, (unsigned long long)st.st_ino);
    return b;
}

// ---- the owner's chats: ts5-tui's own, one list ----
//
// The owner's chats on the chat page are ts5-tui's chats (DATA/chats, in
// ts5-tui's format, linux/tui/store.cpp): the same list on the phone and
// in ts5-tui. The chat page's own format is made from them on the way out
// and taken back into them on the way in. Not for the background service
// (its data is its own: /var/lib/tokenstation, where ts5-tui keeps none):
// there the owner's chats stay under users/ID like any user's.

Json jnum(double v) {
    Json j;
    j.type = Json::NUM;
    j.n = v;
    char b[32];
    if (v == (double)(long long)v) snprintf(b, sizeof b, "%lld", (long long)v);
    else snprintf(b, sizeof b, "%.2f", v);
    j.s = b;
    return j;
}
Json jbool(bool v) {
    Json j;
    j.type = Json::BOOL;
    j.b = v;
    return j;
}

std::string user_data_dir();

// ts5-tui's chats folder (DATA/chats), for the owner, when this server's
// keys are ts5-tui's (DATA/access/keys.tsv: the server ts5-tui starts, or
// one its user runs with them); else "" (the service, or keys elsewhere).
std::string tui_chats_dir(const WebOptions &o, const Server::Grant &g) {
    if (!g.owner || !o.user_dir) return "";
    const std::string ud = o.user_dir(g.user);
    const size_t at = ud.rfind("/users/");
    if (at == std::string::npos) return "";
    const std::string base = ud.substr(0, at);
    char a[PATH_MAX], b[PATH_MAX];
    const std::string data = user_data_dir();
    const bool same = base == data || (realpath(base.c_str(), a) && realpath(data.c_str(), b) && strcmp(a, b) == 0);
    return same ? base + "/chats" : "";
}

// A turn ts5-tui keeps that the chat page shows: words and answers (not a
// tool's result, not a summary: they stay in the file, for ts5-tui).
bool shown_turn(const Json &t) { return !t["summary"].boolean(false) && t["tool"].str().empty(); }

// ts5-tui's chat as the chat page's: {id, title, at, messages}.
Json page_chat(const std::string &id, const Json &c) {
    Json o = Json::object();
    o.set("id", Json::string(id));
    o.set("title", Json::string(c["title"].str("A chat")));
    o.set("at", jnum(c["updated"].num(c["created"].num(0)) * 1000));
    Json ms;
    ms.type = Json::ARR;
    // A turn of ts5-tui's that only called tools (no words of its own: the
    // answer comes in a later turn) isn't a message of its own on the page
    // (an empty one, which the model would get back as an empty answer):
    // the tools it used, and what it thought, go with the answer after it.
    std::string used_before, thought_before;
    const auto alone = [&] {  // no answer came after it: its note alone
        if (used_before.empty()) return;
        Json m = Json::object();
        m.set("role", Json::string("assistant"));
        m.set("content", Json::string(""));
        m.set("done", jbool(true));
        if (!thought_before.empty()) m.set("thinking", Json::string(thought_before));
        m.set("stats", Json::string("used " + used_before));
        ms.a.push_back(m);
        used_before.clear();
        thought_before.clear();
    };
    for (const Json &t : c["turns"].a) {
        if (!shown_turn(t)) continue;
        Json m = Json::object();
        const bool user = t["user"].boolean(false);
        if (user) alone();
        m.set("role", Json::string(user ? "user" : "assistant"));
        m.set("content", Json::string(t["text"].str()));
        if (!user) {
            const std::string end = t["end"].str("done");
            std::string used;
            for (const Json &u : t["calls"].a) used += (used.empty() ? "" : ", ") + u["name"].str();
            if (t["text"].str().empty() && !used.empty() && end != "failed") {
                used_before += (used_before.empty() ? "" : ", ") + used;
                if (!t["reasoning"].str().empty()) thought_before += (thought_before.empty() ? "" : "\n\n") + t["reasoning"].str();
                continue;
            }
            if (!used_before.empty()) used = used.empty() ? used_before : used_before + ", " + used;
            std::string thought = t["reasoning"].str();
            if (!thought_before.empty()) thought = thought.empty() ? thought_before : thought_before + "\n\n" + thought;
            used_before.clear();
            thought_before.clear();
            m.set("done", jbool(true));
            if (!thought.empty()) m.set("thinking", Json::string(thought));
            std::string stats;
            const double tokens = t["tokens"].num(0), tps = t["tps"].num(0);
            if (tokens > 0) {
                char b[64];
                snprintf(b, sizeof b, "%.0f tokens · %.1f tok/s", tokens, tps);
                stats = b;
            }
            if (end == "stopped") stats = stats.empty() ? "stopped" : stats + " · stopped";
            if (!used.empty()) stats += (stats.empty() ? "" : " · ") + std::string("used ") + used;
            if (!stats.empty()) m.set("stats", Json::string(stats));
            if (end == "failed") m.set("error", Json::string(t["error"].str("it failed")));
        }
        ms.a.push_back(m);
    }
    alone();
    o.set("messages", ms);
    return o;
}

// The chat page's messages taken into ts5-tui's chat (old: what the file
// holds now, or none): the page only ever adds messages at the end, so
// those after the ones both have are added, after ts5-tui's own turns.
// An answer still coming isn't kept yet (ts5-tui keeps finished ones).
std::string tui_chat(const Json &page, const Json *old, const std::string &model) {
    const int64_t now = (int64_t)time(nullptr);
    Json c = Json::object();
    c.set("v", jnum(1));
    std::string title = old ? old->operator[]("title").str() : "";
    if (title.empty()) title = page["title"].str();
    std::vector<Json> turns = old ? old->operator[]("turns").a : std::vector<Json>{};
    std::vector<Json> mine;  // the page's messages, as turns
    for (const Json &m : page["messages"].a) {
        const std::string role = m["role"].str();
        if (role != "user" && role != "assistant") continue;
        const bool user = role == "user";
        if (!user && !m["done"].boolean(false)) continue;
        Json t = Json::object();
        t.set("user", jbool(user));
        t.set("text", Json::string(m["content"].str()));
        if (!user) {
            t.set("model", Json::string(m["model"].str(model)));
            const std::string stats = m["stats"].str();
            double tokens = 0, tps = 0;
            sscanf(stats.c_str(), "%lf tokens · %lf tok/s", &tokens, &tps);
            t.set("tps", jnum(tps));
            t.set("tokens", jnum(tokens));
            const bool failed = !m["error"].str().empty();
            t.set("end", Json::string(failed ? "failed" : stats.find("stopped") != std::string::npos ? "stopped" : "done"));
            if (failed) t.set("error", Json::string(m["error"].str()));
            if (!m["thinking"].str().empty()) t.set("reasoning", Json::string(m["thinking"].str()));
        }
        mine.push_back(t);
    }
    // What is matched and kept: words, and answers with something in them.
    // (An empty answer is one that only called tools, in ts5-tui, and one
    // from the page with nothing in it, a Stop before the first word, isn't
    // kept: neither is a place where the two can differ.)
    const auto counts = [](const Json &t) {
        return t["user"].boolean(false) || !t["text"].str().empty() || t["end"].str() == "failed";
    };
    const auto same_turn = [](const Json &a, const Json &b) {
        return a["user"].boolean(false) == b["user"].boolean(false) && a["text"].str() == b["text"].str();
    };
    std::vector<const Json *> file;  // the file's, of those
    for (const Json &t : turns)
        if (shown_turn(t) && counts(t)) file.push_back(&t);
    std::vector<const Json *> sent;  // the page's
    for (const Json &t : mine)
        if (counts(t)) sent.push_back(&t);
    // The ones both have: from the start, in order, the same.
    size_t same = 0;
    while (same < file.size() && same < sent.size() && same_turn(*file[same], *sent[same])) same++;
    // The file may have gone on past the page's copy of it (turns ts5-tui
    // added, an answer kept here) before the page's new ones were written
    // at its end by an earlier save: those it ends with are there already.
    size_t there = 0;
    for (size_t n = std::min(file.size() - same, sent.size() - same); n > 0 && !there; n--) {
        bool ends = true;
        for (size_t i = 0; i < n && ends; i++) ends = same_turn(*file[file.size() - n + i], *sent[same + i]);
        if (ends) there = n;
    }
    std::vector<Json> added;
    for (size_t i = same + there; i < sent.size(); i++) added.push_back(*sent[i]);
    turns.insert(turns.end(), added.begin(), added.end());
    if (title.empty())
        for (const Json &t : turns)
            if (t["user"].boolean(false)) {
                title = t["text"].str().substr(0, 60);
                break;
            }
    c.set("title", Json::string(title.empty() ? "A chat" : title));
    c.set("created", jnum(old ? old->operator[]("created").num((double)now) : (double)now));
    c.set("updated", jnum((double)now));
    Json arr;
    arr.type = Json::ARR;
    arr.a = turns;
    c.set("turns", arr);
    return c.dump() + "\n";
}

// A chat's file: the page's PUT, and an answer kept here (keep_answer).
std::mutex chats_mu;

bool me_and_chats(const WebOptions &o, const Server::Page &p, int fd) {
    const bool me = p.path == "/ts5/me";
    const bool chats = p.path == "/ts5/chats" || p.path.rfind("/ts5/chats/", 0) == 0;
    const bool users = p.path == "/ts5/users" || p.path.rfind("/ts5/users/", 0) == 0;
    if (!me && !chats && !users) return false;
    if (!o.admit) {
        send_error(fd, 404, "this server keeps no users");
        return true;
    }
    Server::Grant g;
    if (!o.admit(p.key, p.client, g)) {
        send_error(fd, 401, p.key.empty() ? "This needs an API key." : "That API key isn't one of this machine's, or it is off.");
        return true;
    }
    // Chats on the machine only for a request with a key: without one
    // (this machine, or a network that lets any machine in) nobody can tell
    // whose they would be.
    const bool kept = !p.key.empty() && !g.key.empty() && o.user_dir;
    if (me) {
        if (p.method != "GET") {
            send_error(fd, 405, "GET only");
            return true;
        }
        Json j = Json::object();
        j.set("user", Json::string(g.user));
        j.set("name", Json::string(g.user_name));
        Json b;
        b.type = Json::BOOL;
        b.b = g.owner;
        j.set("owner", b);
        b.b = g.owner;
        j.set("terminal", b);
        b.b = g.web_search;
        j.set("web_search", b);
        b.b = g.app_tools;
        j.set("app_tools", b);
        Json n;
        n.type = Json::NUM;
        n.n = (double)g.max_reply;
        n.s = std::to_string(g.max_reply);
        j.set("max_reply", n);
        b.b = kept;
        j.set("chats_kept", b);
        b.b = kept && !tui_chats_dir(o, g).empty();
        j.set("tui_chats", b);  // the owner's chats are ts5-tui's own list
        b.b = g.owner_reads;
        j.set("owner_reads", b);
        n.n = (double)g.tokens_day;
        n.s = std::to_string(g.tokens_day);
        j.set("tokens_day", n);
        Json ms;
        ms.type = Json::ARR;
        for (const std::string &m : g.models) ms.a.push_back(Json::string(m));
        j.set("models", ms);
        b.b = g.owner || g.may_switch;
        j.set("may_switch", b);
        j.set("until", Json::string(g.until));
        send_file(fd, 200, "application/json", j.dump(), false);
        return true;
    }
    if (users) {
        // The owner reading a user's chats: only a user who has that set,
        // which the user sees in its own settings.
        if (!kept || !g.owner || !o.readable_users) {
            send_error(fd, 403, "Only the owner's key reads other users' chats, and only theirs who allow it.");
            return true;
        }
        if (p.method != "GET") {
            send_error(fd, 405, "GET only");
            return true;
        }
        const auto readable = o.readable_users();
        if (p.path == "/ts5/users" || p.path == "/ts5/users/") {
            Json list = Json::object();
            Json arr;
            arr.type = Json::ARR;
            for (const auto &u : readable) {
                Json x = Json::object();
                x.set("id", Json::string(u.first));
                x.set("name", Json::string(u.second));
                arr.a.push_back(x);
            }
            list.set("users", arr);
            send_file(fd, 200, "application/json", list.dump(), false);
            return true;
        }
        const std::string rest = p.path.substr(11);
        const size_t slash = rest.find('/');
        const std::string uid = rest.substr(0, slash);
        bool may = false;
        for (const auto &u : readable) may |= u.first == uid;
        if (slash == std::string::npos || rest.substr(slash) != "/chats" || !may) {
            send_error(fd, 404, "no such user, or its chats are its own");
            return true;
        }
        const std::string dir = o.user_dir(uid) + "/chats";
        std::string out = "{\"chats\":[";
        bool first = true;
        for (const std::string &n : chat_files(dir)) {
            const std::string text = read_whole(dir + "/" + n);
            Json c;
            if (!Json::parse(text, c) || c.type != Json::OBJ) continue;
            out += (first ? "" : ",") + text;
            first = false;
        }
        log_line("ts5: the owner read %s's chats (%s)", uid.c_str(), p.client.c_str());
        send_file(fd, 200, "application/json", out + "]}", false);
        return true;
    }
    if (!kept) {
        send_error(fd, 403, "Chats are kept on the machine for a key's user; without a key they stay in the browser.");
        return true;
    }
    // The owner's: ts5-tui's chats (tui), with the ones kept apart before
    // (dir) until each is written again.
    const std::string tui = tui_chats_dir(o, g);
    const std::string dir = o.user_dir(g.user) + "/chats";
    const std::string id = p.path.size() > 11 ? p.path.substr(11) : "";
    if (p.path == "/ts5/chats" || p.path == "/ts5/chats/") {
        if (p.method == "GET") {
            // ?marks=1: which chats there are and each one's mark, and no
            // chat. What the page asks every 15 s: it then fetches the ones
            // whose mark changed, not every chat with all its messages
            // (megabytes, with some dozens of chats). No file is read for it.
            const bool only_marks = p.query.find("marks=1") != std::string::npos;
            std::string out = "{\"chats\":[", marks;
            bool first = true;
            std::vector<std::string> seen;
            // (The mark before the file is read: one written in between is
            // fetched again, never missed.)
            if (!tui.empty())
                for (const std::string &n : chat_files(tui)) {
                    const std::string m = chat_mark(tui + "/" + n);
                    if (!only_marks) {
                        Json c;
                        if (!Json::parse(read_whole(tui + "/" + n), c) || c.type != Json::OBJ || c["turns"].type != Json::ARR)
                            continue;
                        out += (first ? "" : ",") + page_chat(n.substr(0, n.size() - 5), c).dump();
                        first = false;
                    }
                    marks += (marks.empty() ? "" : ",") + json_escape(n.substr(0, n.size() - 5)) + ":" + json_escape(m);
                    seen.push_back(n);
                }
            for (const std::string &n : chat_files(dir)) {
                if (std::find(seen.begin(), seen.end(), n) != seen.end()) continue;
                const std::string m = chat_mark(dir + "/" + n);
                if (!only_marks) {
                    const std::string text = read_whole(dir + "/" + n);
                    Json c;
                    if (!Json::parse(text, c) || c.type != Json::OBJ) continue;
                    out += (first ? "" : ",") + text;
                    first = false;
                }
                marks += (marks.empty() ? "" : ",") + json_escape(n.substr(0, n.size() - 5)) + ":" + json_escape(m);
            }
            send_file(fd, 200, "application/json", (only_marks ? "{" : out + "],") + "\"marks\":{" + marks + "}}", false);
        } else if (p.method == "DELETE") {
            for (const std::string &n : chat_files(dir)) unlink((dir + "/" + n).c_str());
            if (!tui.empty()) {
                for (const std::string &n : chat_files(tui)) unlink((tui + "/" + n).c_str());
                log_line("ts5: the owner deleted all their chats, ts5-tui's too (%s)", p.client.c_str());
            }
            send_file(fd, 200, "application/json", "{\"deleted\":true}", false);
        } else {
            send_error(fd, 405, "GET or DELETE");
        }
        return true;
    }
    if (!good_chat_id(id)) {
        send_error(fd, 400, "a chat's id is 1 to 64 letters, digits, - and _");
        return true;
    }
    const std::string file = dir + "/" + id + ".json";
    const std::string tfile = tui.empty() ? "" : tui + "/" + id + ".json";
    // Where it is written: ts5-tui's folder for the owner (else the user's).
    const std::string wdir = tui.empty() ? dir : tui, wfile = tui.empty() ? file : tfile;
    if (p.method == "PUT") {
        std::lock_guard<std::mutex> lk(chats_mu);
        Json c;
        if (p.body.size() > kChatMax || !Json::parse(p.body, c) || c.type != Json::OBJ || c["id"].str() != id) {
            send_error(fd, 400, "a chat is a JSON object with its id, at most 4 MB");
            return true;
        }
        // A user's limits (so many chats, so much room) aren't the owner's
        // in ts5-tui's own folder: ts5-tui keeps as many there as it likes,
        // and they aren't the chat page's to refuse a new one for.
        if (tui.empty() && access(wfile.c_str(), F_OK) != 0 && (int)chat_files(wdir).size() >= kChatsMax) {
            send_error(fd, 400, "a user keeps at most 1000 chats: delete some first");
            return true;
        }
        std::string body = p.body;
        if (!tui.empty()) {
            Json old;
            const bool had = Json::parse(read_whole(tfile), old) && old.type == Json::OBJ && old["turns"].type == Json::ARR;
            body = tui_chat(c, had ? &old : nullptr, o.model);
        }
        size_t total = body.size();
        if (tui.empty())
            for (const std::string &n : chat_files(wdir)) {
                struct stat st {};
                if (n != id + ".json" && stat((wdir + "/" + n).c_str(), &st) == 0) total += (size_t)st.st_size;
            }
        if (tui.empty() && total > chats_max_bytes()) {
            send_error(fd, 400, "a user's chats take at most " + std::to_string(chats_max_bytes() >> 20) +
                                    " MB on the machine: delete some first");
            return true;
        }
        make_dirs(wdir);
        if (!write_whole(wfile, body)) {
            send_error(fd, 500, "the chat could not be written on the machine");
            return true;
        }
        if (!tui.empty()) unlink(file.c_str());  // kept apart before: ts5-tui's now
        send_file(fd, 200, "application/json", "{\"saved\":true}", false);
    } else if (p.method == "DELETE") {
        unlink(file.c_str());
        if (!tfile.empty()) unlink(tfile.c_str());
        send_file(fd, 200, "application/json", "{\"deleted\":true}", false);
    } else if (p.method == "GET") {
        Json c;
        if (!tfile.empty() && Json::parse(read_whole(tfile), c) && c.type == Json::OBJ && c["turns"].type == Json::ARR) {
            send_file(fd, 200, "application/json", page_chat(id, c).dump(), false);
            return true;
        }
        const std::string text = read_whole(file);
        if (text.empty()) send_error(fd, 404, "no such chat");
        else send_file(fd, 200, "application/json", text, false);
    } else {
        send_error(fd, 405, "GET, PUT or DELETE");
    }
    return true;
}

// ---- notifications: Web Push to a user's devices (webpush.h) ----
//
// A device turns them on in the chat page's Settings: the page subscribes
// with this machine's key (GET /ts5/push) and hands its subscription here
// (POST), kept in user_dir(USER)/push.json. An answer that ends while its
// page is away (a phone's app sent to the background: POST /ts5/away) is
// kept in its chat here, and told to every device of that user, with the
// count for the icon's badge (answers told since the page was last seen).

struct PushUser {
    std::set<std::string> away;     // chats whose page went away while their answer came
    // Chats with an answer coming, and how many requests each has (one
    // asked again while its first still waits for the engine: the first's
    // end isn't the chat's).
    std::map<std::string, int> working;
    int unseen = 0;
    // The browser terminal: how many are open, and whether it went away (its
    // page in the background, or the last one closed). ts5-tui's answers that
    // end then are told (POST /ts5/push {"tell"}). Away is kept in the
    // user's folder too (term_known: read from there): the server starts
    // again with every model ts5-tui loads, and what it knew went with it.
    int term_open = 0;
    bool term_away = false, term_known = false;
};
std::mutex push_mu;
std::map<std::string, PushUser> push_users;  // by user id

// The user's browser terminal is away (push_mu held).
bool term_away(const WebOptions &o, PushUser &u, const std::string &user) {
    if (!u.term_known) {
        u.term_known = true;
        u.term_away = access((o.user_dir(user) + "/terminal-away").c_str(), F_OK) == 0;
    }
    return u.term_away;
}
void term_set_away(const WebOptions &o, PushUser &u, const std::string &user, bool away) {
    if (term_away(o, u, user) == away) return;
    u.term_away = away;
    const std::string file = o.user_dir(user) + "/terminal-away";
    if (away) {
        make_dirs(o.user_dir(user));
        write_whole(file, "\n");
    } else {
        unlink(file.c_str());
    }
}

void push_terminal(const WebOptions &o, const std::string &user, bool open) {
    std::lock_guard<std::mutex> l(push_mu);
    PushUser &u = push_users[user];
    if (open) {
        u.term_open++;
        term_set_away(o, u, user, false);
        u.unseen = 0;
    } else if (--u.term_open <= 0) {
        u.term_open = 0;
        term_set_away(o, u, user, true);
    }
}

constexpr size_t kPushDevices = 10;  // a user's subscriptions kept, the newest

// A device with notifications on: its subscription, the name it gave, and
// which Home Screen app it is ("chat", "terminal"; "" a browser's tab, or a
// device from before apps were told apart).
struct PushDevice {
    PushSubscription sub;
    std::string name, app;
};

// Where this machine's notifications key is: beside users/ (the keys' data).
std::string push_key_path(const WebOptions &o) {
    const std::string ud = o.user_dir("owner");
    const size_t at = ud.rfind("/users/");
    const std::string base = at == std::string::npos ? ud : ud.substr(0, at);
    make_dirs(base);
    return base + "/push-key.pem";
}

std::vector<PushDevice> push_subs(const WebOptions &o, const std::string &user) {
    std::vector<PushDevice> out;
    Json j;
    if (!Json::parse(read_whole(o.user_dir(user) + "/push.json"), j) || j.type != Json::OBJ) return out;
    for (const Json &x : j["devices"].a) {
        PushDevice d;
        d.sub.endpoint = x["endpoint"].str();
        d.sub.p256dh = x["p256dh"].str();
        d.sub.auth = x["auth"].str();
        d.name = x["name"].str();
        d.app = x["app"].str();
        if (d.sub.endpoint.empty()) continue;
        out.push_back(d);
    }
    return out;
}

void push_keep(const WebOptions &o, const std::string &user, const std::vector<PushDevice> &devices) {
    Json j = Json::object(), arr;
    arr.type = Json::ARR;
    for (const PushDevice &d : devices) {
        Json x = Json::object();
        x.set("endpoint", Json::string(d.sub.endpoint));
        x.set("p256dh", Json::string(d.sub.p256dh));
        x.set("auth", Json::string(d.sub.auth));
        x.set("name", Json::string(d.name));
        if (!d.app.empty()) x.set("app", Json::string(d.app));
        arr.a.push_back(x);
    }
    j.set("devices", arr);
    make_dirs(o.user_dir(user));
    write_whole(o.user_dir(user) + "/push.json", j.dump() + "\n");
}

// Sends payload to the user's devices; one gone (404, 410) is forgotten.
// kind: "chat" (an answer of the chat page's) goes to all but the terminal
// app, "terminal" (one of ts5-tui's) to all but the chat app, so a phone
// with both Home Screen apps is told once; to every device where that
// leaves none, and for "". sent: how many took it; why: the first failure.
void push_send_all(const WebOptions &o, const std::string &user, const std::string &payload, const std::string &kind,
                   int &sent, int &devices, std::string &why) {
    sent = 0;
    std::vector<PushDevice> all, subs;
    {
        std::lock_guard<std::mutex> l(push_mu);
        all = push_subs(o, user);
    }
    const std::string other = kind == "chat" ? "terminal" : kind == "terminal" ? "chat" : "";
    for (const PushDevice &d : all)
        if (other.empty() || d.app != other) subs.push_back(d);
    if (subs.empty()) subs = all;
    devices = (int)subs.size();
    std::vector<std::string> gone;
    const std::string key = push_key_path(o);
    for (const PushDevice &d : subs) {
        int status = 0;
        std::string w;
        if (webpush_send(key, d.sub, payload, status, w)) sent++;
        else {
            if (status == 404 || status == 410) gone.push_back(d.sub.endpoint);
            if (why.empty()) why = w;
        }
    }
    if (!gone.empty()) {
        std::lock_guard<std::mutex> l(push_mu);
        std::vector<PushDevice> keep;
        for (const PushDevice &d : push_subs(o, user))
            if (std::find(gone.begin(), gone.end(), d.sub.endpoint) == gone.end()) keep.push_back(d);
        push_keep(o, user, keep);
    }
    log_line("ts5: notifications: %d of %d device%s of %s's took it%s%s%s", sent, devices, devices == 1 ? "" : "s",
             user.c_str(), gone.empty() ? "" : " (", gone.empty() ? "" : (std::to_string(gone.size()) + " gone, forgotten").c_str(),
             gone.empty() ? (why.empty() ? "" : (": " + why).c_str()) : ")");
}

// The word this machine's ts5-tui shows it is that (POST /ts5/push
// {"tell", "secret"}): 32 random bytes in hex, made once, in a file only
// this server's user reads, beside the notifications key (ts5-tui's own
// data folder, for the server it starts); the background service's also in
// /run/tokenstation, for its group (the owner's ts5-tui is in it).
std::string run_dir();
bool as_service();
std::string tell_secret(const WebOptions &o) {
    static std::mutex mu;
    static std::string secret;
    std::lock_guard<std::mutex> l(mu);
    if (!secret.empty()) return secret;
    const std::string ud = o.user_dir("owner");
    const size_t at = ud.rfind("/users/");
    const std::string base = at == std::string::npos ? ud : ud.substr(0, at);
    const std::string file = base + "/tell.key";
    const auto good = [](std::string t) {
        while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
        if (t.size() != 64) return std::string();
        for (char c : t)
            if (!isxdigit((unsigned char)c)) return std::string();
        return t;
    };
    std::string have = good(read_whole(file));
    if (have.empty()) {
        unsigned char raw[32];
        FILE *f = fopen("/dev/urandom", "rb");
        const bool ok = f && fread(raw, 1, sizeof raw, f) == sizeof raw;
        if (f) fclose(f);
        if (!ok) return "";
        static const char hex[] = "0123456789abcdef";
        for (unsigned char c : raw) {
            have += hex[c >> 4];
            have += hex[c & 15];
        }
        make_dirs(base);
        if (!write_whole(file, have + "\n")) return "";
    }
    if (as_service()) {
        const std::string shared = run_dir() + "/tell.key";
        if (good(read_whole(shared)) != have && write_whole(shared, have + "\n")) chmod(shared.c_str(), 0640);
    }
    secret = have;
    return secret;
}
// (Every byte looked at: how long it takes says nothing of where they differ.)
bool same_secret(const std::string &a, const std::string &b) {
    if (a.empty() || a.size() != b.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < a.size(); i++) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}

// s cut to at most n bytes, on a character's start, with "…" when cut.
std::string cut_text(const std::string &s, size_t n) {
    if (s.size() <= n) return s;
    size_t e = n;
    while (e > 0 && ((unsigned char)s[e] & 0xC0) == 0x80) e--;
    return s.substr(0, e) + "…";
}

// The answer kept in its chat on the machine (the page went away before it
// came); the chat's title, "" when there is no such chat.
std::string keep_answer(const WebOptions &o, const Server::Grant &g, const Server::AnswerEnd &a) {
    std::lock_guard<std::mutex> l(chats_mu);
    Json m = Json::object();
    m.set("role", Json::string("assistant"));
    m.set("content", Json::string(a.content));
    if (!a.reasoning.empty()) m.set("thinking", Json::string(a.reasoning));
    m.set("done", jbool(true));
    m.set("model", Json::string(a.model));
    if (a.tokens > 0) {
        char b[64];
        snprintf(b, sizeof b, "%lld tokens · %.1f tok/s", (long long)a.tokens, a.tps);
        m.set("stats", Json::string(b));
    }
    if (!a.error.empty()) m.set("error", Json::string(a.error));
    // What the chat holds now, the page's way: an answer to fill in or add.
    auto with_answer = [&](Json &page) {
        auto at = page.o.find("messages");
        if (at == page.o.end() || at->second.type != Json::ARR || at->second.a.empty()) return false;
        std::vector<Json> &ms = at->second.a;
        Json &last = ms.back();
        if (last["role"].str() == "user") ms.push_back(m);
        else if (last["role"].str() == "assistant" && !last["done"].boolean(false)) last = m;
        else return false;  // answered already (the page kept it)
        return true;
    };
    const std::string tui = tui_chats_dir(o, g);
    if (!tui.empty()) {
        const std::string tfile = tui + "/" + a.chat + ".json";
        Json old;
        if (Json::parse(read_whole(tfile), old) && old.type == Json::OBJ && old["turns"].type == Json::ARR) {
            Json page = page_chat(a.chat, old);
            if (with_answer(page)) write_whole(tfile, tui_chat(page, &old, o.model));
            return old["title"].str("A chat");
        }
    }
    const std::string file = o.user_dir(g.user) + "/chats/" + a.chat + ".json";
    Json page;
    if (!Json::parse(read_whole(file), page) || page.type != Json::OBJ) return "";
    if (with_answer(page)) write_whole(file, page.dump());
    return page["title"].str("A chat");
}

void page_answers(Server &s, const WebOptions &o) {
    Server::PageAnswers pa;
    pa.away = [](const Server::Grant &g, const std::string &chat) {
        std::lock_guard<std::mutex> l(push_mu);
        auto it = push_users.find(g.user);
        return it != push_users.end() && it->second.away.count(chat) > 0;
    };
    pa.start = [](const Server::Grant &g, const std::string &chat) {
        std::lock_guard<std::mutex> l(push_mu);
        PushUser &u = push_users[g.user];
        u.working[chat]++;
        u.away.erase(chat);  // asked from the page: it's there
    };
    pa.end = [o](const Server::Grant &g, const Server::AnswerEnd &a) {
        bool away;
        int badge = 0;
        {
            std::lock_guard<std::mutex> l(push_mu);
            PushUser &u = push_users[g.user];
            auto w = u.working.find(a.chat);
            const bool more = w != u.working.end() && --w->second > 0;
            if (w != u.working.end() && !more) u.working.erase(w);
            // (Another request of the chat's still to come: the page's word
            // that it went away is that one's too.)
            away = more ? u.away.count(a.chat) > 0 : u.away.erase(a.chat) > 0;
            // Kept: an answer that went on without its connection (the page,
            // back by now or not, gets it from here), and one whose page is
            // away. Not one cut short (a Stop, even as the page went): the
            // page has what it had. Told: while the page is away.
            if (!a.went_on && (!away || a.left)) return;
            if (away) badge = ++u.unseen;
        }
        std::thread([o, g, a, away, badge] {
            const std::string title = keep_answer(o, g, a);
            if (!away) return;
            Json j = Json::object();
            j.set("title", Json::string(cut_text(title.empty() ? std::string("TokenStation5") : title, 80)));
            std::string body = a.content;
            for (char &c : body)
                if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            if (!a.error.empty()) body = "It stopped: " + a.error;
            else if (body.find_first_not_of(' ') == std::string::npos) body = "The answer is ready.";
            j.set("body", Json::string(cut_text(body.substr(std::min(body.size(), body.find_first_not_of(' '))), 160)));
            j.set("chat", Json::string(a.chat));
            j.set("badge", jnum(badge));
            int sent = 0, devices = 0;
            std::string why;
            push_send_all(o, g.user, j.dump(), "chat", sent, devices, why);
        }).detach();
    };
    s.set_page_answers(pa);
}

// /ts5/push: GET this machine's key for subscribing and how many of the
// user's devices have notifications on; POST {"subscription": {endpoint,
// keys: {p256dh, auth}}, "name"} turns them on for one, POST {"test": true}
// sends one to each; DELETE {"endpoint"} turns one off. /ts5/away: POST
// {"chat", "away"} from the page as it goes to the background and comes
// back (back: the badge's count starts again); GET: the chats with an answer
// coming. A key's user's own, as its chats are.
bool push_pages(const WebOptions &o, const Server::Page &p, int fd) {
    const bool push = p.path == "/ts5/push", away = p.path == "/ts5/away";
    if (!push && !away) return false;
    if (!o.admit || !o.user_dir) {
        send_error(fd, 404, "this server keeps no users");
        return true;
    }
    Server::Grant g;
    if (!o.admit(p.key, p.client, g)) {
        send_error(fd, 401, p.key.empty() ? "This needs an API key." : "That API key isn't one of this machine's, or it is off.");
        return true;
    }
    Json req;
    if (p.method != "GET" && !p.body.empty() && (!Json::parse(p.body, req) || req.type != Json::OBJ)) {
        send_error(fd, 400, "a JSON object, please");
        return true;
    }
    // ts5-tui's answer has ended: told to the user's devices while their
    // browser terminal is away. Asked with a key, or by ts5-tui on this
    // machine without one, as the owner, with the word only it can read
    // (tell_secret): not by any program here that can reach the port (a
    // command the model runs, a page its browser opens).
    if (push && p.method == "POST" && req["tell"].type == Json::OBJ) {
        const bool keyed = !p.key.empty() && !g.key.empty();
        if (!keyed && !(g.owner && same_secret(req["secret"].str(), tell_secret(o)))) {
            send_error(fd, 403, "A notification is asked for with a key, or by this machine's ts5-tui.");
            return true;
        }
        // "always": an unattended task's end (linux/tui/task.h), told
        // whether the terminal is away or not, to every device.
        const bool always = req["tell"]["always"].boolean(false);
        int badge = 0;
        {
            std::lock_guard<std::mutex> l(push_mu);
            PushUser &u = push_users[g.user];
            if (always || term_away(o, u, g.user)) badge = ++u.unseen;
        }
        if (badge > 0) {
            Json j = Json::object();
            const std::string title = req["tell"]["title"].str();
            std::string body = req["tell"]["body"].str();
            for (char &c : body)
                if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            const size_t from = body.find_first_not_of(' ');
            body = from == std::string::npos ? "The answer is ready." : body.substr(from);
            j.set("title", Json::string(cut_text(title.empty() ? std::string("ts5-tui") : title, 80)));
            j.set("body", Json::string(cut_text(body, 160)));
            j.set("url", Json::string(always ? "/ts5/controls#tasks" : "/ts5/terminal"));
            j.set("badge", jnum(badge));
            std::thread([o, user = g.user, payload = j.dump(), always] {
                int sent = 0, devices = 0;
                std::string why;
                push_send_all(o, user, payload, always ? "" : "terminal", sent, devices, why);
            }).detach();
        }
        Json out = Json::object();
        out.set("told", jbool(badge > 0));
        send_file(fd, 200, "application/json", out.dump(), false);
        return true;
    }
    // (The browser terminal's page on this machine has the owner's say without a key.)
    if ((p.key.empty() || g.key.empty()) && !(away && g.owner && req["terminal"].boolean(false))) {
        send_error(fd, 403, "Notifications are for a key's user: open the chat page with your key.");
        return true;
    }
    if (away) {
        Json out = Json::object(), arr;
        arr.type = Json::ARR;
        {
            std::lock_guard<std::mutex> l(push_mu);
            PushUser &u = push_users[g.user];
            if (p.method == "POST" && req["terminal"].boolean(false)) {
                // The browser terminal's page went to the background, or came back.
                term_set_away(o, u, g.user, req["away"].boolean(false));
                if (!u.term_away) u.unseen = 0;
            } else if (p.method == "POST") {
                const std::string chat = req["chat"].str();
                const bool going = req["away"].boolean(false);
                if (going && good_chat_id(chat) && u.working.count(chat)) u.away.insert(chat);
                if (!going) {
                    if (!chat.empty()) u.away.erase(chat);
                    u.unseen = 0;
                }
            } else if (p.method != "GET") {
                send_error(fd, 405, "GET or POST");
                return true;
            }
            for (const auto &c : u.working) arr.a.push_back(Json::string(c.first));
        }
        out.set("working", arr);
        send_file(fd, 200, "application/json", out.dump(), false);
        return true;
    }
    const std::string key = push_key_path(o);
    if (p.method == "GET") {
        std::string pub, why;
        if (!vapid_key(key, pub, why)) {
            send_error(fd, 500, why);
            return true;
        }
        Json out = Json::object();
        out.set("key", Json::string(pub));
        std::lock_guard<std::mutex> l(push_mu);
        out.set("devices", jnum((double)push_subs(o, g.user).size()));
        send_file(fd, 200, "application/json", out.dump(), false);
        return true;
    }
    if (p.method == "POST" && req["test"].boolean(false)) {
        Json j = Json::object();
        j.set("title", Json::string("TokenStation5"));
        j.set("body", Json::string("Notifications are on. You get one when an answer finishes while you're away."));
        j.set("badge", jnum(0));
        int sent = 0, devices = 0;
        std::string why;
        push_send_all(o, g.user, j.dump(), "", sent, devices, why);
        Json out = Json::object();
        out.set("sent", jnum(sent));
        out.set("devices", jnum(devices));
        if (sent < devices) out.set("error", Json::string(why));
        send_file(fd, sent > 0 || devices == 0 ? 200 : 502, "application/json", out.dump(), false);
        return true;
    }
    if (p.method == "POST" || p.method == "DELETE") {
        const Json &sub = req["subscription"];
        PushDevice d;
        PushSubscription &s = d.sub;
        s.endpoint = p.method == "POST" ? sub["endpoint"].str() : req["endpoint"].str();
        s.p256dh = sub["keys"]["p256dh"].str();
        s.auth = sub["keys"]["auth"].str();
        std::string raw_key, raw_auth;
        if (p.method == "POST" &&
            (!push_endpoint_ok(s.endpoint) || !b64url_decode(s.p256dh, raw_key) || raw_key.size() != 65 ||
             !b64url_decode(s.auth, raw_auth) || raw_auth.size() != 16)) {
            send_error(fd, 400, push_endpoint_ok(s.endpoint)
                                    ? "the subscription's keys aren't a browser's (p256dh: 65 bytes, auth: 16)"
                                    : "that isn't a push service this machine sends to (Apple's, Google's, Mozilla's, Microsoft's)");
            return true;
        }
        d.name = req["name"].str();
        if (d.name.size() > 60) d.name = cut_text(d.name, 60);
        d.app = req["app"].str();
        if (d.app != "chat" && d.app != "terminal") d.app.clear();
        std::lock_guard<std::mutex> l(push_mu);
        std::vector<PushDevice> keep;
        for (const PushDevice &x : push_subs(o, g.user))
            if (x.sub.endpoint != s.endpoint) keep.push_back(x);
        if (p.method == "POST") {
            keep.push_back(d);
            while (keep.size() > kPushDevices) keep.erase(keep.begin());
        }
        push_keep(o, g.user, keep);
        log_line("ts5: notifications: %s %s them for %s (%s)", g.user_name.empty() ? g.user.c_str() : g.user_name.c_str(),
                 p.method == "POST" ? "turned on" : "turned off", p.client.c_str(),
                 p.method == "POST" ? (d.name.empty() ? "a device" : d.name.c_str()) : "a device");
        Json out = Json::object();
        out.set("devices", jnum((double)keep.size()));
        send_file(fd, 200, "application/json", out.dump(), false);
        return true;
    }
    send_error(fd, 405, "GET, POST or DELETE");
    return true;
}

// ---- the phone's controls: /ts5/control, to ts5-tui (linux/tui/control.h) ----

std::string user_data_dir() {
    if (const char *e = getenv("TS5_DATA_DIR"); e && *e) return e;
    if (const char *x = getenv("XDG_DATA_HOME"); x && *x == '/') return std::string(x) + "/tokenstation";
    const char *h = getenv("HOME");
    return std::string(h && *h ? h : "") + "/.local/share/tokenstation";
}

std::string run_dir() {
    const char *e = getenv("TS5_RUN_DIR");
    return e && *e == '/' ? e : "/run/tokenstation";
}

// The background service (its own folder in /run, or started by systemd):
// it can't start the owner's ts5-tui, and looks for it in /run first.
bool as_service() {
    struct stat st{};
    return getenv("INVOCATION_ID") || (stat(run_dir().c_str(), &st) == 0 && st.st_uid == getuid());
}

// Where ts5-tui answers, in the order asked: this user's own first (a ts5
// serve the owner runs), the service's (/run) first for the service.
std::vector<std::string> control_sockets() {
    if (const char *e = getenv("TS5_CONTROL_SOCKET"); e && *e) return {e};
    const std::string own = user_data_dir() + "/control.sock", shared = run_dir() + "/control.sock";
    if (as_service()) return {shared, own};
    return {own, shared};
}

// One request to the ts5-tui at path and its answer; false when none
// answers there.
bool control_ask(const std::string &path, const std::string &line, std::string &out) {
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    if (path.size() >= sizeof a.sun_path) return false;
    memcpy(a.sun_path, path.c_str(), path.size() + 1);
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return false;
    if (connect(fd, (sockaddr *)&a, sizeof a) != 0) {
        close(fd);
        return false;
    }
    // ts5-tui answers between frames; a page it is busy drawing waits.
    timeval tv{30, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    const std::string req = line + "\n";
    bool ok = true;
    for (size_t at = 0; ok && at < req.size();) {
        const ssize_t n = send(fd, req.data() + at, req.size() - at, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        ok = n > 0;
        if (ok) at += (size_t)n;
    }
    out.clear();
    char b[16384];
    while (ok) {
        const ssize_t n = recv(fd, b, sizeof b, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        out.append(b, (size_t)n);
        if (out.size() > (16u << 20)) break;
    }
    close(fd);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return ok && !out.empty();
}

std::string tui_path() {
    const std::string c = tui_command();  // "'DIR/ts5-tui' --tmux" or "ts5-tui --tmux"
    if (c[0] == '\'') return c.substr(1, c.find('\'', 1) - 1);
    return "ts5-tui";
}

// ts5-tui in its tmux session, in the background (ts5-tui --start); as
// the browser terminal would have it, so it shows the same one later.
bool start_tui(std::string &why) {
    static std::mutex mu;
    std::lock_guard<std::mutex> l(mu);
    const pid_t pid = fork();
    if (pid < 0) {
        why = strerror(errno);
        return false;
    }
    if (pid == 0) {
        // The server's secrets stay with the server.
        unsetenv("TS5_API_KEY");
        unsetenv("TS5_TS_AUTH_KEY");
        unsetenv("TS5_CONTROL_SOCKET");
        if (!getenv("LANG") && !getenv("LC_ALL")) setenv("LANG", "C.UTF-8", 1);
        setsid();
        const int null = open("/dev/null", O_RDWR);
        if (null >= 0) {
            dup2(null, 0);
            dup2(null, 1);
            dup2(null, 2);
        }
        signal(SIGPIPE, SIG_DFL);
        const std::string tui = tui_path();
        execlp(tui.c_str(), "ts5-tui", "--start", (char *)nullptr);
        _exit(127);
    }
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {
    }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        why = WIFEXITED(st) && WEXITSTATUS(st) == 127 ? "ts5-tui isn't installed here"
                                                      : "ts5-tui --start failed (it needs tmux)";
        return false;
    }
    return true;
}

// One request to the running ts5-tui (a JSON line) and its answer; with
// none running, one is started for it (not by the service). False with why.
bool control_call(const std::string &line, std::string &out, std::string &why) {
    for (const std::string &path : control_sockets())
        if (control_ask(path, line, out)) return true;
    if (as_service()) {
        why = "ts5-tui isn't running on the machine: start it there (it can stay open in its tmux session), then "
              "the controls reach it.";
        return false;
    }
    if (!start_tui(why)) {
        why = "ts5-tui couldn't be started: " + why + ".";
        return false;
    }
    const std::string own = control_sockets()[0];
    for (int i = 0; i < 50; i++) {
        usleep(200000);
        if (control_ask(own, line, out)) {
            log_line("ts5: controls: started ts5-tui for them (tmux session \"ts5\")");
            return true;
        }
    }
    why = "ts5-tui was started but doesn't answer yet: try again in a moment.";
    return false;
}

// /ts5/control: POST {"do": ...} (GET: {"do": "state"}), the owner's key
// only (always a key, on this machine too), from this machine or the tailnet: handed to ts5-tui, its answer
// handed back. ts5-tui not running: started for it, unless this is the
// service (it can't start the owner's).
void control(const WebOptions &o, const Server::Page &p, int fd) {
    if (!o.admit) {
        send_error(fd, 404, "this server keeps no users");
        return;
    }
    Server::Grant g;
    // A key of this machine's, even from this machine (which gets into the
    // API without one): another user of this computer is not the owner.
    if (!o.admit(p.key, p.client, g) || g.key.empty()) {
        send_error(fd, 401, p.key.empty() ? "The controls need your key, on this machine too."
                                          : "That API key isn't one of this machine's, or it is off.");
        return;
    }
    if (!g.owner) {
        log_line("ts5: controls: %s: the key %s is %s's, not the owner's: refused", p.client.c_str(), g.key.c_str(),
                 g.user_name.c_str());
        send_error(fd, 403, "The controls are the owner's alone.");
        return;
    }
    if (!terminal_client(p.client)) {
        log_line("ts5: controls: %s asked for them, refused: only this machine and the tailnet's devices may",
                 p.client.c_str());
        send_error(fd, 403, "The controls are only for this machine and your tailnet's devices (Tailscale).");
        return;
    }
    if (p.method != "GET" && p.method != "POST") {
        send_error(fd, 405, "GET or POST");
        return;
    }
    Json req;
    if (p.method == "GET") {
        req = Json::object();
        req.set("do", Json::string("state"));
    } else if (!Json::parse(p.body, req) || req.type != Json::OBJ) {
        send_error(fd, 400, "a request is a JSON object: {\"do\": ...}");
        return;
    }
    const std::string what = req["do"].str("state");
    if (what != "state") {
        // What was done, never the words a setting was given (a key, say).
        // (A user's or key's: its id, and the name a new one gets.)
        std::string on = req["model"].str(req["mode"].str(req["key"].str()));
        for (const char *f : {"id", "user", "name", "label"})
            if (!req[f].str().empty()) on += (on.empty() ? "" : " ") + req[f].str();
        for (char &c : on)
            if ((unsigned char)c < 0x20 || c == 0x7f) c = ' ';  // one line in the log, whatever was sent
        if (on.size() > 120) {  // (at a character's start)
            on.resize(120);
            while (!on.empty() && ((unsigned char)on.back() & 0xC0) == 0x80) on.pop_back();
            if (!on.empty() && (unsigned char)on.back() >= 0xC0) on.pop_back();
        }
        log_line("ts5: controls: %s: %s%s%s", p.client.c_str(), what.c_str(), on.empty() ? "" : " ", on.c_str());
    }
    std::string out, why;
    if (control_call(req.dump(), out, why)) send_file(fd, 200, "application/json", out, false);
    else send_error(fd, 503, why);
}

// The owner and a user who may switch (Server::Grant::may_switch): which
// of its models are here, and the one loaded.
//   GET /ts5/switch                 {"models": [{"name", "loaded"}], "loading"}
//   POST /ts5/switch {"model": M}   ts5-tui loads M (it asks nothing: a
//                                   model it would ask the owner about is
//                                   refused); then as GET.
// Through the owner's ts5-tui, as the controls. A user's switch (not the
// owner's) waits for an answer being written to end, up to a minute, and
// is refused if it hasn't by then: the switch starts the server again, and
// would cut that answer off for whoever asked for it.
void switch_model(Server &srv, const WebOptions &o, const Server::Page &p, int fd) {
    if (!o.admit) {
        send_error(fd, 404, "this server keeps no users");
        return;
    }
    Server::Grant g;
    if (!o.admit(p.key, p.client, g) || g.key.empty()) {
        send_error(fd, 401, p.key.empty() ? "This needs your key." : "That API key isn't one of this machine's, or it is off.");
        return;
    }
    if (!g.owner && !g.may_switch) {
        send_error(fd, 403, "Only the owner switches the model here: " + g.user_name + " may not.");
        return;
    }
    if (p.method != "GET" && p.method != "POST") {
        send_error(fd, 405, "GET or POST");
        return;
    }
    std::string out, why;
    Json st;
    if (!control_call("{\"do\":\"state\"}", out, why) || !Json::parse(out, st)) {
        send_error(fd, 503, why.empty() ? "ts5-tui's answer wasn't one" : why);
        return;
    }
    if (p.method == "POST") {
        Json req;
        const std::string want = Json::parse(p.body, req) ? req["model"].str() : "";
        bool here = false;
        for (const Json &m : st["models"].a) here |= m["name"].str() == want;
        if (want.empty() || !here || !model_allowed(g.models, want)) {
            send_error(fd, 400, want.empty() ? "which model? {\"model\": NAME}"
                                             : want + " isn't one of the models " + (g.owner ? "here" : g.user_name + " may use here"));
            return;
        }
        if (!g.owner) {
            const auto writing = [&srv] {
                const Server::Activity a = srv.activity();
                return a.generating || a.phase != 0 || a.waiting > 0;
            };
            const char *e = getenv("TS5_SWITCH_WAIT");  // seconds (tests)
            const double until = now_s() + (e && *e ? atof(e) : 60);
            while (writing() && now_s() < until) usleep(200 * 1000);
            if (writing()) {
                send_error(fd, 409, "An answer is still being written here: try again when it has ended.");
                return;
            }
        }
        Json ask = Json::object();
        ask.set("do", Json::string("load"));
        ask.set("model", Json::string(want));
        ask.set("quiet", Json::string("1"));
        ask.set("by", Json::string(g.owner ? "" : g.user_name));
        // On the user's own list (or the owner's pick): chosen by the owner.
        if (g.owner || !g.models.empty()) ask.set("listed", Json::string("1"));
        if (!control_call(ask.dump(), out, why) || !Json::parse(out, st)) {
            send_error(fd, 503, why.empty() ? "ts5-tui's answer wasn't one" : why);
            return;
        }
        if (st["error"].type == Json::STR) {
            send_error(fd, 409, st["error"].str());
            return;
        }
        log_line("ts5: switch: %s: %s loads %s", p.client.c_str(), g.owner ? "the owner" : g.user_name.c_str(), want.c_str());
    }
    Json list = Json::object(), arr;
    arr.type = Json::ARR;
    for (const Json &m : st["models"].a) {
        if (!model_allowed(g.models, m["name"].str())) continue;
        Json x = Json::object();
        x.set("name", Json::string(m["name"].str()));
        Json b;
        b.type = Json::BOOL;
        b.b = m["loaded"].boolean(false);
        x.set("loaded", b);
        arr.a.push_back(x);
    }
    list.set("models", arr);
    list.set("loading", Json::string(st["loading"]["model"].str()));
    send_file(fd, 200, "application/json", list.dump(), false);
}

}  // namespace

void attach_web(Server &s, const WebOptions &o) {
    if (o.admit && o.user_dir) {
        page_answers(s, o);
        tell_secret(o);  // there before ts5-tui's first answer ends
    }
    s.set_pages([o, &s](const Server::Page &p, int fd) {
        // The address apps are given (…/v1), opened in a browser: the chat page.
        if ((p.path == "/v1" || p.path == "/v1/") && p.method == "GET" &&
            p.header("accept").find("text/html") != std::string::npos) {
            send_text(fd, "HTTP/1.1 302 Found\r\nLocation: /\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return true;
        }
        // Icons a browser asks for by itself (Safari's home-screen ones too):
        // the logo, with no key, as the pages' own links give it.
        if (p.method == "GET" && (p.path == "/favicon.ico" || p.path.rfind("/apple-touch-icon", 0) == 0) &&
            serve_asset(fd, p, "icon.png", false))
            return true;
        // A page's "save the key to my passwords" form, sent for real: Safari
        // (iCloud Keychain) offers to save a password only when a form with
        // it is sent and the page moves on. Its body (the key) is not read,
        // kept or logged; the browser goes back to the page it came from.
        if (p.path == "/ts5/keep") {
            if (p.method != "POST") {
                send_error(fd, 405, "POST only");
                return true;
            }
            const char *to = p.query.find("back=terminal") != std::string::npos ? "/ts5/terminal" : "/";
            web_log(p, 303, 0);
            send_text(fd, std::string("HTTP/1.1 303 See Other\r\nLocation: ") + to +
                              "\r\nCache-Control: no-store\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return true;
        }
        if (me_and_chats(o, p, fd)) return true;
        if (push_pages(o, p, fd)) return true;
        // The chat page as a Home Screen app: its service worker at the top
        // (its scope: the whole site) and its manifest.
        if (p.method == "GET" && (p.path == "/sw.js" || p.path == "/manifest.webmanifest") &&
            serve_asset(fd, p, p.path.substr(1), false))
            return true;
        if (p.path == "/ts5/control") {
            control(o, p, fd);
            return true;
        }
        if (p.path == "/ts5/switch") {
            switch_model(s, o, p, fd);
            return true;
        }
        const bool web = p.path == "/" || p.path == "/chat" || p.path.rfind("/ts5/web/", 0) == 0 ||
                         p.path == "/ts5/terminal" || p.path == "/ts5/terminal/socket" || p.path == "/ts5/controls";
        if (!web) return false;
        if (p.path == "/ts5/terminal/socket") {
            terminal(o, p, fd);
            return true;
        }
        if (p.method != "GET") {
            send_error(fd, 405, "GET only");
            return true;
        }
        bool ok;
        if (p.path == "/" || p.path == "/chat") ok = serve_asset(fd, p, "chat.html", true);
        else if (p.path == "/ts5/terminal") ok = serve_asset(fd, p, "terminal.html", true);
        else if (p.path == "/ts5/controls") ok = serve_asset(fd, p, "controls.html", true);
        else {
            const std::string name = p.path.substr(9);
            ok = name != "chat.html" && name != "terminal.html" && name != "controls.html" && serve_asset(fd, p, name, false);
        }
        if (!ok) {
            send_error(fd, 404, "not found: " + p.path);
            web_log(p, 404, 0);
        }
        return true;
    });
}

}  // namespace ts5
