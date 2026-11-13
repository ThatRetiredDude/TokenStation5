#include "web_serve.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <pty.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "json.h"
#include "log.h"
#include "mbedtls/base64.h"
#include "mbedtls/sha1.h"

namespace ts5 {

}  // namespace ts5

// The pages' files, built in (engine/web/web_assets.S).
#define TS5_WEB_FILE(sym) extern "C" const unsigned char sym[], sym##_end[];
TS5_WEB_FILE(ts5_web_chat_html)
TS5_WEB_FILE(ts5_web_terminal_html)
TS5_WEB_FILE(ts5_web_xterm_js)
TS5_WEB_FILE(ts5_web_xterm_css)
TS5_WEB_FILE(ts5_web_addon_fit_js)
TS5_WEB_FILE(ts5_web_xterm_license)
TS5_WEB_FILE(ts5_web_addon_fit_license)
TS5_WEB_FILE(ts5_web_icon_png)
TS5_WEB_FILE(ts5_web_keep_js)
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
    {"xterm.js", ts5_web_xterm_js, ts5_web_xterm_js_end},
    {"xterm.css", ts5_web_xterm_css, ts5_web_xterm_css_end},
    {"addon-fit.js", ts5_web_addon_fit_js, ts5_web_addon_fit_js_end},
    {"xterm-LICENSE", ts5_web_xterm_license, ts5_web_xterm_license_end},
    {"addon-fit-LICENSE", ts5_web_addon_fit_license, ts5_web_addon_fit_license_end},
    {"icon.png", ts5_web_icon_png, ts5_web_icon_png_end},
    {"keep.js", ts5_web_keep_js, ts5_web_keep_js_end},
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
        default: return "Error";
    }
}

// A page or file: no key needed, kept out of other sites' frames, and the
// pages' scripts and connections only from here.
void send_file(int fd, int code, const std::string &type, const std::string &body, bool page) {
    std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\nContent-Type: " + type +
                    "\r\nContent-Length: " + std::to_string(body.size()) +
                    "\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n";
    if (page)
        h += "Cache-Control: no-cache\r\nX-Frame-Options: DENY\r\n"
             "Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; "
             "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'\r\n";
    else
        h += "Cache-Control: max-age=3600\r\n";
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
    return "text/plain; charset=utf-8";
}

bool serve_asset(int fd, const std::string &name, bool page) {
    const WebAsset *a = asset(name);
    if (!a) return false;
    send_file(fd, 200, type_of(name), std::string((const char *)a->data, (size_t)(a->end - a->data)), page);
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
    if (!o.authorize || !o.authorize(sent, p.client, name)) {
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

}  // namespace

void attach_web(Server &s, const WebOptions &o) {
    s.set_pages([o](const Server::Page &p, int fd) {
        // The address apps are given (…/v1), opened in a browser: the chat page.
        if ((p.path == "/v1" || p.path == "/v1/") && p.method == "GET" &&
            p.header("accept").find("text/html") != std::string::npos) {
            send_text(fd, "HTTP/1.1 302 Found\r\nLocation: /\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return true;
        }
        // Icons a browser asks for by itself (Safari's home-screen ones too):
        // the logo, with no key, as the pages' own links give it.
        if (p.method == "GET" && (p.path == "/favicon.ico" || p.path.rfind("/apple-touch-icon", 0) == 0) &&
            serve_asset(fd, "icon.png", false))
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
            send_text(fd, std::string("HTTP/1.1 303 See Other\r\nLocation: ") + to +
                              "\r\nCache-Control: no-store\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return true;
        }
        const bool web = p.path == "/" || p.path == "/chat" || p.path.rfind("/ts5/web/", 0) == 0 ||
                         p.path == "/ts5/terminal" || p.path == "/ts5/terminal/socket";
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
        if (p.path == "/" || p.path == "/chat") ok = serve_asset(fd, "chat.html", true);
        else if (p.path == "/ts5/terminal") ok = serve_asset(fd, "terminal.html", true);
        else {
            const std::string name = p.path.substr(9);
            ok = name != "chat.html" && name != "terminal.html" && serve_asset(fd, name, false);
        }
        if (!ok) send_error(fd, 404, "not found: " + p.path);
        return true;
    });
}

}  // namespace ts5
