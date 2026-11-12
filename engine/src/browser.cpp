// ts5's browser for a model (browser.h).
#include "browser.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <ftw.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "json.h"
#include "net.h"
#include "websearch.h"

namespace ts5 {

namespace {

const size_t npos = std::string::npos;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void sleep_ms(int ms) { usleep((useconds_t)ms * 1000); }

std::string lower(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

bool ends_with(const std::string &s, const char *end) {
    const size_t n = strlen(end);
    return s.size() >= n && s.compare(s.size() - n, n, end) == 0;
}

// s with whatever is not UTF-8 replaced by '?', and no control characters
// but the newline.
std::string clean_text(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        if (c < 0x80) {
            if (c >= 0x20 || c == '\n') out += (char)c;
            else if (c == '\t') out += ' ';
            i++;
            continue;
        }
        const int n = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
        bool ok = n > 0 && i + n <= s.size();
        for (int k = 1; ok && k < n; k++) ok = ((unsigned char)s[i + k] & 0xC0) == 0x80;
        if (ok) out.append(s, i, n), i += n;
        else out += '?', i++;
    }
    return out;
}

// At most max bytes of s on one line, cut between characters.
std::string brief(std::string s, size_t max) {
    for (char &c : s)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    if (s.size() > max) {
        size_t cut = max;
        while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) cut--;
        s = s.substr(0, cut) + "...";
    }
    return s;
}

// ---- Addresses

// The host of an http(s) address, in lower case, without its port.
std::string host_of(const std::string &url) {
    const size_t s = url.find("://");
    if (s == npos) return "";
    const size_t e = url.find_first_of("/?#", s + 3);
    std::string hp = url.substr(s + 3, e == npos ? npos : e - s - 3);
    const size_t at = hp.rfind('@');
    if (at != npos) hp.erase(0, at + 1);
    if (!hp.empty() && hp[0] == '[') {
        const size_t c = hp.find(']');
        return lower(hp.substr(0, c == npos ? npos : c + 1));
    }
    const size_t colon = hp.rfind(':');
    if (colon != npos) hp.resize(colon);
    return lower(hp);
}

// Whether a host is this machine or one on the user's own network: a page
// there is a router's or a service's own, which a model shouldn't be sent
// to by a page it read without the user's word.
bool local_host(const std::string &h) {
    if (h.empty() || h[0] == '[') return true;  // (an IPv6 address written out is a device's, not a site's)
    in_addr a{};
    if (inet_pton(AF_INET, h.c_str(), &a) == 1) {
        const uint32_t ip = ntohl(a.s_addr);
        const unsigned b0 = ip >> 24, b1 = (ip >> 16) & 255;
        return b0 == 0 || b0 == 10 || b0 == 127 || (b0 == 169 && b1 == 254) || (b0 == 172 && b1 >= 16 && b1 < 32) ||
               (b0 == 192 && b1 == 168) || (b0 == 100 && b1 >= 64 && b1 < 128);
    }
    if (h.find('.') == npos) return true;  // "localhost", a machine's bare name
    for (const char *end : {".localhost", ".local", ".lan", ".internal", ".home.arpa", ".ts.net"})
        if (ends_with(h, end)) return true;
    return false;
}

// A path without its "." and ".." parts.
std::string plain_path(const std::string &path) {
    std::vector<std::string> parts;
    size_t i = 0;
    const bool dir = !path.empty() && path.back() == '/';
    while (i <= path.size()) {
        const size_t e = std::min(path.find('/', i), path.size());
        const std::string p = path.substr(i, e - i);
        if (p == "..") {
            if (!parts.empty()) parts.pop_back();
        } else if (!p.empty() && p != ".") {
            parts.push_back(p);
        }
        i = e + 1;
    }
    std::string out;
    for (auto &p : parts) out += "/" + p;
    if (out.empty() || dir || ends_with(path, "/.") || ends_with(path, "/..")) out += "/";
    return out;
}

// The address a page at `base` means by `ref` (a link's href, a form's
// action), without its fragment; "" when that is no http(s) address.
std::string resolve_url(const std::string &base, std::string ref) {
    while (!ref.empty() && isspace((unsigned char)ref.front())) ref.erase(0, 1);
    while (!ref.empty() && isspace((unsigned char)ref.back())) ref.pop_back();
    const size_t hash = ref.find('#');
    if (hash != npos) ref.resize(hash);
    for (size_t i = 0; i < ref.size(); i++)
        if (ref[i] == ' ') ref.replace(i, 1, "%20");
        else if (ref[i] == '\n' || ref[i] == '\r' || ref[i] == '\t') ref.erase(i--, 1);
    const size_t se = base.find("://");
    if (se == npos) return "";
    const std::string scheme = base.substr(0, se);
    const size_t he = base.find_first_of("/?#", se + 3);
    const std::string origin = base.substr(0, he);
    std::string rest = he == npos ? "/" : base.substr(he);
    const size_t fh = rest.find('#');
    if (fh != npos) rest.resize(fh);
    if (rest.empty() || rest[0] != '/') rest = "/" + rest;
    const size_t q = rest.find('?');
    const std::string path = rest.substr(0, q), query = q == npos ? "" : rest.substr(q);
    if (ref.empty()) return origin + path + query;
    size_t colon = 0;
    while (colon < ref.size() && (isalnum((unsigned char)ref[colon]) || ref[colon] == '+' || ref[colon] == '-' || ref[colon] == '.'))
        colon++;
    if (colon > 0 && colon < ref.size() && ref[colon] == ':' && isalpha((unsigned char)ref[0])) {
        const std::string s = lower(ref.substr(0, colon));
        if ((s != "http" && s != "https") || ref.compare(colon, 3, "://") != 0) return "";
        const size_t h = ref.find_first_of("/?", colon + 3);
        return h == npos ? ref + "/" : ref;
    }
    auto with_query = [](const std::string &p) {
        const size_t k = p.find('?');
        return plain_path(p.substr(0, k)) + (k == npos ? "" : p.substr(k));
    };
    if (ref.compare(0, 2, "//") == 0) return scheme + ":" + ref;
    if (ref[0] == '/') return origin + with_query(ref);
    if (ref[0] == '?') return origin + path + ref;
    return origin + with_query(path.substr(0, path.rfind('/') + 1) + ref);
}

// A form's field as it goes into an address.
std::string form_encode(const std::string &s) {
    std::string out;
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '*') out += (char)c;
        else if (c == ' ') out += '+';
        else {
            char b[4];
            snprintf(b, sizeof b, "%%%02X", c);
            out += b;
        }
    }
    return out;
}

std::string base64(const unsigned char *d, size_t n) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < n; i += 3) {
        const unsigned v = (d[i] << 16) | (i + 1 < n ? d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += i + 1 < n ? t[(v >> 6) & 63] : '=';
        out += i + 2 < n ? t[v & 63] : '=';
    }
    return out;
}

// ---- The Chrome DevTools Protocol, over a WebSocket on this machine

class WebSocket {
public:
    bool open(const std::string &host, int port, const std::string &path, std::string &err) {
        if (!s_.open(host, port, false, 10, err)) return false;
        unsigned char key[16];
        if (!random_bytes(key, sizeof key))
            for (auto &k : key) k = (unsigned char)rand();
        const std::string req = "GET " + path + " HTTP/1.1\r\nHost: " + host + ":" + std::to_string(port) +
                                "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: " +
                                base64(key, sizeof key) + "\r\nSec-WebSocket-Version: 13\r\n\r\n";
        if (!s_.write(req.data(), req.size(), err)) return false;
        std::string head;
        const double until = now_s() + 10;
        size_t he;
        while ((he = head.find("\r\n\r\n")) == npos) {
            char buf[4096];
            const int left = (int)((until - now_s()) * 1000);
            const int n = left > 0 ? s_.read(buf, sizeof buf, left, err) : -2;
            if (n <= 0) {
                if (err.empty()) err = n == -2 ? "no answer to the WebSocket request" : "the connection was closed";
                return false;
            }
            head.append(buf, (size_t)n);
        }
        if (head.compare(0, 12, "HTTP/1.1 101") != 0) {
            err = "the WebSocket request was refused: " + head.substr(0, head.find("\r\n"));
            return false;
        }
        in_ = head.substr(he + 4);
        return true;
    }
    bool send(const std::string &text, std::string &err, unsigned char op = 0x1) {
        std::string f;
        f += (char)(0x80 | op);
        const size_t n = text.size();
        if (n < 126) f += (char)(0x80 | n);
        else if (n < 65536) {
            f += (char)(0x80 | 126);
            f += (char)(n >> 8);
            f += (char)(n & 255);
        } else {
            f += (char)(0x80 | 127);
            for (int k = 7; k >= 0; k--) f += (char)(((uint64_t)n >> (8 * k)) & 255);
        }
        unsigned char mask[4];
        if (!random_bytes(mask, 4))
            for (auto &m : mask) m = (unsigned char)rand();
        f.append((const char *)mask, 4);
        const size_t at = f.size();
        f += text;
        for (size_t i = 0; i < n; i++) f[at + i] = (char)(f[at + i] ^ mask[i & 3]);
        return s_.write(f.data(), f.size(), err);
    }
    // One whole message: 1 it is in msg, 0 none came in wait_ms, -1 the
    // connection is gone (err says).
    int recv(std::string &msg, int wait_ms, std::string &err) {
        const double until = now_s() + wait_ms / 1000.0;
        for (;;) {
            bool fin;
            unsigned op;
            std::string payload;
            while (frame(fin, op, payload)) {
                if (op == 0x9) {
                    std::string e;
                    send(payload, e, 0xA);
                    continue;
                }
                if (op == 0xA) continue;
                if (op == 0x8) {
                    err = "the browser closed the connection";
                    return -1;
                }
                part_ += payload;
                if (fin) {
                    msg.swap(part_);
                    part_.clear();
                    return 1;
                }
            }
            const int left = (int)((until - now_s()) * 1000);
            if (left <= 0) return 0;
            static char buf[1 << 16];
            const int n = s_.read(buf, sizeof buf, left, err);
            if (n == -2) return 0;
            if (n <= 0) {
                if (err.empty()) err = "the browser closed the connection";
                return -1;
            }
            in_.append(buf, (size_t)n);
        }
    }
    void close() { s_.close(); }

private:
    // The next frame of in_, taken off it; false until it is all there.
    bool frame(bool &fin, unsigned &op, std::string &payload) {
        const size_t have = in_.size() - at_;
        if (have < 2) return compact(), false;
        const unsigned char b0 = in_[at_], b1 = in_[at_ + 1];
        size_t off = 2;
        uint64_t len = b1 & 0x7F;
        if (len == 126) {
            if (have < 4) return false;
            len = ((unsigned char)in_[at_ + 2] << 8) | (unsigned char)in_[at_ + 3];
            off = 4;
        } else if (len == 127) {
            if (have < 10) return false;
            len = 0;
            for (int k = 0; k < 8; k++) len = (len << 8) | (unsigned char)in_[at_ + 2 + k];
            off = 10;
        }
        const bool masked = b1 & 0x80;
        if (have < off + (masked ? 4 : 0) + len) return false;
        payload.assign(in_, at_ + off + (masked ? 4 : 0), (size_t)len);
        if (masked)
            for (size_t i = 0; i < payload.size(); i++) payload[i] = (char)(payload[i] ^ in_[at_ + off + (i & 3)]);
        at_ += off + (masked ? 4 : 0) + (size_t)len;
        fin = b0 & 0x80;
        op = b0 & 0x0F;
        return true;
    }
    void compact() {
        if (at_ > 0) in_.erase(0, at_), at_ = 0;
    }
    Stream s_;
    std::string in_, part_;
    size_t at_ = 0;  // in_ is read from here: frames are taken off without moving the rest each time
};

// TS5_BROWSER_TRACE=FILE: what goes to the browser and what comes back, a
// line each with the time, for a browser that hangs or answers oddly (on
// the PS5 Obscura stopped answering after the first click).
void trace(const char *way, const std::string &what) {
    static FILE *f = [] {
        const char *path = getenv("TS5_BROWSER_TRACE");
        return path && *path ? fopen(path, "a") : nullptr;
    }();
    if (!f) return;
    static const double t0 = now_s();
    fprintf(f, "%9.3f %s %s\n", now_s() - t0, way, brief(what, 400).c_str());
    fflush(f);
}

class Cdp {
public:
    // Connects to the browser listening on 127.0.0.1:port and takes its
    // page (the one it opened with, or a new one).
    bool attach(int port, std::string &err) {
        HttpResponse resp;
        if (!http_get("http://127.0.0.1:" + std::to_string(port) + "/json/version", {}, resp, err, 1u << 20, 5)) return false;
        Json v;
        const std::string ws = Json::parse(resp.body, v) ? v["webSocketDebuggerUrl"].str() : "";
        const size_t s = ws.find("://");
        const size_t p = s == npos ? npos : ws.find('/', s + 3);
        if (p == npos) {
            err = "the browser's debugging port gave no address to connect to";
            return false;
        }
        if (!ws_.open("127.0.0.1", port, ws.substr(p), err)) return false;
        Json r;
        std::string target;
        if (call("Target.getTargets", "{}", r, err, 10000, false))
            for (auto &t : r["targetInfos"].a)
                if (t["type"].str() == "page" && target.empty()) target = t["targetId"].str();
        if (target.empty()) {
            if (!call("Target.createTarget", "{\"url\":\"about:blank\"}", r, err, 15000, false)) return false;
            target = r["targetId"].str();
        }
        if (!call("Target.attachToTarget", "{\"targetId\":" + json_escape(target) + ",\"flatten\":true}", r, err, 15000, false))
            return false;
        session_ = r["sessionId"].str();
        if (session_.empty()) {
            err = "the browser gave no session for its page";
            return false;
        }
        std::string e;
        call("Page.enable", "{}", r, e, 10000);  // for a page's alert(): it is answered, or it would hold everything
        plain_agent(v["User-Agent"].str(), port);
        return true;
    }
    // A Chromium without a window calls itself "HeadlessChrome" where one
    // with a window says "Chrome", and some sites turn that word away
    // unread. This one browses for the person at the keyboard, a page at a
    // time, so it says what their own Chrome would: the same engine,
    // version and system, in its User-Agent header and in what it tells a
    // page's scripts (and its Sec-CH-UA headers), which are its own less
    // that one name. Nothing else about it is dressed up, and
    // TS5_BROWSER_AGENT=own leaves it its own words.
    void plain_agent(const std::string &own, int port) {
        const char *as = getenv("TS5_BROWSER_AGENT");
        const size_t at = own.find("HeadlessChrome/");
        if ((as && std::string(as) == "own") || at == npos) return;
        std::string agent = own;
        agent.erase(at, 8);  // "Headless"
        // The names are asked of the browser itself, on a page of its own
        // in a tab opened for the question (it gives them only to a page
        // that came safely, as one from this machine has).
        static const char *const kData = R"JS((async () => {
  const d = navigator.userAgentData;
  if (!d || location.protocol !== 'http:' || document.readyState !== 'complete') return '';
  const h = await d.getHighEntropyValues(['architecture', 'bitness', 'model', 'platformVersion', 'fullVersionList', 'wow64']);
  const keep = l => (l || []).filter(b => !/headless/i.test(b.brand));
  return JSON.stringify({brands: keep(d.brands), fullVersionList: keep(h.fullVersionList), platform: d.platform || '',
    platformVersion: h.platformVersion || '', architecture: h.architecture || '', model: h.model || '',
    mobile: !!d.mobile, bitness: h.bitness || '', wow64: !!h.wow64});
})())JS";
        std::string data, e;
        Json r;
        if (call("Target.createTarget", "{\"url\":\"http://127.0.0.1:" + std::to_string(port) + "/json/version\"}", r, e, 15000,
                 false)) {
            const std::string tab = r["targetId"].str(), mine = session_;
            if (call("Target.attachToTarget", "{\"targetId\":" + json_escape(tab) + ",\"flatten\":true}", r, e, 15000, false)) {
                session_ = r["sessionId"].str();
                for (int i = 0; i < 30 && data.empty() && !gone_; i++) {
                    if (i) sleep_ms(100);
                    eval(kData, data, e, 5000);
                }
                session_ = mine;
            }
            call("Target.closeTarget", "{\"targetId\":" + json_escape(tab) + "}", r, e, 5000, false);
        }
        std::string params = "{\"userAgent\":" + json_escape(agent);
        if (data.size() > 2 && data[0] == '{') params += ",\"userAgentMetadata\":" + data;
        call("Emulation.setUserAgentOverride", params + "}", r, e, 10000);
    }
    // One command and its result. page: for the page, not the browser.
    bool call(const std::string &method, const std::string &params, Json &result, std::string &err,
              int timeout_ms = 30000, bool page = true) {
        const int id = next_++;
        const std::string session = page && !session_.empty() ? ",\"sessionId\":" + json_escape(session_) : "";
        trace(">", std::to_string(id) + " " + method + " " + params);
        if (!ws_.send("{\"id\":" + std::to_string(id) + ",\"method\":" + json_escape(method) + ",\"params\":" + params +
                          session + "}",
                      err)) {
            trace("!", "could not send: " + err);
            gone_ = true;
            return false;
        }
        const double until = now_s() + timeout_ms / 1000.0;
        for (;;) {
            const int left = (int)((until - now_s()) * 1000);
            std::string msg;
            const int got = left > 0 ? ws_.recv(msg, left, err) : 0;
            if (got < 0) {
                trace("!", "the connection ended: " + err);
                gone_ = true;
                return false;
            }
            if (got == 0) {
                err = "the browser didn't answer in " + std::to_string(timeout_ms / 1000) + " s";
                trace("!", std::to_string(id) + " " + method + ": no answer in " + std::to_string(timeout_ms) + " ms");
                return false;
            }
            trace("<", msg);
            Json j;
            if (!Json::parse(msg, j)) continue;
            if (j["id"].num(-1) == id) {
                if (j.o.count("error")) {
                    err = j["error"]["message"].str("the browser refused " + method);
                    return false;
                }
                result = j["result"];
                return true;
            }
            const std::string event = j["method"].str();
            if (event == "Page.javascriptDialogOpening") {
                std::string e;
                ws_.send("{\"id\":" + std::to_string(next_++) +
                             ",\"method\":\"Page.handleJavaScriptDialog\",\"params\":{\"accept\":true}" + session + "}",
                         e);
            } else if (event == "Inspector.targetCrashed" || event == "Target.targetCrashed") {
                err = "the page crashed the browser's tab";
                gone_ = true;
                return false;
            }
        }
    }
    // A script run in the page; its result has to be a string.
    bool eval(const std::string &js, std::string &value, std::string &err, int timeout_ms = 30000) {
        Json r;
        if (!call("Runtime.evaluate",
                  "{\"expression\":" + json_escape(js) + ",\"returnByValue\":true,\"awaitPromise\":true,\"userGesture\":true}",
                  r, err, timeout_ms))
            return false;
        if (!r["exceptionDetails"].is_null()) {
            err = r["exceptionDetails"]["exception"]["description"].str(r["exceptionDetails"]["text"].str("the script failed"));
            return false;
        }
        value = r["result"]["value"].str();
        return true;
    }
    bool gone() const { return gone_; }
    void close() {
        Json r;
        std::string e;
        if (!gone_) call("Browser.close", "{}", r, e, 2000, false);
        ws_.close();
    }

private:
    WebSocket ws_;
    int next_ = 1;
    std::string session_;
    bool gone_ = false;
};

// Run in the page before it is read: every element that takes a click or
// text gets a mark (data-ts5) the reader lists it under and a click finds
// it by, what the page's styles hide is flagged for the reader to leave
// out, a box's text and a tick as they are now are written where the
// reader looks, and a link or a script that would open another window opens
// here instead (there is one page).
const char *const kMark = R"JS((() => {
  const doc = document;
  for (const el of doc.querySelectorAll('[data-ts5]')) el.removeAttribute('data-ts5');
  for (const el of doc.querySelectorAll('[data-ts5-hidden]')) el.removeAttribute('data-ts5-hidden');
  if (doc.body) {
    // What the page's styles hide. Chromium answers getComputedStyle at
    // once; a browser that works an element's styles out anew on every
    // asking does not (Obscura on the PS5: 15 s for one Wikipedia article,
    // and no answer at all from the next, with every later request stuck
    // behind this script). So the asking gets STYLE_MS milliseconds, and
    // the rest of the page goes by its attributes alone, which the reader
    // knows itself (hidden, aria-hidden, an inline display:none).
    const t0 = Date.now();
    let n = 0, late = STYLE_MS <= 0;
    const w = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT, { acceptNode(el) {
      if (late) return NodeFilter.FILTER_ACCEPT;
      if ((++n & 31) === 0 && Date.now() - t0 > STYLE_MS) {
        late = true;
        return NodeFilter.FILTER_ACCEPT;
      }
      const st = getComputedStyle(el);
      if (st.display === 'none' || st.visibility === 'hidden') {
        el.setAttribute('data-ts5-hidden', '');
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    } });
    while (!late && w.nextNode());
  }
  let k = 0;
  const sel = 'a[href],button,summary,input,textarea,[role=button],[role=link],[role=tab],[role=menuitem],[onclick]';
  for (const el of doc.querySelectorAll(sel)) {
    el.setAttribute('data-ts5', String(++k));
    if (el.tagName === 'A') el.removeAttribute('target');
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      if (el.type !== 'password' && el.type !== 'file') el.setAttribute('data-ts5-value', el.value);
      if (el.checked) el.setAttribute('checked', ''); else el.removeAttribute('checked');
    }
  }
  for (const f of doc.querySelectorAll('form[target]')) f.removeAttribute('target');
  window.open = (u) => { if (u) location.href = u; return window; };
  const html = doc.documentElement ? doc.documentElement.outerHTML : '';
  return JSON.stringify({ url: location.href, title: doc.title, html: html });
})())JS";

// kMark with its time for the styles filled in: a second and a half, or
// TS5_BROWSER_STYLE_MS (0: the styles are never asked for).
std::string mark_js() {
    std::string js = kMark;
    const char *v = getenv("TS5_BROWSER_STYLE_MS");
    const std::string ms = v && *v && strspn(v, "0123456789") == strlen(v) ? v : "1500";
    for (size_t at = js.find("STYLE_MS"); at != npos; at = js.find("STYLE_MS", at)) js.replace(at, 8, ms);
    return js;
}

// ---- The browser program

struct Program {
    pid_t pid = -1;
    int port = 0;
    std::string dir;  // its profile folder and its log
};

int free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof a;
    int port = 0;
    if (bind(fd, (sockaddr *)&a, sizeof a) == 0 && getsockname(fd, (sockaddr *)&a, &len) == 0) port = ntohs(a.sin_port);
    close(fd);
    return port;
}

bool make_dirs(const std::string &path) {
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') {
            const std::string p = path.substr(0, i);
            if (mkdir(p.c_str(), 0700) != 0 && errno != EEXIST) return false;
        }
    return true;
}

void remove_tree(const std::string &dir) {
    if (dir.find("/browser-") == npos) return;  // only a folder this program made
    nftw(dir.c_str(), [](const char *p, const struct stat *, int, struct FTW *) { return remove(p); }, 16,
         FTW_DEPTH | FTW_PHYS);
}

std::string last_line(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return "";
    std::string all;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0 && all.size() < (1u << 20)) all.append(buf, n);
    fclose(f);
    while (!all.empty() && isspace((unsigned char)all.back())) all.pop_back();
    const size_t nl = all.rfind('\n');
    return brief(clean_text(nl == npos ? all : all.substr(nl + 1)), 200);
}

void stop(Program &b) {
    if (b.pid > 0) {
        kill(-b.pid, SIGTERM);
        const double until = now_s() + 2;
        int st;
        while (waitpid(b.pid, &st, WNOHANG) == 0 && now_s() < until) sleep_ms(50);
        kill(-b.pid, SIGKILL);
        waitpid(b.pid, &st, 0);
        b.pid = -1;
    }
    if (!b.dir.empty()) remove_tree(b.dir);
    b.dir.clear();
}

// How long a browser has to open its debugging port: 30 s, or
// TS5_BROWSER_START_S.
static int start_s() {
    const char *v = getenv("TS5_BROWSER_START_S");
    const int n = v && *v ? atoi(v) : 30;
    return n < 1 ? 30 : std::min(n, 600);
}

// Starts the browser without a window, its debugging port on 127.0.0.1,
// and waits until the port answers. slow: it ran, and the port never came.
bool launch(const std::string &program, bool obscura, Program &b, std::string &err, bool &slow) {
    slow = false;
    const char *cache = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
    // Under the user's cache, not /tmp: that is memory on many systems.
    std::string dir = (cache && *cache ? std::string(cache) : std::string(home && *home ? home : "/tmp") + "/.cache") +
                      "/tokenstation";
    if (!make_dirs(dir)) dir = "/tmp";
    dir += "/browser-XXXXXX";
    if (!mkdtemp(&dir[0])) {
        err = std::string("no folder for the browser's profile: ") + strerror(errno);
        return false;
    }
    b.dir = dir;
    b.port = free_port();
    std::vector<std::string> args;
    if (obscura) {
        args = {program, "--storage-dir", dir, "serve", "--port", std::to_string(b.port)};
    } else {
        args = {program,
                "--headless",
                "--remote-debugging-port=" + std::to_string(b.port),
                "--user-data-dir=" + dir,
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu",  // the GPU's memory is the model's
                "--disable-extensions",
                "--disable-sync",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-default-apps",
                // Else it asks the desktop's keyring for a key to its cookies,
                // and where no keyring answers (a machine reached over SSH)
                // no page ever loads.
                "--password-store=basic",
                "--mute-audio",
                "--hide-scrollbars",
                "--window-size=1280,1600",
                "--disk-cache-size=33554432"};
        // Chromium refuses to run as root with its sandbox.
        const char *no = getenv("TS5_BROWSER_NO_SANDBOX");
        if (geteuid() == 0 || (no && *no == '1')) args.push_back("--no-sandbox");
    }
    if (const char *more = getenv("TS5_BROWSER_FLAGS")) {
        std::string f = more;
        for (size_t i = 0; i < f.size();) {
            const size_t e = std::min(f.find(' ', i), f.size());
            if (e > i) args.push_back(f.substr(i, e - i));
            i = e + 1;
        }
    }
    if (!obscura) args.push_back("about:blank");
    const std::string log = dir + "/browser.log";
    const pid_t pid = fork();
    if (pid < 0) {
        err = std::string("can't start a program: ") + strerror(errno);
        remove_tree(dir);
        return false;
    }
    if (pid == 0) {
        setpgid(0, 0);
#ifdef __linux__
        prctl(PR_SET_PDEATHSIG, SIGKILL);  // it never outlives this program
#endif
        const int in = ::open("/dev/null", O_RDONLY), out = ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (in >= 0) dup2(in, 0);
        if (out >= 0) dup2(out, 1), dup2(out, 2);
        std::vector<char *> argv;
        for (auto &a : args) argv.push_back(&a[0]);
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        fprintf(stderr, "%s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    setpgid(pid, pid);
    b.pid = pid;
    const int limit = start_s();
    const double until = now_s() + limit;
    while (now_s() < until) {
        int st;
        if (waitpid(pid, &st, WNOHANG) == pid) {
            b.pid = -1;
            const std::string said = last_line(log);
            err = "the browser program ended as it started" + (said.empty() ? "" : " (" + said + ")");
            stop(b);
            return false;
        }
        HttpResponse resp;
        std::string e;
        if (http_get("http://127.0.0.1:" + std::to_string(b.port) + "/json/version", {}, resp, e, 1u << 20, 2) &&
            resp.status == 200)
            return true;
        sleep_ms(150);
    }
    err = "the browser program didn't open its debugging port in " + std::to_string(limit) + " s";
    slow = true;
    stop(b);
    return false;
}

// Where a program named `name` is: as given with a '/', else on the PATH
// or in ~/.local/bin; "" when it isn't there.
// (A folder has the "x" right too: only a file is a program.)
bool runnable(const std::string &p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && access(p.c_str(), X_OK) == 0;
}

std::string which(const std::string &name) {
    if (name.find('/') != npos) return runnable(name) ? name : "";
    std::string path = getenv("PATH") ? getenv("PATH") : "/usr/local/bin:/usr/bin:/bin";
    if (const char *home = getenv("HOME")) path += std::string(":") + home + "/.local/bin";
    for (size_t i = 0; i <= path.size();) {
        const size_t e = std::min(path.find(':', i), path.size());
        const std::string p = path.substr(i, e - i) + "/" + name;
        if (e > i && runnable(p)) return p;
        i = e + 1;
    }
    return "";
}

bool is_obscura(const std::string &program) {
    const size_t slash = program.rfind('/');
    return lower(program.substr(slash == npos ? 0 : slash + 1)).find("obscura") != npos;
}

// The browser that is installed, for Auto: a Chromium (it shows more sites
// as they are, gives memory back between pages and runs each page in a
// sandbox). Obscura (the light one) only when it is asked for by kind: on
// the PS5 it took 12 s for a page Chromium showed in 5, and stopped
// answering after a click (2026-11-11, twice), so Auto goes from Chromium
// to the text view.
std::vector<std::string> find_browsers(const std::string &kind) {
    std::vector<std::string> out;
    if (kind != "obscura")
        for (const char *name : {"chromium", "chromium-browser", "chromium-headless-shell", "chrome-headless-shell",
                                 "headless_shell", "google-chrome", "google-chrome-stable", "chrome", "brave-browser",
                                 "microsoft-edge"}) {
            const std::string p = which(name);
            if (!p.empty()) {
                out.push_back(p);
                break;
            }
        }
    if (kind == "obscura") {
        const std::string p = which("obscura");
        if (!p.empty()) out.push_back(p);
    }
    return out;
}

// ---- The page that is open, and what is done on it

struct Answer {
    bool ok = true;
    std::string text;
};

class Session {
public:
    // spares: the other browsers installed, tried in turn when this one
    // doesn't start.
    Session(const std::string &program, bool obscura, size_t chars, std::vector<std::string> spares = {})
        : program_(program), obscura_(obscura), chars_(std::max<size_t>(chars, 500)), spares_(std::move(spares)) {}
    ~Session() { quit(); }

    std::string mode() const { return program_.empty() ? "text" : obscura_ ? "obscura" : "chromium"; }
    bool running() const { return proc_.pid > 0; }
    const std::string &url() const { return url_; }
    const std::string &title() const { return title_; }

    Answer open(const std::string &address) {
        std::string url = address;
        while (!url.empty() && isspace((unsigned char)url.front())) url.erase(0, 1);
        while (!url.empty() && isspace((unsigned char)url.back())) url.pop_back();
        if (url.empty()) return fail("browser_open needs an address, such as {\"url\": \"https://example.com\"}.");
        if (url.find("://") == npos && url.find(' ') == npos && url.find('.') != npos) url = "https://" + url;
        const std::string full = resolve_url("http://x/", url);
        if (full.empty() || url.find("://") == npos)
            return fail("Only web pages can be opened: an address that starts with https:// or http://.");
        return go(full, true);
    }

    Answer click(int n) {
        if (url_.empty()) return fail("No page is open yet. Open one with browser_open first.");
        if (n < 1 || (size_t)n > els_.items.size()) return no_such(n);
        const size_t i = (size_t)n - 1;
        const PageElement &el = els_.items[i];
        if (el.kind == "input")
            return fail("[" + std::to_string(n) + "] is a text box. Write in it with browser_type.");
        if (el.kind == "link" && !el.href.empty() && el.href[0] != '#' && lower(el.href).compare(0, 11, "javascript:") != 0 &&
            resolve_url(url_, el.href).empty())
            return fail("That link doesn't lead to a web page (" + brief(el.href, 60) + ").");
        // Obscura stops answering once a page has moved on by itself (on
        // the PS5 every command after a click on a link timed out, while
        // its events still came): there a link is opened by its address.
        if (obscura_ && !program_.empty() && el.kind == "link" && !el.href.empty() && el.href[0] != '#' &&
            lower(el.href).compare(0, 11, "javascript:") != 0)
            return go(resolve_url(url_, el.href), true);
        if (!program_.empty()) return act("(() => { const el = document.querySelector('[data-ts5=\"" + el.mark +
                                              "\"]'); if (!el) return 'gone'; el.scrollIntoView({block: 'center'}); "
                                              "el.click(); return 'ok'; })()",
                                          true, "");
        if (el.kind == "link") return go(resolve_url(url_, el.href), true);
        if (el.kind == "check") {
            if (el.type == "radio") {
                for (size_t k = 0; k < els_.items.size(); k++)
                    if (els_.items[k].type == "radio" && els_.items[k].form == el.form && els_.items[k].name == el.name)
                        ticked_[k] = false;
                ticked_[i] = true;
            } else {
                ticked_[i] = !ticked_[i];
            }
            Answer a;
            a.text = "Box [" + std::to_string(n) + "] is now " + (ticked_[i] ? "ticked" : "empty") +
                     ". Nothing is sent until the form's button is pressed.";
            return a;
        }
        if (el.form < 0) return fail("This button does nothing in the text view: it needs a full browser.");
        return submit(el.form, (int)i);
    }

    Answer type(int n, const std::string &text, bool enter) {
        if (url_.empty()) return fail("No page is open yet. Open one with browser_open first.");
        if (n < 1 || (size_t)n > els_.items.size()) return no_such(n);
        const size_t i = (size_t)n - 1;
        const PageElement &el = els_.items[i];
        if (el.kind != "input")
            return fail("[" + std::to_string(n) + "] is not a text box (it is a " +
                        (el.kind == "check" ? "tick box" : el.kind) + "). Text boxes look like [4: text box \"Search\"].");
        const std::string done = "Typed into [" + std::to_string(n) + "]. Nothing is sent yet: press the form's button "
                                 "with browser_click, or call browser_type with \"enter\": true.";
        if (!program_.empty()) {
            // The value is set the way a page's own scripts (React and the
            // like) notice, then Enter as a key: a page's handler sees it,
            // and where none takes it the form is sent as a browser would.
            std::string js = "(() => { const el = document.querySelector('[data-ts5=\"" + el.mark +
                             "\"]'); if (!el) return 'gone'; el.scrollIntoView({block: 'center'}); el.focus(); "
                             "const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : "
                             "HTMLInputElement.prototype; const set = Object.getOwnPropertyDescriptor(proto, 'value'); "
                             "if (set && set.set) set.set.call(el, " + json_escape(text) + "); else el.value = " +
                             json_escape(text) + "; el.dispatchEvent(new Event('input', {bubbles: true})); "
                             "el.dispatchEvent(new Event('change', {bubbles: true})); ";
            if (enter)
                js += "const key = {key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true}; "
                      "const went = el.dispatchEvent(new KeyboardEvent('keydown', key)); "
                      "el.dispatchEvent(new KeyboardEvent('keypress', key)); "
                      "if (went && el.form && el.tagName !== 'TEXTAREA') { if (el.form.requestSubmit) el.form.requestSubmit(); "
                      "else el.form.submit(); } el.dispatchEvent(new KeyboardEvent('keyup', key)); ";
            js += "return 'ok'; })()";
            return act(js, enter, done);
        }
        typed_[i] = text;
        if (!enter) {
            Answer a;
            a.text = done;
            return a;
        }
        if (el.form < 0) return fail("This box sends nothing in the text view: it needs a full browser.");
        return submit(el.form, -1);
    }

    Answer more() {
        if (url_.empty()) return fail("No page is open yet. Open one with browser_open first.");
        if (part_ + 1 >= cuts_.size()) {
            Answer a;
            a.text = "That was the end of the page.";
            return a;
        }
        part_++;
        return show();
    }

    Answer back() {
        // (Obscura goes back by ts5's own list, each page opened by its
        // address: its own history is a move of the page's, see click.)
        if (!program_.empty() && !obscura_) {
            if (url_.empty()) return fail("There is no page to go back to.");
            std::string why;
            if (!awake(why)) return fail(why);
            // By the browser's own list of the pages it has been to (the
            // first is the empty page it started with).
            Json r;
            std::string err;
            if (cdp_->call("Page.getNavigationHistory", "{}", r, err, 10000)) {
                const int at = (int)r["currentIndex"].num(-1);
                const std::vector<Json> &pages = r["entries"].a;
                if (at < 1 || (size_t)at >= pages.size() || pages[(size_t)at - 1]["url"].str().compare(0, 6, "about:") == 0)
                    return fail("There is no page to go back to.");
                if (!cdp_->call("Page.navigateToHistoryEntry",
                                "{\"entryId\":" + std::to_string((int)pages[(size_t)at - 1]["id"].num(0)) + "}", r, err, 30000))
                    return fail("Going back didn't work: " + lost(err) + ".");
                sleep_ms(300);
                settle(15);
                return snapshot();
            }
            return act("(() => { history.back(); return 'ok'; })()", true, "");
        }
        if (back_.empty()) return fail("There is no page to go back to.");
        const std::string url = back_.back();
        back_.pop_back();
        const Answer a = go(url, false);
        if (!a.ok) back_.push_back(url);
        return a;
    }

    // What a request would do, in words for the user, and whether to ask.
    void what(const Json &rq, std::string &line, bool &asks) {
        const std::string act = rq["act"].str();
        const std::string here = host_of(url_);
        const int n = (int)rq["n"].num(0);
        const PageElement *el = n >= 1 && (size_t)n <= els_.items.size() ? &els_.items[(size_t)n - 1] : nullptr;
        asks = false;
        if (act == "open") {
            std::string url = rq["url"].str();
            if (url.find("://") == npos) url = "https://" + url;
            asks = local_host(host_of(url));
            line = "open " + brief(url, 200) + (asks ? " (on this machine or your own network)" : "");
        } else if (act == "click" && el && el->kind == "input") {
            line = "click a text box";  // refused (click): nothing to ask
        } else if (act == "type" && el && el->kind != "input") {
            line = "type into what is not a text box";  // refused (type)
        } else if (act == "click" && el) {
            const std::string name = el->label.empty() ? "" : " \"" + el->label + "\"";
            if (el->kind == "link") {
                const std::string to = resolve_url(url_, el->href);
                asks = !to.empty() && local_host(host_of(to)) && !local_host(here);
                line = "follow the link" + name + (to.empty() ? "" : " to " + brief(to, 200)) +
                       (asks ? " (on this machine or your own network)" : "");
            } else if (el->kind == "check") {
                line = "tick the box" + name + " on " + here;
            } else {
                asks = true;
                line = "press the button" + name + " on " + here;
            }
        } else if (act == "type" && el) {
            asks = true;
            line = "type \"" + brief(rq["text"].str(), 200) + "\" into" +
                   (el->label.empty() ? std::string(" a text box") : " \"" + el->label + "\"") + " on " + here +
                   (rq["enter"].boolean(true) ? " and press Enter" : "");
        } else {
            line = act;
        }
    }

    // The browser program is stopped; the next request starts it again and
    // opens the page that was open.
    void rest() {
        if (!running()) return;
        quit();
        asleep_ = !url_.empty();
    }

    void quit() {
        if (cdp_) {
            cdp_->close();
            delete cdp_;
            cdp_ = nullptr;
        }
        stop(proc_);
    }

private:
    Answer fail(const std::string &why) {
        Answer a;
        a.ok = false;
        a.text = why;
        return a;
    }
    Answer no_such(int n) {
        if (els_.items.empty()) return fail("This page has nothing numbered to act on.");
        return fail("There is no [" + std::to_string(n) + "] on this page: its numbers go from 1 to " +
                    std::to_string(els_.items.size()) + ".");
    }

    // The part of the page the model is at, under the page's name and address.
    Answer show() {
        Answer a;
        a.text = "Page: " + (title_.empty() ? std::string("(no title)") : title_) + "\nAddress: " + brief(url_, 300) + "\n";
        if (!note_.empty()) a.text += note_ + "\n";
        note_.clear();
        const size_t parts = cuts_.size();
        if (parts > 1) a.text += "Part " + std::to_string(part_ + 1) + " of " + std::to_string(parts) + ".\n";
        const size_t from = cuts_[part_], to = part_ + 1 < parts ? cuts_[part_ + 1] : view_.size();
        std::string body = view_.substr(from, to - from);
        while (!body.empty() && body.back() == '\n') body.pop_back();
        if (view_.empty())
            body = program_.empty() ? "(This page shows no text here: it is built by scripts, which the text view doesn't run.)"
                                    : "(This page shows no text.)";
        a.text += "\n" + body;
        if (part_ + 1 < parts) a.text += "\n\n[The page goes on. browser_more shows its next part.]";
        return a;
    }

    // html is the page at url: read, numbered and cut into parts.
    void set_page(const std::string &url, const std::string &html, const std::string *title) {
        url_ = url;
        els_ = PageElements();
        bool is_html = false;
        for (size_t i = 0; i < html.size() && i < 2048 && !is_html; i++)
            is_html = html[i] == '<' && (isalpha((unsigned char)html[i + 1 < html.size() ? i + 1 : i]) || html[i + 1] == '!');
        view_ = clean_text(is_html ? html_view(html, els_) : html);
        title_ = clean_text(title ? brief(*title, 200) : is_html ? html_title(html) : std::string());
        while (!view_.empty() && isspace((unsigned char)view_.back())) view_.pop_back();
        // (The reader puts the page's title on top; it is said above the page already.)
        if (!title_.empty() && view_.compare(0, title_.size(), title_) == 0 &&
            (view_.size() == title_.size() || view_[title_.size()] == '\n'))
            view_.erase(0, view_.find_first_not_of('\n', title_.size()));
        typed_.assign(els_.items.size(), "");
        ticked_.assign(els_.items.size(), false);
        for (size_t i = 0; i < els_.items.size(); i++) {
            if (els_.items[i].kind == "input") typed_[i] = els_.items[i].value;
            ticked_[i] = els_.items[i].checked;
        }
        // Parts of about chars_ bytes, each ending with a line.
        cuts_.assign(1, 0);
        while (view_.size() - cuts_.back() > chars_ + chars_ / 4) {
            const size_t from = cuts_.back();
            size_t cut = view_.rfind('\n', from + chars_);
            if (cut == npos || cut < from + chars_ / 2) {
                cut = from + chars_;
                while (cut > from && ((unsigned char)view_[cut] & 0xC0) == 0x80) cut--;
            } else {
                cut++;
            }
            cuts_.push_back(cut);
        }
        part_ = 0;
    }

    // Opens url. remember: the page it leaves can be gone back to.
    Answer go(const std::string &url, bool remember) {
        if (url.empty()) return fail("That doesn't lead to a web page.");
        if (!program_.empty()) {
            std::string why;
            if (!awake(why, false)) {
                if (!fell_back_) return fail(why);
            } else {
                Json r;
                std::string err;
                bool ok = cdp_->call("Page.navigate", "{\"url\":" + json_escape(url) + "}", r, err, 40000);
                if (!ok && obscura_ && !cdp_->gone() && err.find("didn't answer") != npos) {
                    // (An Obscura that is stuck, see act: once more, with a fresh one.)
                    quit();
                    std::string again;
                    if (!awake(again, false)) return fail(again);
                    ok = cdp_->call("Page.navigate", "{\"url\":" + json_escape(url) + "}", r, err, 40000);
                }
                if (!ok) return fail("The page could not be opened: " + lost(err) + ".");
                const std::string net = r["errorText"].str();
                if (!net.empty())
                    return fail("The page could not be opened: " +
                                std::string(net.find("NAME_NOT_RESOLVED") != npos ? "there is no such site"
                                            : net.find("CERT") != npos || net.find("SSL") != npos
                                                ? "the site's identity couldn't be checked"
                                                : net.find("ABORTED") != npos ? "it is a file to download, not a page"
                                                                              : "the site couldn't be reached") +
                                " (" + net + ").");
                const std::string left = url_;
                const char *silent = "The page could not be read: the browser didn't answer.";
                Answer a = settle(20) ? snapshot() : fail(silent);
                if (!a.ok && obscura_ && cdp_ && !cdp_->gone() && a.text.find("didn't answer") != npos) {
                    // Obscura that answers nothing once a page has loaded (on
                    // the PS5 the second article in a row, a link followed or
                    // an address opened alike, while a fresh one showed that
                    // page and ten more): a fresh one, and the page once more.
                    quit();
                    std::string again;
                    if (!awake(again, false)) return fail(again);
                    if (!cdp_->call("Page.navigate", "{\"url\":" + json_escape(url) + "}", r, err, 40000))
                        return fail("The page could not be opened: " + lost(err) + ".");
                    a = settle(20) ? snapshot() : fail(silent);
                }
                if (obscura_ && a.ok && remember && !left.empty() && left != url_) {
                    back_.push_back(left);
                    if (back_.size() > 50) back_.erase(back_.begin());
                }
                return a;
            }
        }
        HttpResponse resp;
        std::string err;
        if (!http_get(url, page_headers(), resp, err, 4u << 20, 20))
            return fail("The page could not be opened: " +
                        std::string(err.find("certificate") != npos || err.find("TLS") != npos
                                        ? "the site's identity couldn't be checked"
                                        : "the site couldn't be reached") +
                        ".");
        if (resp.status < 200 || resp.status >= 300)
            return fail("The page could not be opened: " +
                        (resp.status == 404   ? std::string("there is no such page (404)")
                         : resp.status == 403 ? std::string("the site refused (403)")
                         : resp.status == 429 ? std::string("the site says too many requests (429)")
                                              : "the site answered with error " + std::to_string(resp.status)) +
                        ".");
        if (resp.body.find('\0') != npos) return fail("That address is a file, an image or a PDF, not a page.");
        if (remember && !url_.empty()) {
            back_.push_back(url_);
            if (back_.size() > 50) back_.erase(back_.begin());
        }
        set_page(resp.url.empty() ? url : resp.url, resp.body, nullptr);
        return show();
    }

    // The text view's form f, sent: by its button `via` (an item), or by
    // Enter (-1), which a browser sends as the form's first button.
    Answer submit(int f, int via) {
        const PageForm &form = els_.forms[(size_t)f];
        if (form.method == "post")
            return fail("This form can only be sent by a full browser (it posts what is typed), and this is the text view.");
        std::string q;
        auto add = [&](const std::string &name, const std::string &value) {
            q += (q.empty() ? "" : "&") + form_encode(name) + "=" + form_encode(value);
        };
        for (auto &fd : form.fields) {
            if (fd.item < 0) add(fd.name, fd.value);
            else if (els_.items[(size_t)fd.item].kind == "input") add(fd.name, typed_[(size_t)fd.item]);
            else if (ticked_[(size_t)fd.item]) add(fd.name, fd.value);
        }
        if (via < 0)
            for (size_t k = 0; k < els_.items.size() && via < 0; k++)
                if (els_.items[k].form == f && els_.items[k].kind == "button") via = (int)k;
        if (via >= 0 && !els_.items[(size_t)via].name.empty()) add(els_.items[(size_t)via].name, els_.items[(size_t)via].value);
        std::string to = resolve_url(url_, form.action);
        if (to.empty()) return fail("This form doesn't lead to a web page.");
        const size_t qm = to.find('?');
        if (qm != npos) to.resize(qm);
        return go(to + "?" + q, true);
    }

    // ---- With a browser program

    // The browser is running and connected, started if need be.
    // The browser, started if it isn't running. One that had stopped (it
    // rests when unused) comes back with the page it had, unless the caller
    // is about to open another (reopen false); woke_ then says so.
    bool awake(std::string &why, bool reopen = true) {
        woke_ = false;
        if (cdp_ && !cdp_->gone()) return true;
        const bool again = (cdp_ || asleep_) && reopen;
        quit();
        for (bool second = false;;) {
            std::string err;
            bool slow;
            cdp_ = new Cdp;
            if (launch(program_, obscura_, proc_, err, slow) && cdp_->attach(proc_.port, err)) break;
            quit();
            // A start that hung is tried once more: on a CI machine a Chrome
            // that had not opened its port in 30 s came up at once the
            // second time, and the text view would have been the whole
            // chat's.
            if (slow && !second) {
                second = true;
                continue;
            }
            second = false;
            why = "The browser (" + program_ + ") could not be started: " + err + ".";
            if (!spares_.empty()) {
                // Another browser is installed: that one, then.
                note_ += "(" + why + ")\n";
                program_ = spares_.front();
                spares_.erase(spares_.begin());
                obscura_ = is_obscura(program_);
                continue;
            }
            // Without a browser the text view still reads pages.
            note_ += "(" + why + " This is the simpler text view, which runs no scripts.)";
            fell_back_ = true;
            program_.clear();
            return false;
        }
        asleep_ = false;
        if (again && !url_.empty()) {
            Json r;
            std::string e;
            if (cdp_->call("Page.navigate", "{\"url\":" + json_escape(url_) + "}", r, e, 40000)) settle(20);
            woke_ = true;
        }
        return true;
    }

    std::string lost(const std::string &err) {
        return cdp_ && cdp_->gone() ? "the browser stopped (" + err + ")" : err;
    }

    // Waits for the page to be loaded and still: complete, and its size the
    // same on two looks, or three seconds past complete (a page that never
    // stops moving); max_s at most.
    // False when the browser gave no answer twice running (it is stuck).
    bool settle(double max_s) {
        const double t0 = now_s();
        double complete = 0;
        std::string last;
        int same = 0, silent = 0;
        while (now_s() - t0 < max_s) {
            std::string v, err;
            const bool ok = cdp_->eval("document.readyState + ' ' + (document.documentElement ? "
                                       "document.documentElement.outerHTML.length : 0)",
                                       v, err, 5000);
            if (cdp_->gone()) return true;
            silent = !ok && err.find("didn't answer") != npos ? silent + 1 : 0;
            if (silent >= 2) return false;
            if (ok && v.compare(0, 9, "complete ") == 0) {
                if (complete == 0) complete = now_s();
                same = v == last ? same + 1 : 0;
                if (same >= 2 || now_s() - complete > 3) return true;
            } else {
                complete = 0;
                same = 0;
            }
            last = ok ? v : "";
            sleep_ms(200);
        }
        return true;
    }

    // The page as the browser has it now.
    Answer snapshot() {
        std::string v, err;
        const std::string mark = mark_js();
        if (!cdp_->eval(mark, v, err, 30000)) {
            // (Between two pages there is nothing to run a script in. A
            // browser that didn't answer is still at the script: a second
            // one would only wait behind it.)
            if (err.find("didn't answer") != npos) return fail("The page could not be read: " + lost(err) + ".");
            sleep_ms(500);
            settle(10);
            if (!cdp_->eval(mark, v, err, 30000)) return fail("The page could not be read: " + lost(err) + ".");
        }
        Json j;
        if (!Json::parse(v, j)) return fail("The page could not be read.");
        const std::string url = j["url"].str();
        if (url.compare(0, 15, "chrome-error://") == 0) return fail("The page could not be opened: the site couldn't be reached.");
        const std::string title = j["title"].str();
        set_page(url, j["html"].str(), &title);
        return show();
    }

    // Runs a script that acts on the page ('ok', or 'gone' when what it
    // acts on is no longer there). moved: the page may change or be left,
    // so it is read again and shown; else `done` is the answer.
    Answer act(const std::string &js, bool moved, const std::string &done) {
        std::string why;
        const std::string before = view_;
        if (!awake(why)) return fail(why);
        if (woke_) {
            // The numbers are the page's as it was shown: they hold only if
            // the page opened again is that same page.
            const Answer a = snapshot();
            if (!a.ok) return a;
            if (view_ != before)
                return fail("The browser had stopped, so the page was opened again, and it is not as it was. "
                            "Nothing was done. Here it is now:\n\n" + a.text);
        }
        std::string v, err;
        if (!cdp_->eval(js, v, err, 30000)) {
            // A click that leaves the page takes the script's answer with it.
            const bool left = err.find("navigated") != npos || err.find("context") != npos || err.find("closed") != npos;
            if (!left || cdp_->gone()) return fail("That didn't work: " + lost(err) + ".");
            v = "ok";
            moved = true;
        }
        if (v == "gone") {
            const Answer a = snapshot();
            if (!a.ok) return a;
            return fail("The page has changed since it was shown, and that number is gone. Here it is now:\n\n" + a.text);
        }
        if (!moved) {
            Answer a;
            a.text = done;
            return a;
        }
        sleep_ms(300);
        const bool answers = settle(15);
        Answer a = answers ? snapshot() : fail("The page could not be read: the browser didn't answer.");
        if (!a.ok && obscura_ && cdp_ && !cdp_->gone() && a.text.find("didn't answer") != npos) {
            // Obscura after a button or Enter that moved the page on (see
            // click): it answers nothing more. Stopped, it comes back with
            // the next request, on the page it was on before.
            rest();
            a.text = "Obscura stopped answering after that, as it does when a button or Enter moves a page on. It was "
                     "stopped; the next request starts it again on the page before (" + brief(url_, 120) +
                     "). What was done may or may not have been sent. A Chromium does this properly.";
        }
        return a;
    }

    std::string program_;  // "" for the text view
    bool obscura_;
    size_t chars_;
    Program proc_;
    Cdp *cdp_ = nullptr;
    bool asleep_ = false, fell_back_ = false, woke_ = false;
    std::vector<std::string> spares_;

    std::string url_, title_, view_, note_;
    PageElements els_;
    std::vector<std::string> typed_;  // the text view: what is in each box
    std::vector<bool> ticked_;
    std::vector<size_t> cuts_ = {0};  // where each part of view_ starts
    size_t part_ = 0;
    std::vector<std::string> back_;  // the text view: the pages before this one
};

volatile sig_atomic_t g_quit = 0;

}  // namespace

int browser_main(int argc, char **argv) {
    std::string program = "auto", kind;
    size_t chars = 6000;
    double idle = 600;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--program" && i + 1 < argc) program = argv[++i];
        else if (a == "--kind" && i + 1 < argc) kind = argv[++i];
        else if (a == "--chars" && i + 1 < argc) chars = (size_t)atol(argv[++i]);
        else if (a == "--idle" && i + 1 < argc) idle = atof(argv[++i]);
        else {
            fprintf(stderr, "usage: ts5 browser [--program auto|text|PATH] [--kind chromium|obscura] [--chars N] [--idle SECONDS]\n");
            return 2;
        }
    }
    std::string note;
    std::vector<std::string> spares;
    if (program == "text") program.clear();
    else if (program == "auto") {
        spares = find_browsers(kind);
        program = spares.empty() ? "" : spares.front();
        if (!spares.empty()) spares.erase(spares.begin());
    }
    else if (which(program).empty()) {
        note = "(The browser " + program + " is not installed. This is the simpler text view, which runs no scripts.)";
        program.clear();
    } else program = which(program);
    const bool obscura = kind.empty() ? is_obscura(program) : kind == "obscura";
    Session s(program, obscura, chars, spares);

    signal(SIGPIPE, SIG_IGN);
    for (int sig : {SIGTERM, SIGINT, SIGHUP}) signal(sig, [](int) { g_quit = 1; });
    std::string in;
    double used = now_s();
    bool open = true;
    while (open && !g_quit) {
        size_t nl;
        while ((nl = in.find('\n')) == npos && open && !g_quit) {
            pollfd p{0, POLLIN, 0};
            const int r = poll(&p, 1, 1000);
            if (r > 0) {
                char buf[65536];
                const ssize_t n = read(0, buf, sizeof buf);
                if (n <= 0) open = false;
                else in.append(buf, (size_t)n);
            } else if (r == 0 && now_s() - used > idle) {
                s.rest();
            }
        }
        if (nl == npos) break;
        const std::string line = in.substr(0, nl);
        in.erase(0, nl + 1);
        if (line.find_first_not_of(" \r\t") == npos) continue;
        Json rq;
        Answer a;
        std::string extra;
        const std::string what = Json::parse(line, rq) ? rq["do"].str() : "";
        if (what == "open") a = s.open(rq["url"].str());
        else if (what == "click") a = s.click((int)rq["n"].num(0));
        else if (what == "type") a = s.type((int)rq["n"].num(0), rq["text"].str(), rq["enter"].boolean(true));
        else if (what == "more") a = s.more();
        else if (what == "back") a = s.back();
        else if (what == "what") {
            std::string say;
            bool asks = false;
            s.what(rq, say, asks);
            extra = std::string(",\"asks\":") + (asks ? "true" : "false") + ",\"line\":" + json_escape(clean_text(say));
        } else if (what == "status") {
            extra = std::string(",\"running\":") + (s.running() ? "true" : "false") + ",\"program\":" + json_escape(program);
        } else if (what == "close") {
            open = false;
        } else {
            a.ok = false;
            a.text = "ts5 browser: a request is one JSON object a line with \"do\": open, click, type, more, back, "
                     "what, status or close.";
        }
        if (a.ok && !note.empty() && (what == "open") ) {
            a.text = note + "\n" + a.text;
            note.clear();
        }
        const std::string out = std::string("{\"ok\":") + (a.ok ? "true" : "false") + ",\"text\":" + json_escape(a.text) +
                                ",\"url\":" + json_escape(s.url()) + ",\"title\":" + json_escape(s.title()) +
                                ",\"mode\":" + json_escape(s.mode()) + extra + "}\n";
        if (fwrite(out.data(), 1, out.size(), stdout) != out.size() || fflush(stdout) != 0) break;
        used = now_s();
    }
    s.quit();
    return 0;
}

}  // namespace ts5
