// ts5-tui's HTTP client (client.h).
#include "client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <strings.h>

namespace tui {

bool Endpoint::local() const {
    return host == "127.0.0.1" || host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
}

bool parse_endpoint(const std::string &url, Endpoint &ep, std::string &why) {
    std::string u = url;
    if (u.rfind("https://", 0) == 0) {
        why = "TokenStation's API is http://, not https://";
        return false;
    }
    if (u.rfind("http://", 0) == 0) u = u.substr(7);
    const size_t slash = u.find('/');
    if (slash != std::string::npos) u.resize(slash);
    if (u.empty()) {
        why = "no host in " + url;
        return false;
    }
    std::string host = u;
    int port = 8081;
    if (u[0] == '[') {  // [::1]:8081
        const size_t close = u.find(']');
        if (close == std::string::npos) {
            why = "bad address " + url;
            return false;
        }
        host = u.substr(1, close - 1);
        if (close + 1 < u.size() && u[close + 1] == ':') port = atoi(u.c_str() + close + 2);
    } else {
        const size_t colon = u.rfind(':');
        if (colon != std::string::npos && u.find(':') == colon) {
            host = u.substr(0, colon);
            port = atoi(u.c_str() + colon + 1);
        }
    }
    if (port <= 0 || port > 65535) {
        why = "bad port in " + url;
        return false;
    }
    ep.host = host;
    ep.port = port;
    return true;
}

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int connect_to(const Endpoint &ep, int timeout_ms, std::string &err) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const std::string port = std::to_string(ep.port);
    if (int rc = getaddrinfo(ep.host.c_str(), port.c_str(), &hints, &res); rc != 0) {
        err = "can't find " + ep.host + ": " + gai_strerror(rc);
        return -1;
    }
    int fd = -1;
    err = "can't connect to " + ep.host + ":" + port;
    for (addrinfo *a = res; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        const int fl = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        int rc = connect(fd, a->ai_addr, a->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            pollfd p{fd, POLLOUT, 0};
            rc = poll(&p, 1, timeout_ms) == 1 ? 0 : -1;
            int soerr = 0;
            socklen_t len = sizeof soerr;
            if (rc == 0 && (getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len) != 0 || soerr != 0)) {
                rc = -1;
                if (soerr == ECONNREFUSED) err = "nothing is listening on " + ep.host + ":" + port;
            } else if (rc != 0) {
                err = ep.host + ":" + port + " did not answer";
            }
        } else if (rc != 0 && errno == ECONNREFUSED) {
            err = "nothing is listening on " + ep.host + ":" + port;
        }
        if (rc == 0) {
            fcntl(fd, F_SETFL, fl);
            const int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

bool send_all(int fd, const std::string &s) {
    size_t off = 0;
    while (off < s.size()) {
#ifdef MSG_NOSIGNAL
        const ssize_t n = send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
#else
        const ssize_t n = send(fd, s.data() + off, s.size() - off, 0);
#endif
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        off += (size_t)n;
    }
    return true;
}

}  // namespace

Reply http(const Endpoint &ep, const std::string &method, const std::string &path, const std::string &body,
           int timeout_ms, const std::function<bool(const std::string &)> &on_data, std::atomic<int> *fd_out) {
    Reply r;
    const int fd = connect_to(ep, std::min(timeout_ms, 3000), r.err);
    if (fd < 0) return r;
    if (fd_out) fd_out->store(fd);
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + ep.host + ":" + std::to_string(ep.port) +
                      "\r\nUser-Agent: ts5-tui\r\nAccept: */*\r\nConnection: close\r\n";
    if (!ep.key.empty()) req += "Authorization: Bearer " + ep.key + "\r\n";
    if (!body.empty() || method == "POST")
        req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    auto finish = [&] {
        if (fd_out) fd_out->store(-1);
        close(fd);
        return r;
    };
    if (!send_all(fd, req)) {
        r.err = "the connection closed while sending";
        return finish();
    }
    // Headers, then the body: Content-Length, chunked, or to the close.
    std::string buf, head;
    bool have_head = false, chunked = false, stopped = false;
    long long length = -1, got = 0;
    std::string chunk_buf;  // chunked: bytes not yet a whole chunk
    auto deliver = [&](const std::string &data) {
        if (data.empty() || stopped) return;
        got += (long long)data.size();
        // An error's body comes back whole in Reply::body, for the caller.
        if (on_data && r.status < 400) {
            if (!on_data(data)) stopped = true;
        } else {
            r.body += data;
        }
    };
    auto dechunk = [&](const std::string &data) {
        chunk_buf += data;
        for (;;) {
            const size_t eol = chunk_buf.find("\r\n");
            if (eol == std::string::npos) return;
            const long long n = strtoll(chunk_buf.c_str(), nullptr, 16);
            if (n == 0) {
                stopped = true;
                return;
            }
            if (chunk_buf.size() < eol + 2 + (size_t)n + 2) return;
            deliver(chunk_buf.substr(eol + 2, (size_t)n));
            chunk_buf.erase(0, eol + 2 + (size_t)n + 2);
        }
    };
    double deadline = now_ms() + timeout_ms;
    char tmp[16384];
    while (!stopped) {
        const int left = (int)(deadline - now_ms());
        if (left <= 0) {
            r.err = "no answer within " + std::to_string(timeout_ms / 1000) + " s";
            break;
        }
        pollfd p{fd, POLLIN, 0};
        const int pr = poll(&p, 1, std::min(left, 250));
        if (pr < 0 && errno != EINTR) break;
        if (pr <= 0) continue;
        const ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;  // closed (or shut down by another thread)
        // While data flows, the time limit is from the last bytes: a long
        // answer streams for minutes.
        if (on_data) deadline = now_ms() + timeout_ms;
        if (!have_head) {
            buf.append(tmp, (size_t)n);
            const size_t end = buf.find("\r\n\r\n");
            if (end == std::string::npos) {
                if (buf.size() > 65536) break;
                continue;
            }
            head = buf.substr(0, end);
            const std::string rest = buf.substr(end + 4);
            have_head = true;
            r.status = atoi(head.c_str() + (head.find(' ') == std::string::npos ? 0 : head.find(' ') + 1));
            size_t pos = 0;
            while ((pos = head.find("\r\n", pos)) != std::string::npos) {
                pos += 2;
                const size_t colon = head.find(':', pos);
                const size_t eol = head.find("\r\n", pos);
                if (colon == std::string::npos || (eol != std::string::npos && colon > eol)) continue;
                std::string name = head.substr(pos, colon - pos);
                std::string val = head.substr(colon + 1, (eol == std::string::npos ? head.size() : eol) - colon - 1);
                while (!val.empty() && val[0] == ' ') val.erase(0, 1);
                if (strcasecmp(name.c_str(), "content-length") == 0) length = atoll(val.c_str());
                if (strcasecmp(name.c_str(), "transfer-encoding") == 0 && strcasestr(val.c_str(), "chunked"))
                    chunked = true;
            }
            if (chunked) dechunk(rest); else deliver(rest);
        } else {
            const std::string data(tmp, (size_t)n);
            if (chunked) dechunk(data); else deliver(data);
        }
        if (length >= 0 && got >= length) break;
    }
    if (!have_head && r.err.empty()) r.err = "the server closed the connection without an answer";
    if (!have_head) r.status = 0;
    return finish();
}

}  // namespace tui
