#include "ftp.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "settings.h"

namespace app {

int dial(int port, uint32_t addr, int timeout_s) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    timeval tv{timeout_s, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    sa.sin_addr.s_addr = htonl(addr);
    if (connect(fd, (sockaddr *)&sa, sizeof sa) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

namespace {

std::vector<std::string> lines(const std::string &all) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < all.size()) {
        size_t e = all.find('\n', i);
        if (e == std::string::npos) e = all.size();
        std::string l = all.substr(i, e - i);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        out.push_back(l);
        i = e + 1;
    }
    return out;
}

}  // namespace

Ftp::~Ftp() {
    if (fd_ >= 0) {
        send("QUIT");
        ::close(fd_);
    }
}

bool Ftp::open(int port, int timeout_s) {
    fd_ = dial(port, 0x7f000001, timeout_s);
    return fd_ >= 0 && reply() == 220;
}

void Ftp::login() {
    if (cmd("USER anonymous") == 331) cmd("PASS anonymous");
    cmd("TYPE I");
}

int Ftp::cmd(const std::string &line) { return send(line) ? reply() : 0; }

bool Ftp::names(const std::string &dir, std::vector<std::string> &out) {
    std::string all;
    if (transfer("NLST " + dir, all)) {
        for (std::string &name : lines(all)) {
            // Some servers give each name with the folder.
            const size_t slash = name.rfind('/');
            if (slash != std::string::npos) name = name.substr(slash + 1);
            if (!name.empty() && name != "." && name != "..") out.push_back(name);
        }
        return true;
    }
    if (!transfer("LIST " + dir, all)) return false;
    // ls -l lines: mode, links, owner, group, size, 3 for the time, name.
    for (const std::string &l : lines(all)) {
        if (l.empty() || l[0] != '-') continue;  // a folder, a link, or "total"
        size_t i = 0;
        for (int field = 0; field < 8 && i < l.size(); field++) {
            while (i < l.size() && l[i] != ' ') i++;
            while (i < l.size() && l[i] == ' ') i++;
        }
        if (i < l.size()) out.push_back(l.substr(i));
    }
    return true;
}

long long Ftp::size(const std::string &path) {
    return cmd("SIZE " + path) == 213 ? strtoll(text_.c_str() + 4, nullptr, 10) : -1;
}

int Ftp::data(const std::string &command, int timeout_s) {
    if (cmd("PASV") != 227) return -1;
    unsigned h[4], p[2];
    const size_t paren = text_.find('(');
    if (paren == std::string::npos ||
        sscanf(text_.c_str() + paren, "(%u,%u,%u,%u,%u,%u)", &h[0], &h[1], &h[2], &h[3], &p[0], &p[1]) != 6)
        return -1;
    // The console itself, else the address the server gives for it.
    const int port = (int)(p[0] * 256 + p[1]);
    int fd = dial(port, 0x7f000001, timeout_s);
    if (fd < 0) fd = dial(port, h[0] << 24 | h[1] << 16 | h[2] << 8 | h[3], timeout_s);
    if (fd < 0) return -1;
    const int code = cmd(command);
    if (code != 150 && code != 125) {
        ::close(fd);
        return -1;
    }
    return fd;
}

bool Ftp::transfer(const std::string &command, std::string &all) {
    all.clear();
    const int fd = data(command);
    if (fd < 0) return false;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof buf, 0)) > 0) all.append(buf, (size_t)n);
    ::close(fd);
    return reply() / 100 == 2;
}

bool Ftp::send(const std::string &line) {
    const std::string s = line + "\r\n";
    return ::send(fd_, s.data(), s.size(), 0) == (ssize_t)s.size();
}

bool Ftp::line(std::string &out) {
    out.clear();
    char c;
    while (recv(fd_, &c, 1, 0) == 1) {
        if (c == '\n') {
            if (!out.empty() && out.back() == '\r') out.pop_back();
            return true;
        }
        out += c;
    }
    return false;
}

// A reply's code; a reply of several lines ends with its code and a space.
int Ftp::reply() {
    std::string l;
    if (!line(l) || l.size() < 3) return 0;
    text_ = l;
    const int code = atoi(l.substr(0, 3).c_str());
    if (l.size() > 3 && l[3] == '-') {
        const std::string end = l.substr(0, 3) + " ";
        while (line(l))
            if (l.compare(0, 4, end) == 0) {
                text_ = l;
                break;
            }
    }
    return code;
}

std::vector<int> ftp_ports() {
    std::vector<int> ports = {2121, 1337};
    if (int p = atoi(setting("TS5_FTP_PORT").c_str()); p > 0) ports.insert(ports.begin(), p);
    return ports;
}

int find_console_ftp(const std::string &dir, const std::string &dir_on_server, const std::vector<int> &ports,
                     std::string *why) {
    Ftp ftp;
    int port = 0;
    for (int p : ports)
        if (ftp.open(p)) {
            port = p;
            break;
        }
    if (!port) {
        if (why) *why = "no FTP server on the console";
        return 0;
    }
    ftp.login();
    const std::string mark = ".ts5-here-" + std::to_string(getpid()) + "-" + std::to_string(time(nullptr));
    bool seen = false;
    if (FILE *f = fopen((dir + "/" + mark).c_str(), "wb")) {
        fputs(mark.c_str(), f);
        fclose(f);
        seen = ftp.size(dir_on_server + "/" + mark) == (long long)mark.size();
        unlink((dir + "/" + mark).c_str());
    }
    if (!seen) {
        if (why) *why = "the FTP server on port " + std::to_string(port) + " doesn't show this title's folder at " +
                        dir_on_server;
        return 0;
    }
    return port;
}

FtpSink::~FtpSink() {
    if (data_ >= 0) ::close(data_);
}

bool FtpSink::open(uint64_t at, std::string &err) {
    // 30 s for each send and reply: the server may wait on the drive.
    if (!ftp_.open(port_, 30)) {
        err = "the console's FTP server didn't answer on port " + std::to_string(port_);
        return false;
    }
    ftp_.login();
    data_ = ftp_.data((at > 0 ? "APPE " : "STOR ") + remote_, 30);
    if (data_ < 0) {
        err = "the console's FTP server wouldn't take the file: " + ftp_.text();
        return false;
    }
    const int buf = 4 << 20;
    setsockopt(data_, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    return true;
}

bool FtpSink::write(const void *data, size_t n, std::string &err) {
    const char *p = (const char *)data;
    while (n > 0) {
        const ssize_t sent = ::send(data_, p, n, 0);
        if (sent <= 0) {
            err = std::string("cannot write the file: the console's FTP server stopped taking it (") +
                  (sent < 0 ? strerror(errno) : "closed") + ")";
            return false;
        }
        p += sent;
        n -= (size_t)sent;
    }
    return true;
}

bool FtpSink::close(std::string &err) {
    if (data_ < 0) return true;
    ::close(data_);
    data_ = -1;
    // 226 once the server has written it all.
    if (ftp_.reply() / 100 != 2) {
        err = "cannot write the file: the console's FTP server said " + ftp_.text();
        return false;
    }
    return true;
}

}  // namespace app
