#include "keep_models.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "log.h"

namespace app {

namespace {

// A connection to port at addr (host order), the console itself by default.
int dial(int port, uint32_t addr = INADDR_LOOPBACK) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    timeval tv{5, 0};
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

// An FTP control connection, with what each reply said.
class Ftp {
public:
    ~Ftp() {
        if (fd_ >= 0) {
            send("QUIT");
            close(fd_);
        }
    }
    bool open(int port) {
        fd_ = dial(port);
        return fd_ >= 0 && reply() == 220;
    }
    // A command and its reply's code; 0 when the connection failed.
    int cmd(const std::string &line) { return send(line) ? reply() : 0; }
    const std::string &text() const { return text_; }
    // The names in a folder (NLST), over a passive data connection.
    bool names(const std::string &dir, std::vector<std::string> &out) {
        if (cmd("PASV") != 227) return false;
        unsigned h[4], p[2];
        const size_t paren = text_.find('(');
        if (paren == std::string::npos ||
            sscanf(text_.c_str() + paren, "(%u,%u,%u,%u,%u,%u)", &h[0], &h[1], &h[2], &h[3], &p[0], &p[1]) != 6)
            return false;
        // The console itself, else the address the server gives for it.
        const int port = (int)(p[0] * 256 + p[1]);
        int data = dial(port);
        if (data < 0) data = dial(port, h[0] << 24 | h[1] << 16 | h[2] << 8 | h[3]);
        if (data < 0) return false;
        const int code = cmd("NLST " + dir);
        std::string all;
        if (code == 150 || code == 125) {
            char buf[4096];
            ssize_t n;
            while ((n = recv(data, buf, sizeof buf, 0)) > 0) all.append(buf, (size_t)n);
        }
        close(data);
        if ((code != 150 && code != 125) || reply() != 226) return false;
        size_t i = 0;
        while (i < all.size()) {
            size_t e = all.find('\n', i);
            if (e == std::string::npos) e = all.size();
            std::string name = all.substr(i, e - i);
            if (!name.empty() && name.back() == '\r') name.pop_back();
            // Some servers give each name with the folder.
            const size_t slash = name.rfind('/');
            if (slash != std::string::npos) name = name.substr(slash + 1);
            if (!name.empty() && name != "." && name != "..") out.push_back(name);
            i = e + 1;
        }
        return true;
    }
    // A file's size, or -1.
    long long size(const std::string &path) {
        return cmd("SIZE " + path) == 213 ? strtoll(text_.c_str() + 4, nullptr, 10) : -1;
    }

private:
    bool send(const std::string &line) {
        const std::string s = line + "\r\n";
        return ::send(fd_, s.data(), s.size(), 0) == (ssize_t)s.size();
    }
    bool line(std::string &out) {
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
    int reply() {
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
    int fd_ = -1;
    std::string text_;
};

// A file that's being written, or isn't a model: left where it is.
bool skip(const std::string &name) {
    auto ends = [&](const char *e) {
        const size_t n = strlen(e);
        return name.size() >= n && name.compare(name.size() - n, n, e) == 0;
    };
    return name[0] == '.' || ends(".part") || ends(".new") || ends(".tmp");
}

}  // namespace

BroughtBack bring_back_models(const std::string &models, const std::string &title_id, const std::string &eboot,
                              const std::vector<int> &ports) {
    BroughtBack r;
    Ftp ftp;
    int port = 0;
    for (int p : ports)
        if (ftp.open(p)) {
            port = p;
            break;
        }
    if (!port) {
        r.why = "no FTP server on the console";
        return r;
    }
    if (ftp.cmd("USER anonymous") == 331) ftp.cmd("PASS anonymous");
    ftp.cmd("TYPE I");
    // The folder the server calls /data/homebrew/<title ID> is this one: its
    // eboot.bin is the size this one's is.
    const std::string home = "/data/homebrew/" + title_id, old = "/data/prosperostore/previous/" + title_id + "/models";
    struct stat st {};
    if (stat(eboot.c_str(), &st) != 0 || ftp.size(home + "/eboot.bin") != (long long)st.st_size) {
        r.why = "the FTP server on port " + std::to_string(port) + " doesn't show this title's folder at " + home;
        return r;
    }
    r.looked = true;
    std::vector<std::string> names;
    if (!ftp.names(old, names)) {
        r.why = "no models from a last version (" + old + ": " + ftp.text() + ")";
        return r;
    }
    for (const std::string &name : names) {
        if (skip(name)) continue;
        struct stat here {};
        if (stat((models + "/" + name).c_str(), &here) == 0) continue;  // the new folder has it already
        if (ftp.size(old + "/" + name) < 0) continue;                    // a folder, or gone
        if (ftp.cmd("RNFR " + old + "/" + name) == 350 && ftp.cmd("RNTO " + home + "/models/" + name) == 250 &&
            stat((models + "/" + name).c_str(), &here) == 0) {
            r.moved.push_back(name);
        } else {
            r.left++;
            ts5::log_line("ts5-title: couldn't bring back %s: %s", name.c_str(), ftp.text().c_str());
        }
    }
    return r;
}

}  // namespace app
