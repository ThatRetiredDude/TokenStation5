// The console's own FTP server (the jailbreak's ftpsrv), reached at
// 127.0.0.1. It runs outside the title's sandbox, so it can do two things
// the title can't: see /data (keep_models.h), and write to the drive at full
// speed. The title's own writes reach the disk at about 2 MB/s however
// they're made, while the server writes 20 to 30 MB/s, so downloads hand
// their bytes to it (FtpSink).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "net.h"

namespace app {

// A connection to port at addr (host order), the console itself by default;
// each send and receive waits at most timeout_s.
int dial(int port, uint32_t addr = 0x7f000001, int timeout_s = 5);

// An FTP control connection, with what each reply said.
class Ftp {
public:
    ~Ftp();
    // timeout_s: for each reply, and each send.
    bool open(int port, int timeout_s = 5);
    // USER anonymous, binary mode.
    void login();
    // A command and its reply's code; 0 when the connection failed.
    int cmd(const std::string &line);
    const std::string &text() const { return text_; }
    // The files in a folder, over a passive data connection: NLST's names,
    // else LIST's (the console's ftpsrv has no NLST), without its folders.
    bool names(const std::string &dir, std::vector<std::string> &out);
    // A file's size, or -1.
    long long size(const std::string &path);
    // A passive data connection for command (STOR, APPE, ...), once the
    // server says it's ready (150 or 125): its socket, else -1.
    int data(const std::string &command, int timeout_s = 5);
    // The reply that ends a transfer, after its data connection is closed.
    int reply();

private:
    bool transfer(const std::string &command, std::string &all);
    bool send(const std::string &line);
    bool line(std::string &out);
    int fd_ = -1;
    std::string text_;
};

// Where the console's FTP server is: TS5_FTP_PORT, then 2121 and 1337.
std::vector<int> ftp_ports();

// The port of the FTP server on the console (ports, in order) that shows
// the title's folder dir as dir_on_server: a file the title writes in dir,
// under a name of its own, is there the same size. 0 if none does, and why.
int find_console_ftp(const std::string &dir, const std::string &dir_on_server, const std::vector<int> &ports,
                     std::string *why = nullptr);

// A download's bytes, written by the FTP server instead of the title: STOR
// from the start, APPE after a part already there (the title cuts the part
// to where it goes on from first). remote: the file's path on the server.
class FtpSink : public ts5::FileSink {
public:
    FtpSink(int port, std::string remote) : port_(port), remote_(std::move(remote)) {}
    ~FtpSink() override;
    bool open(uint64_t at, std::string &err) override;
    bool write(const void *data, size_t n, std::string &err) override;
    bool close(std::string &err) override;

private:
    int port_;
    std::string remote_;
    Ftp ftp_;
    int data_ = -1;
};

}  // namespace app
