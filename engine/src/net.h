// A small HTTP(S) client: one POST read to the end, for the web search tool,
// and a GET streamed to a file, for downloading a model onto the console.
// HTTPS is Mbed TLS, checking the server's certificate against the system's
// CA bundle, or TS5_CA_FILE, or the Mozilla bundle built in (the console has
// no CA files a title can read).
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ts5 {

struct HttpResponse {
    int status = 0;
    std::string body;
};

// n random bytes from the system (kern.arandom on the PS5, /dev/urandom
// elsewhere); false if it has none.
bool random_bytes(unsigned char *out, size_t n);

// The SHA-256 of data in lowercase hex; "" in a build without HTTPS.
std::string sha256_hex(const std::string &data);

// url: http://host[:port]/path or https://host[:port]/path. headers: "Name: value".
bool http_post(const std::string &url, const std::vector<std::string> &headers, const std::string &body,
               HttpResponse &resp, std::string &err, int timeout_s = 30);

// GET url into resp, at most max_bytes of its body (the rest is left
// unread), following up to 5 redirects; headers go to every hop.
bool http_get(const std::string &url, const std::vector<std::string> &headers, HttpResponse &resp, std::string &err,
              size_t max_bytes = 16u << 20, int timeout_s = 30);

// How a download is going, for another thread to read while it runs, and
// two flags for that thread to set. The download looks at them between
// writes (every 256 KiB) and while it hashes a part it goes on from.
struct FetchProgress {
    std::atomic<uint64_t> done{0};   // bytes in the file so far, the part it went on from included
    std::atomic<uint64_t> total{0};  // bytes the whole file has; 0 until the server says
    std::atomic<double> speed{0};    // bytes a second, over the last couple of seconds
    std::atomic<bool> cancel{false};  // set: the download stops and fails with "cancelled"
    std::atomic<bool> pause{false};   // set: the download waits, the connection held open
};

// The SHA-256 so far, at an offset in the file, so a download can go on
// from a checkpoint without hashing the part again. The state is the
// mbedtls_sha256_context copied as bytes, so this header needs no Mbed TLS.
struct Sha256Resume {
    uint64_t offset = 0;
    std::vector<unsigned char> state;  // empty: no checkpoint
    // Every 64 MiB, once the part is flushed: the offset and the state there.
    std::function<void(uint64_t offset, const std::vector<unsigned char> &state)> on_checkpoint;
};

// GET url into the file at path, which already holds its first `offset`
// bytes (0: none), asking the server for the rest with Range. Follows up to
// 5 redirects. Writes 256 KiB at a time through a 1 MiB buffer, with the
// SHA-256 updated as it goes; with expected_sha256 given (hex), fails when
// the finished file's hash is another. A body with neither a size nor
// chunks is taken only with a hash to check it by. Beside the file, path +
// ".from" notes the URL (without its query string) and the server's ETag or
// Last-Modified, so only a part of the same URL is gone on from, asking
// with If-Range, and a file that changed on the server starts over; it goes
// once the file is whole. The file is kept on a failure the next call can
// go on from (a dropped connection, a timeout, a full disk, a cancel) and
// removed when it turned out wrong (another size or hash than it should
// have, or longer than the file), so that call starts over. sha: a
// checkpoint to go on from, and where to report new ones; without one, a
// resume hashes the file's first `offset` bytes first. Each socket
// operation has 30 s. A full disk reports "storage full".
bool http_get_file(const std::string &url, const std::string &path, uint64_t offset,
                   const std::string &expected_sha256, FetchProgress &progress, std::string &err,
                   Sha256Resume *sha = nullptr);

// A connection held open both ways, plain or TLS, for protocols that send and
// receive as they go (Tailscale's control server and relays, ps5/ts). One
// thread at a time.
class Stream {
public:
    Stream();
    ~Stream();
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
    // Connects to host:port (TLS checks the certificate as the requests above
    // do); timeout_s is for the connection and for each write.
    bool open(const std::string &host, int port, bool tls, int timeout_s, std::string &err);
    bool write(const void *data, size_t len, std::string &err);
    // Up to cap bytes, waiting at most wait_ms for the first: how many, 0
    // once the other side has closed, -1 on an error (err says), -2 when
    // nothing came in time.
    int read(void *buf, size_t cap, int wait_ms, std::string &err);
    void close();
    bool is_open() const;

private:
    struct Impl;
    Impl *p_;
};

}  // namespace ts5
