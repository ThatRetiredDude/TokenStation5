// The HTTP/2 that Tailscale's control server speaks inside the Noise
// connection (ps5/ts/noise.h): a client of just what the PS5 needs. Requests
// with small bodies, responses read as they come (the map is one long
// response), PINGs answered, the receive windows opened as data arrives.
// HPACK: headers sent as plain literals; received ones decoded in full
// (static table, dynamic table, Huffman), though only :status is kept.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ts {

class H2 {
public:
    struct Response {
        int status = 0;     // 0 until the headers came
        std::string body;   // what has come of the body and isn't taken yet
        bool done = false;  // the server finished it (or reset it: reset set)
        bool reset = false;
    };

    // The client's opening: the connection preface and its SETTINGS.
    std::string start();
    // A request's frames; id is its stream.
    std::string request(const std::string &method, const std::string &authority, const std::string &path,
                        const std::vector<std::pair<std::string, std::string>> &headers, const std::string &body,
                        int &id);
    // Bytes from the server. Anything to send back (SETTINGS and PING acks,
    // window updates) is appended to reply. False, with why, on a protocol
    // error: the connection is then done.
    bool feed(const char *data, size_t len, std::string &reply, std::string &why);

    Response *response(int id);
    void forget(int id) { streams_.erase(id); }
    // The server said GOAWAY: no new requests on this connection.
    bool going_away() const { return goaway_; }

private:
    bool frame(uint8_t type, uint8_t flags, uint32_t id, const uint8_t *p, size_t n, std::string &reply,
               std::string &why);
    bool headers(uint32_t id, const std::string &block, std::string &why);
    std::string in_;
    std::map<int, Response> streams_;
    int next_id_ = 1;
    bool goaway_ = false;
    // A header block still in CONTINUATION frames.
    uint32_t cont_id_ = 0;
    bool cont_end_stream_ = false;
    std::string cont_;
    // HPACK's dynamic table, newest first, and its size by RFC 7541's count.
    std::vector<std::pair<std::string, std::string>> dyn_;
    size_t dyn_size_ = 0, dyn_max_ = 4096;
};

}  // namespace ts
