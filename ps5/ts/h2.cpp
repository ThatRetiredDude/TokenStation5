#include "h2.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ts {

namespace {

enum : uint8_t { DATA = 0, HEADERS = 1, PRIORITY = 2, RST_STREAM = 3, SETTINGS = 4, PUSH_PROMISE = 5, PING = 6,
                 GOAWAY = 7, WINDOW_UPDATE = 8, CONTINUATION = 9 };
enum : uint8_t { END_STREAM = 1, ACK = 1, END_HEADERS = 4, PADDED = 8, PRIO = 0x20 };

constexpr uint32_t kWindow = 1u << 30;
constexpr size_t kMaxFrame = 16384;  // the default SETTINGS_MAX_FRAME_SIZE, both ways

// HPACK's Huffman code (RFC 7541, appendix B): code and bit length of each byte.
static const uint32_t kHuffCode[256] = {
    0x1ff8, 0x7fffd8, 0xfffffe2, 0xfffffe3, 0xfffffe4, 0xfffffe5, 0xfffffe6, 0xfffffe7,
    0xfffffe8, 0xffffea, 0x3ffffffc, 0xfffffe9, 0xfffffea, 0x3ffffffd, 0xfffffeb, 0xfffffec,
    0xfffffed, 0xfffffee, 0xfffffef, 0xffffff0, 0xffffff1, 0xffffff2, 0x3ffffffe, 0xffffff3,
    0xffffff4, 0xffffff5, 0xffffff6, 0xffffff7, 0xffffff8, 0xffffff9, 0xffffffa, 0xffffffb,
    0x14, 0x3f8, 0x3f9, 0xffa, 0x1ff9, 0x15, 0xf8, 0x7fa,
    0x3fa, 0x3fb, 0xf9, 0x7fb, 0xfa, 0x16, 0x17, 0x18,
    0x0, 0x1, 0x2, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
    0x1e, 0x1f, 0x5c, 0xfb, 0x7ffc, 0x20, 0xffb, 0x3fc,
    0x1ffa, 0x21, 0x5d, 0x5e, 0x5f, 0x60, 0x61, 0x62,
    0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
    0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72,
    0xfc, 0x73, 0xfd, 0x1ffb, 0x7fff0, 0x1ffc, 0x3ffc, 0x22,
    0x7ffd, 0x3, 0x23, 0x4, 0x24, 0x5, 0x25, 0x26,
    0x27, 0x6, 0x74, 0x75, 0x28, 0x29, 0x2a, 0x7,
    0x2b, 0x76, 0x2c, 0x8, 0x9, 0x2d, 0x77, 0x78,
    0x79, 0x7a, 0x7b, 0x7ffe, 0x7fc, 0x3ffd, 0x1ffd, 0xffffffc,
    0xfffe6, 0x3fffd2, 0xfffe7, 0xfffe8, 0x3fffd3, 0x3fffd4, 0x3fffd5, 0x7fffd9,
    0x3fffd6, 0x7fffda, 0x7fffdb, 0x7fffdc, 0x7fffdd, 0x7fffde, 0xffffeb, 0x7fffdf,
    0xffffec, 0xffffed, 0x3fffd7, 0x7fffe0, 0xffffee, 0x7fffe1, 0x7fffe2, 0x7fffe3,
    0x7fffe4, 0x1fffdc, 0x3fffd8, 0x7fffe5, 0x3fffd9, 0x7fffe6, 0x7fffe7, 0xffffef,
    0x3fffda, 0x1fffdd, 0xfffe9, 0x3fffdb, 0x3fffdc, 0x7fffe8, 0x7fffe9, 0x1fffde,
    0x7fffea, 0x3fffdd, 0x3fffde, 0xfffff0, 0x1fffdf, 0x3fffdf, 0x7fffeb, 0x7fffec,
    0x1fffe0, 0x1fffe1, 0x3fffe0, 0x1fffe2, 0x7fffed, 0x3fffe1, 0x7fffee, 0x7fffef,
    0xfffea, 0x3fffe2, 0x3fffe3, 0x3fffe4, 0x7ffff0, 0x3fffe5, 0x3fffe6, 0x7ffff1,
    0x3ffffe0, 0x3ffffe1, 0xfffeb, 0x7fff1, 0x3fffe7, 0x7ffff2, 0x3fffe8, 0x1ffffec,
    0x3ffffe2, 0x3ffffe3, 0x3ffffe4, 0x7ffffde, 0x7ffffdf, 0x3ffffe5, 0xfffff1, 0x1ffffed,
    0x7fff2, 0x1fffe3, 0x3ffffe6, 0x7ffffe0, 0x7ffffe1, 0x3ffffe7, 0x7ffffe2, 0xfffff2,
    0x1fffe4, 0x1fffe5, 0x3ffffe8, 0x3ffffe9, 0xffffffd, 0x7ffffe3, 0x7ffffe4, 0x7ffffe5,
    0xfffec, 0xfffff3, 0xfffed, 0x1fffe6, 0x3fffe9, 0x1fffe7, 0x1fffe8, 0x7ffff3,
    0x3fffea, 0x3fffeb, 0x1ffffee, 0x1ffffef, 0xfffff4, 0xfffff5, 0x3ffffea, 0x7ffff4,
    0x3ffffeb, 0x7ffffe6, 0x3ffffec, 0x3ffffed, 0x7ffffe7, 0x7ffffe8, 0x7ffffe9, 0x7ffffea,
    0x7ffffeb, 0xffffffe, 0x7ffffec, 0x7ffffed, 0x7ffffee, 0x7ffffef, 0x7fffff0, 0x3ffffee,
};
static const uint8_t kHuffLen[256] = {
    13, 23, 28, 28, 28, 28, 28, 28, 28, 24, 30, 28, 28, 30, 28, 28,
    28, 28, 28, 28, 28, 28, 30, 28, 28, 28, 28, 28, 28, 28, 28, 28,
    6, 10, 10, 12, 13, 6, 8, 11, 10, 10, 8, 11, 8, 6, 6, 6,
    5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 15, 6, 12, 10,
    13, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 8, 7, 8, 13, 19, 13, 14, 6,
    15, 5, 6, 5, 6, 5, 6, 6, 6, 5, 7, 7, 6, 6, 6, 5,
    6, 7, 6, 5, 5, 6, 7, 7, 7, 7, 7, 15, 11, 14, 13, 28,
    20, 22, 20, 20, 22, 22, 22, 23, 22, 23, 23, 23, 23, 23, 24, 23,
    24, 24, 22, 23, 24, 23, 23, 23, 23, 21, 22, 23, 22, 23, 23, 24,
    22, 21, 20, 22, 22, 23, 23, 21, 23, 22, 22, 24, 21, 22, 23, 23,
    21, 21, 22, 21, 23, 22, 23, 23, 20, 22, 22, 22, 23, 22, 22, 23,
    26, 26, 20, 19, 22, 23, 22, 25, 26, 26, 26, 27, 27, 26, 24, 25,
    19, 21, 26, 27, 27, 26, 27, 24, 21, 21, 26, 26, 28, 27, 27, 27,
    20, 24, 20, 21, 22, 21, 21, 23, 22, 22, 25, 25, 24, 24, 26, 23,
    26, 27, 26, 26, 27, 27, 27, 27, 27, 28, 27, 27, 27, 27, 27, 26,
};

// HPACK's static table (RFC 7541, appendix A), from index 1.
const char *const kStatic[61][2] = {
    {":authority", ""}, {":method", "GET"}, {":method", "POST"}, {":path", "/"}, {":path", "/index.html"},
    {":scheme", "http"}, {":scheme", "https"}, {":status", "200"}, {":status", "204"}, {":status", "206"},
    {":status", "304"}, {":status", "400"}, {":status", "404"}, {":status", "500"}, {"accept-charset", ""},
    {"accept-encoding", "gzip, deflate"}, {"accept-language", ""}, {"accept-ranges", ""}, {"accept", ""},
    {"access-control-allow-origin", ""}, {"age", ""}, {"allow", ""}, {"authorization", ""}, {"cache-control", ""},
    {"content-disposition", ""}, {"content-encoding", ""}, {"content-language", ""}, {"content-length", ""},
    {"content-location", ""}, {"content-range", ""}, {"content-type", ""}, {"cookie", ""}, {"date", ""},
    {"etag", ""}, {"expect", ""}, {"expires", ""}, {"from", ""}, {"host", ""}, {"if-match", ""},
    {"if-modified-since", ""}, {"if-none-match", ""}, {"if-range", ""}, {"if-unmodified-since", ""},
    {"last-modified", ""}, {"link", ""}, {"location", ""}, {"max-forwards", ""}, {"proxy-authenticate", ""},
    {"proxy-authorization", ""}, {"range", ""}, {"referer", ""}, {"refresh", ""}, {"retry-after", ""},
    {"server", ""}, {"set-cookie", ""}, {"strict-transport-security", ""}, {"transfer-encoding", ""},
    {"user-agent", ""}, {"vary", ""}, {"via", ""}, {"www-authenticate", ""},
};

void put_frame(std::string &out, uint8_t type, uint8_t flags, uint32_t id, const void *p, size_t n) {
    const uint8_t h[9] = {(uint8_t)(n >> 16), (uint8_t)(n >> 8), (uint8_t)n, type, flags,
                          (uint8_t)((id >> 24) & 0x7f), (uint8_t)(id >> 16), (uint8_t)(id >> 8), (uint8_t)id};
    out.append((const char *)h, 9);
    if (n) out.append((const char *)p, n);
}

void put_u32(std::string &out, uint32_t v) {
    const uint8_t b[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
    out.append((const char *)b, 4);
}

void window_update(std::string &out, uint32_t id, uint32_t n) {
    std::string p;
    put_u32(p, n & 0x7fffffffu);
    put_frame(out, WINDOW_UPDATE, 0, id, p.data(), p.size());
}

// An HPACK integer with an n-bit prefix; first carries the bits above it.
void put_int(std::string &out, uint8_t first, int bits, size_t v) {
    const size_t max = (1u << bits) - 1;
    if (v < max) {
        out += (char)(first | v);
        return;
    }
    out += (char)(first | max);
    v -= max;
    while (v >= 128) {
        out += (char)(0x80 | (v & 0x7f));
        v >>= 7;
    }
    out += (char)v;
}

bool get_int(const uint8_t *&p, const uint8_t *end, int bits, size_t &v) {
    if (p >= end) return false;
    const size_t max = (1u << bits) - 1;
    v = *p++ & max;
    if (v < max) return true;
    for (int shift = 0; shift < 28; shift += 7) {
        if (p >= end) return false;
        const uint8_t b = *p++;
        v += (size_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) return true;
    }
    return false;
}

bool huffman(const uint8_t *p, size_t n, std::string &out) {
    uint64_t bits = 0;
    int have = 0;
    for (size_t i = 0; i < n; i++) {
        bits = (bits << 8) | p[i];
        have += 8;
        // Codes are 5 to 30 bits: match the shortest that fits.
        while (have >= 5) {
            bool found = false;
            for (int len = 5; len <= 30 && len <= have && !found; len++) {
                const uint32_t code = (uint32_t)((bits >> (have - len)) & ((1ull << len) - 1));
                for (int c = 0; c < 256; c++)
                    if (kHuffLen[c] == len && kHuffCode[c] == code) {
                        out += (char)c;
                        have -= len;
                        found = true;
                        break;
                    }
            }
            if (!found) break;
        }
        if (have > 30) return false;
    }
    // What's left must be padding: fewer than 8 bits, all ones.
    return have < 8 && (bits & ((1ull << have) - 1)) == ((1ull << have) - 1);
}

bool get_str(const uint8_t *&p, const uint8_t *end, std::string &s) {
    if (p >= end) return false;
    const bool huff = *p & 0x80;
    size_t n;
    if (!get_int(p, end, 7, n) || (size_t)(end - p) < n) return false;
    s.clear();
    if (huff) {
        if (!huffman(p, n, s)) return false;
    } else {
        s.assign((const char *)p, n);
    }
    p += n;
    return true;
}

}  // namespace

std::string H2::start() {
    std::string out = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
    std::string s;
    auto setting = [&](uint16_t k, uint32_t v) {
        s += (char)(k >> 8);
        s += (char)k;
        put_u32(s, v);
    };
    setting(2, 0);        // ENABLE_PUSH: off
    setting(4, kWindow);  // INITIAL_WINDOW_SIZE
    put_frame(out, SETTINGS, 0, 0, s.data(), s.size());
    window_update(out, 0, kWindow - 65535);
    return out;
}

std::string H2::request(const std::string &method, const std::string &authority, const std::string &path,
                        const std::vector<std::pair<std::string, std::string>> &hdrs, const std::string &body, int &id) {
    id = next_id_;
    next_id_ += 2;
    std::string block;
    auto literal = [&](const std::string &k, const std::string &v) {
        block += (char)0x00;  // literal without indexing, new name
        put_int(block, 0, 7, k.size());
        block += k;
        put_int(block, 0, 7, v.size());
        block += v;
    };
    literal(":method", method);
    literal(":scheme", "https");
    literal(":authority", authority);
    literal(":path", path);
    for (const auto &[k, v] : hdrs) literal(k, v);
    if (!body.empty()) literal("content-length", std::to_string(body.size()));
    std::string out;
    const uint8_t first = body.empty() ? END_STREAM : 0;
    for (size_t at = 0; at < block.size() || at == 0;) {
        const size_t n = std::min(kMaxFrame, block.size() - at);
        const bool last = at + n >= block.size();
        put_frame(out, at == 0 ? HEADERS : CONTINUATION, (uint8_t)((at == 0 ? first : 0) | (last ? END_HEADERS : 0)),
                  (uint32_t)id, block.data() + at, n);
        at += n;
        if (last) break;
    }
    for (size_t at = 0; at < body.size();) {
        const size_t n = std::min(kMaxFrame, body.size() - at);
        put_frame(out, DATA, at + n >= body.size() ? END_STREAM : 0, (uint32_t)id, body.data() + at, n);
        at += n;
    }
    streams_[id];
    return out;
}

H2::Response *H2::response(int id) {
    auto it = streams_.find(id);
    return it == streams_.end() ? nullptr : &it->second;
}

bool H2::feed(const char *data, size_t len, std::string &reply, std::string &why) {
    in_.append(data, len);
    size_t at = 0;
    while (in_.size() - at >= 9) {
        const uint8_t *h = (const uint8_t *)in_.data() + at;
        const size_t n = ((size_t)h[0] << 16) | ((size_t)h[1] << 8) | h[2];
        if (n > (1u << 20)) {
            why = "an HTTP/2 frame too big";
            return false;
        }
        if (in_.size() - at < 9 + n) break;
        const uint32_t id = (((uint32_t)h[5] & 0x7f) << 24) | ((uint32_t)h[6] << 16) | ((uint32_t)h[7] << 8) | h[8];
        if (!frame(h[3], h[4], id, h + 9, n, reply, why)) return false;
        at += 9 + n;
    }
    in_.erase(0, at);
    return true;
}

bool H2::frame(uint8_t type, uint8_t flags, uint32_t id, const uint8_t *p, size_t n, std::string &reply,
               std::string &why) {
    if (cont_id_ && type != CONTINUATION) {
        why = "HTTP/2 headers broken off";
        return false;
    }
    // Padding off DATA and HEADERS.
    auto unpad = [&]() -> bool {
        if (!(flags & PADDED)) return true;
        if (n < 1 || p[0] >= n) return false;
        n -= 1 + p[0];
        p += 1;
        return true;
    };
    switch (type) {
        case DATA: {
            const size_t full = n;
            if (!unpad()) break;
            auto it = streams_.find((int)id);
            if (it != streams_.end()) {
                it->second.body.append((const char *)p, n);
                if (flags & END_STREAM) it->second.done = true;
            }
            if (full) {
                window_update(reply, 0, (uint32_t)full);
                if (!(flags & END_STREAM) && it != streams_.end()) window_update(reply, id, (uint32_t)full);
            }
            return true;
        }
        case HEADERS: {
            if (!unpad()) break;
            if (flags & PRIO) {
                if (n < 5) break;
                p += 5;
                n -= 5;
            }
            cont_.assign((const char *)p, n);
            cont_end_stream_ = flags & END_STREAM;
            if (!(flags & END_HEADERS)) {
                cont_id_ = id;
                return true;
            }
            if (!headers(id, cont_, why)) return false;
            if (cont_end_stream_)
                if (Response *r = response((int)id)) r->done = true;
            return true;
        }
        case CONTINUATION: {
            if (id != cont_id_) break;
            cont_.append((const char *)p, n);
            if (!(flags & END_HEADERS)) return true;
            cont_id_ = 0;
            if (!headers(id, cont_, why)) return false;
            if (cont_end_stream_)
                if (Response *r = response((int)id)) r->done = true;
            return true;
        }
        case RST_STREAM:
            if (Response *r = response((int)id)) r->done = r->reset = true;
            return true;
        case SETTINGS:
            if (!(flags & ACK)) {
                for (size_t i = 0; i + 6 <= n; i += 6) {
                    const uint16_t k = (uint16_t)((p[i] << 8) | p[i + 1]);
                    const uint32_t v = ((uint32_t)p[i + 2] << 24) | ((uint32_t)p[i + 3] << 16) | ((uint32_t)p[i + 4] << 8) | p[i + 5];
                    (void)k;
                    (void)v;  // the frames sent here are small; nothing to adjust
                }
                put_frame(reply, SETTINGS, ACK, 0, nullptr, 0);
            }
            return true;
        case PING:
            if (!(flags & ACK) && n == 8) put_frame(reply, PING, ACK, 0, p, 8);
            return true;
        case GOAWAY:
            goaway_ = true;
            return true;
        case PUSH_PROMISE:
            why = "the server pushed, which was turned off";
            return false;
        default:  // PRIORITY, WINDOW_UPDATE and unknown frames: nothing to do
            return true;
    }
    why = "a malformed HTTP/2 frame";
    return false;
}

bool H2::headers(uint32_t id, const std::string &block, std::string &why) {
    const uint8_t *p = (const uint8_t *)block.data(), *end = p + block.size();
    auto lookup = [&](size_t i, std::string &k, std::string &v) -> bool {
        if (i >= 1 && i <= 61) {
            k = kStatic[i - 1][0];
            v = kStatic[i - 1][1];
            return true;
        }
        if (i > 61 && i - 62 < dyn_.size()) {
            k = dyn_[i - 62].first;
            v = dyn_[i - 62].second;
            return true;
        }
        return false;
    };
    auto evict = [&]() {
        while (dyn_size_ > dyn_max_ && !dyn_.empty()) {
            dyn_size_ -= dyn_.back().first.size() + dyn_.back().second.size() + 32;
            dyn_.pop_back();
        }
    };
    Response *r = response((int)id);
    while (p < end) {
        std::string k, v;
        const uint8_t b = *p;
        size_t i;
        if (b & 0x80) {  // indexed
            if (!get_int(p, end, 7, i) || !lookup(i, k, v)) break;
        } else if ((b & 0xe0) == 0x20) {  // table size update
            if (!get_int(p, end, 5, i)) break;
            dyn_max_ = i;
            evict();
            continue;
        } else {
            const bool index = (b & 0xc0) == 0x40;
            if (!get_int(p, end, index ? 6 : 4, i)) break;
            if (i == 0) {
                if (!get_str(p, end, k)) break;
            } else {
                std::string unused;
                if (!lookup(i, k, unused)) break;
            }
            if (!get_str(p, end, v)) break;
            if (index) {
                dyn_.insert(dyn_.begin(), {k, v});
                dyn_size_ += k.size() + v.size() + 32;
                evict();
            }
        }
        if (r && k == ":status") r->status = atoi(v.c_str());
    }
    if (p != end) {
        why = "HTTP/2 headers that don't decode";
        return false;
    }
    return true;
}

}  // namespace ts
