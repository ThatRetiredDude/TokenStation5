#include "nacl.h"

#include <cstring>

#include "net.h"
#include "noise.h"

extern "C" {
#include "crypto/refc/poly1305-donna.h"
}

namespace ts {

namespace {

uint32_t rotl(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }
uint32_t le32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
void put_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

const uint8_t kSigma[16] = {'e', 'x', 'p', 'a', 'n', 'd', ' ', '3', '2', '-', 'b', 'y', 't', 'e', ' ', 'k'};

// Salsa20's 20 rounds on x (in place).
void rounds(uint32_t x[16]) {
    for (int i = 0; i < 10; i++) {
#define QR(a, b, c, d) \
    x[b] ^= rotl(x[a] + x[d], 7); x[c] ^= rotl(x[b] + x[a], 9); x[d] ^= rotl(x[c] + x[b], 13); x[a] ^= rotl(x[d] + x[c], 18);
        QR(0, 4, 8, 12) QR(5, 9, 13, 1) QR(10, 14, 2, 6) QR(15, 3, 7, 11)
        QR(0, 1, 2, 3) QR(5, 6, 7, 4) QR(10, 11, 8, 9) QR(15, 12, 13, 14)
#undef QR
    }
}

void setup(uint32_t x[16], const uint8_t key[32], const uint8_t in[16]) {
    x[0] = le32(kSigma);
    x[5] = le32(kSigma + 4);
    x[10] = le32(kSigma + 8);
    x[15] = le32(kSigma + 12);
    for (int i = 0; i < 4; i++) {
        x[1 + i] = le32(key + 4 * i);
        x[11 + i] = le32(key + 16 + 4 * i);
        x[6 + i] = le32(in + 4 * i);
    }
}

// HSalsa20: a subkey from key and 16 bytes of nonce.
void hsalsa20(uint8_t out[32], const uint8_t key[32], const uint8_t in[16]) {
    uint32_t x[16];
    setup(x, key, in);
    rounds(x);
    const int pick[8] = {0, 5, 10, 15, 6, 7, 8, 9};
    for (int i = 0; i < 8; i++) put_le32(out + 4 * i, x[pick[i]]);
}

// XSalsa20's key stream XORed into buf (from block 0 of the stream).
void xsalsa20(uint8_t *buf, size_t n, const uint8_t key[32], const uint8_t nonce[24]) {
    uint8_t sub[32];
    hsalsa20(sub, key, nonce);
    uint8_t in[16] = {};
    memcpy(in, nonce + 16, 8);
    for (uint64_t block = 0; n > 0; block++) {
        for (int i = 0; i < 8; i++) in[8 + i] = (uint8_t)(block >> (8 * i));
        uint32_t x[16], j[16];
        setup(x, sub, in);
        memcpy(j, x, sizeof x);
        rounds(x);
        uint8_t ks[64];
        for (int i = 0; i < 16; i++) put_le32(ks + 4 * i, x[i] + j[i]);
        const size_t k = n < 64 ? n : 64;
        for (size_t i = 0; i < k; i++) buf[i] ^= ks[i];
        buf += k;
        n -= k;
    }
}

bool shared_key(uint8_t k[32], const uint8_t priv[32], const uint8_t pub[32]) {
    uint8_t s[32];
    if (!x25519(s, priv, pub)) return false;
    const uint8_t zero[16] = {};
    hsalsa20(k, s, zero);
    return true;
}

}  // namespace

std::string box_seal(const uint8_t priv[32], const uint8_t pub[32], const std::string &msg) {
    uint8_t k[32], nonce[24];
    if (!shared_key(k, priv, pub) || !ts5::random_bytes(nonce, sizeof nonce)) return "";
    // secretbox: the stream's first 32 bytes are Poly1305's key, the message
    // is XORed with the stream from byte 32.
    std::string stream(32 + msg.size(), '\0');
    memcpy(&stream[32], msg.data(), msg.size());
    xsalsa20((uint8_t *)&stream[0], stream.size(), k, nonce);
    uint8_t tag[16];
    poly1305_context p;
    poly1305_init(&p, (const unsigned char *)stream.data());
    poly1305_update(&p, (const unsigned char *)stream.data() + 32, msg.size());
    poly1305_finish(&p, tag);
    std::string out((const char *)nonce, 24);
    out.append((const char *)tag, 16);
    out.append(stream, 32, std::string::npos);
    return out;
}

bool box_open(const uint8_t priv[32], const uint8_t pub[32], const std::string &sealed, std::string &msg) {
    if (sealed.size() < 40) return false;
    uint8_t k[32];
    if (!shared_key(k, priv, pub)) return false;
    const uint8_t *nonce = (const uint8_t *)sealed.data(), *tag = nonce + 24, *ct = nonce + 40;
    const size_t n = sealed.size() - 40;
    std::string stream(32 + n, '\0');
    memcpy(&stream[32], ct, n);
    xsalsa20((uint8_t *)&stream[0], stream.size(), k, nonce);
    uint8_t want[16];
    poly1305_context p;
    poly1305_init(&p, (const unsigned char *)stream.data());
    poly1305_update(&p, ct, n);
    poly1305_finish(&p, want);
    uint8_t diff = 0;
    for (int i = 0; i < 16; i++) diff |= want[i] ^ tag[i];
    if (diff) return false;
    msg.assign(stream, 32, std::string::npos);
    return true;
}

}  // namespace ts
