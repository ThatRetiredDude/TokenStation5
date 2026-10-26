#include "noise.h"

#include <cstring>

#include "net.h"

extern "C" {
#include "crypto/refc/blake2s.h"
#include "crypto/refc/chacha20poly1305.h"
#include "crypto/refc/x25519.h"
}

namespace ts {

namespace {

const char kProtocol[] = "Noise_IK_25519_ChaChaPoly_BLAKE2s";
const char kPrologue[] = "Tailscale Control Protocol v";
constexpr size_t kTag = 16;
constexpr size_t kMaxRecord = 4096;  // header and ciphertext
constexpr size_t kMaxPlain = kMaxRecord - 3 - kTag;

void wipe(void *p, size_t n) {
    volatile uint8_t *b = (volatile uint8_t *)p;
    while (n--) *b++ = 0;
}

void hmac(uint8_t out[32], const uint8_t *key, size_t key_len, const uint8_t *data, size_t len) {
    uint8_t k[64] = {}, pad[64];
    if (key_len > 64) blake2s(k, 32, nullptr, 0, key, key_len);
    else memcpy(k, key, key_len);
    uint8_t inner[32];
    blake2s_ctx c;
    for (int i = 0; i < 64; i++) pad[i] = k[i] ^ 0x36;
    blake2s_init(&c, 32, nullptr, 0);
    blake2s_update(&c, pad, 64);
    blake2s_update(&c, data, len);
    blake2s_final(&c, inner);
    for (int i = 0; i < 64; i++) pad[i] = k[i] ^ 0x5c;
    blake2s_init(&c, 32, nullptr, 0);
    blake2s_update(&c, pad, 64);
    blake2s_update(&c, inner, 32);
    blake2s_final(&c, out);
    wipe(k, sizeof k);
    wipe(inner, sizeof inner);
}

// HKDF with BLAKE2s: two 32-byte outputs from chaining key ck and input.
void hkdf2(uint8_t out1[32], uint8_t out2[32], const uint8_t ck[32], const uint8_t *input, size_t len) {
    uint8_t prk[32], t[33];
    hmac(prk, ck, 32, input, len);
    t[0] = 1;
    hmac(out1, prk, 32, t, 1);
    memcpy(t, out1, 32);
    t[32] = 2;
    hmac(out2, prk, 32, t, 33);
    wipe(prk, sizeof prk);
    wipe(t, sizeof t);
}

// ChaCha20-Poly1305 with Noise's nonce: 4 zero bytes and a big-endian
// counter. WireGuard's code takes the counter's 8 bytes little-endian, so
// it gets the counter byte-swapped.
void seal_with(const uint8_t key[32], uint64_t n, const uint8_t *ad, size_t ad_len, const uint8_t *in, size_t len,
               uint8_t *out) {
    chacha20poly1305_encrypt(out, in, len, ad, ad_len, __builtin_bswap64(n), key);
}

bool open_with(const uint8_t key[32], uint64_t n, const uint8_t *ad, size_t ad_len, const uint8_t *in, size_t len,
               uint8_t *out) {
    return len >= kTag && chacha20poly1305_decrypt(out, in, len, ad, ad_len, __builtin_bswap64(n), key);
}

}  // namespace

bool new_private_key(uint8_t priv[kKeyLen]) {
    if (!ts5::random_bytes(priv, kKeyLen)) return false;
    priv[0] &= 248;
    priv[31] = (uint8_t)((priv[31] & 127) | 64);
    return true;
}

void public_key(uint8_t pub[kKeyLen], const uint8_t priv[kKeyLen]) { x25519_base(pub, priv, 1); }

bool x25519(uint8_t out[kKeyLen], const uint8_t priv[kKeyLen], const uint8_t pub[kKeyLen]) {
    ::x25519(out, priv, pub, 1);
    uint8_t any = 0;
    for (int i = 0; i < kKeyLen; i++) any |= out[i];
    return any != 0;
}

std::string key_text(const char *prefix, const uint8_t key[kKeyLen]) {
    static const char hex[] = "0123456789abcdef";
    std::string s = prefix;
    for (int i = 0; i < kKeyLen; i++) {
        s += hex[key[i] >> 4];
        s += hex[key[i] & 15];
    }
    return s;
}

bool parse_key(const std::string &text, const char *prefix, uint8_t key[kKeyLen]) {
    const size_t p = strlen(prefix);
    if (text.size() != p + 2 * kKeyLen || text.compare(0, p, prefix) != 0) return false;
    for (int i = 0; i < kKeyLen; i++) {
        int v = 0;
        for (int j = 0; j < 2; j++) {
            const char c = text[p + 2 * i + j];
            const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (d < 0) return false;
            v = v * 16 + d;
        }
        key[i] = (uint8_t)v;
    }
    return true;
}

Noise::~Noise() {
    wipe(machine_priv_, sizeof machine_priv_);
    wipe(eph_priv_, sizeof eph_priv_);
    wipe(tx_key_, sizeof tx_key_);
    wipe(rx_key_, sizeof rx_key_);
    wipe(ck_, sizeof ck_);
}

void Noise::mix_hash(const void *data, size_t len) {
    blake2s_ctx c;
    blake2s_init(&c, 32, nullptr, 0);
    blake2s_update(&c, h_, 32);
    blake2s_update(&c, data, len);
    blake2s_final(&c, h_);
}

bool Noise::mix_dh(const uint8_t priv[kKeyLen], const uint8_t pub[kKeyLen], uint8_t key[32]) {
    uint8_t shared[32];
    const bool ok = x25519(shared, priv, pub);
    hkdf2(ck_, key, ck_, shared, 32);
    wipe(shared, sizeof shared);
    return ok;
}

std::string Noise::initiation(const uint8_t machine_priv[kKeyLen], const uint8_t control_pub[kKeyLen], uint16_t version) {
    ready_ = false;
    in_.clear();
    tx_n_ = rx_n_ = 0;
    memcpy(machine_priv_, machine_priv, kKeyLen);
    blake2s(h_, 32, nullptr, 0, kProtocol, strlen(kProtocol));
    memcpy(ck_, h_, 32);
    const std::string prologue = kPrologue + std::to_string(version);
    mix_hash(prologue.data(), prologue.size());
    mix_hash(control_pub, kKeyLen);

    uint8_t msg[101] = {};
    msg[0] = (uint8_t)(version >> 8);
    msg[1] = (uint8_t)version;
    msg[2] = 1;  // initiation
    msg[3] = 0;
    msg[4] = 96;  // payload: ephemeral key, sealed machine key, sealed empty payload
    new_private_key(eph_priv_);
    uint8_t *eph_pub = msg + 5, *sealed_machine = msg + 5 + 32, *tag = msg + 5 + 32 + 48;
    public_key(eph_pub, eph_priv_);
    mix_hash(eph_pub, kKeyLen);
    uint8_t k[32];
    mix_dh(eph_priv_, control_pub, k);  // es
    uint8_t machine_pub[kKeyLen];
    public_key(machine_pub, machine_priv_);
    seal_with(k, 0, h_, 32, machine_pub, kKeyLen, sealed_machine);
    mix_hash(sealed_machine, 48);
    mix_dh(machine_priv_, control_pub, k);  // ss
    seal_with(k, 0, h_, 32, nullptr, 0, tag);
    mix_hash(tag, kTag);
    wipe(k, sizeof k);
    return std::string((const char *)msg, sizeof msg);
}

bool Noise::finish(const std::string &r, std::string &why) {
    if (r.size() >= 3 && (uint8_t)r[0] == 3) {
        why = "the control server refused: " + r.substr(3);
        return false;
    }
    if (r.size() != 51 || r[0] != 2 || (uint8_t)r[1] != 0 || (uint8_t)r[2] != 48) {
        why = "not a Tailscale control server answer";
        return false;
    }
    const uint8_t *eph = (const uint8_t *)r.data() + 3, *tag = eph + 32;
    mix_hash(eph, 32);
    uint8_t k[32];
    mix_dh(eph_priv_, eph, k);  // ee
    mix_dh(machine_priv_, eph, k);  // se
    uint8_t none[1];
    if (!open_with(k, 0, h_, 32, tag, kTag, none)) {
        why = "the control server's key isn't the one it said";
        return false;
    }
    mix_hash(tag, kTag);
    hkdf2(tx_key_, rx_key_, ck_, nullptr, 0);
    wipe(k, sizeof k);
    wipe(eph_priv_, sizeof eph_priv_);
    ready_ = true;
    return true;
}

std::string Noise::seal(const void *data, size_t len) {
    std::string out;
    const uint8_t *p = (const uint8_t *)data;
    while (len > 0) {
        const size_t n = len < kMaxPlain ? len : kMaxPlain;
        uint8_t rec[kMaxRecord];
        rec[0] = 4;  // record
        rec[1] = (uint8_t)((n + kTag) >> 8);
        rec[2] = (uint8_t)(n + kTag);
        seal_with(tx_key_, tx_n_++, nullptr, 0, p, n, rec + 3);
        out.append((const char *)rec, 3 + n + kTag);
        p += n;
        len -= n;
    }
    return out;
}

bool Noise::open(const void *data, size_t len, std::string &out, std::string &why) {
    in_.append((const char *)data, len);
    size_t at = 0;
    while (in_.size() - at >= 3) {
        const uint8_t *h = (const uint8_t *)in_.data() + at;
        const size_t n = ((size_t)h[1] << 8) | h[2];
        if (h[0] != 4) {
            why = h[0] == 3 ? "the control server sent an error: " + in_.substr(at + 3, n) : "a record of unknown type";
            return false;
        }
        if (n < kTag || 3 + n > kMaxRecord) {
            why = "a record of a wrong size";
            return false;
        }
        if (in_.size() - at < 3 + n) break;
        uint8_t plain[kMaxRecord];
        if (!open_with(rx_key_, rx_n_++, nullptr, 0, h + 3, n, plain)) {
            why = "a record didn't decrypt";
            return false;
        }
        out.append((const char *)plain, n - kTag);
        at += 3 + n;
    }
    in_.erase(0, at);
    return true;
}

}  // namespace ts
