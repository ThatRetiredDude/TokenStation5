// Tailscale's control protocol's encryption ("ts2021"): Noise IK
// (Noise_IK_25519_ChaChaPoly_BLAKE2s) from the PS5's machine key to the
// control server's, then records of at most 4096 bytes each way. Written from
// Tailscale's control/controlbase, which it must match byte for byte.
#pragma once
#include <cstdint>
#include <string>

namespace ts {

constexpr int kKeyLen = 32;

// X25519: a new private key (clamped) and the public key of a private one.
bool new_private_key(uint8_t priv[kKeyLen]);
void public_key(uint8_t pub[kKeyLen], const uint8_t priv[kKeyLen]);
// The shared secret; false when it comes out all zeros (a bad public key).
bool x25519(uint8_t out[kKeyLen], const uint8_t priv[kKeyLen], const uint8_t pub[kKeyLen]);

// Keys as Tailscale writes them: "mkey:", "nodekey:" or "discokey:" and hex.
std::string key_text(const char *prefix, const uint8_t key[kKeyLen]);
bool parse_key(const std::string &text, const char *prefix, uint8_t key[kKeyLen]);

class Noise {
public:
    ~Noise();
    // The handshake's first message (101 bytes), for the X-Tailscale-Handshake
    // header. version: the protocol version the client speaks.
    std::string initiation(const uint8_t machine_priv[kKeyLen], const uint8_t control_pub[kKeyLen], uint16_t version);
    // The server's answer (51 bytes; an error is type 3, its length, its
    // text). False, with why, when it isn't a good one.
    bool finish(const std::string &response, std::string &why);
    bool ready() const { return ready_; }

    // Plaintext into records to send.
    std::string seal(const void *data, size_t len);
    // Bytes from the server: whole records are opened and their plaintext
    // appended to out; a part of one waits for the rest. False when a
    // record doesn't open (the connection is then no good).
    bool open(const void *data, size_t len, std::string &out, std::string &why);

private:
    void mix_hash(const void *data, size_t len);
    bool mix_dh(const uint8_t priv[kKeyLen], const uint8_t pub[kKeyLen], uint8_t key[kKeyLen]);
    uint8_t h_[32] = {}, ck_[32] = {};
    uint8_t machine_priv_[kKeyLen] = {}, eph_priv_[kKeyLen] = {};
    uint8_t tx_key_[32] = {}, rx_key_[32] = {};
    uint64_t tx_n_ = 0, rx_n_ = 0;
    bool ready_ = false;
    std::string in_;  // a record not all here yet
};

}  // namespace ts
