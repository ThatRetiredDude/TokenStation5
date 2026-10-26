// NaCl's crypto_box (X25519, XSalsa20, Poly1305), as Tailscale's relays
// (DERP) use it for the client's hello: Go's golang.org/x/crypto/nacl/box.
#pragma once
#include <cstdint>
#include <string>

namespace ts {

// The box from priv to pub: nonce (24 bytes) first, then the sealed message
// (16-byte tag and the ciphertext), as box.Seal(nonce, ...) appends them.
// "" when pub is no good.
std::string box_seal(const uint8_t priv[32], const uint8_t pub[32], const std::string &msg);
// Opens what box_seal made (nonce first). False when it doesn't open.
bool box_open(const uint8_t priv[32], const uint8_t pub[32], const std::string &sealed, std::string &msg);

}  // namespace ts
