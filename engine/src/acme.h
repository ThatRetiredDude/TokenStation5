// A certificate for a name, from an ACME certificate authority (Let's
// Encrypt), proved with a DNS record (dns-01): what Tailscale's own app does
// for `tailscale cert`, so ts5 serve can answer HTTPS on its tailnet name
// (tailnet_serve.h). The DNS record is put up by the caller (on a tailnet,
// through Tailscale's control server).
//
// The account key and the certificate's key are P-256 (ECDSA). Nothing here
// is logged but what happens, never a key.
#pragma once
#include <functional>
#include <string>

namespace ts5 {

struct AcmeOptions {
    // Let's Encrypt's; TS5_ACME_DIRECTORY gives another (its staging, or a
    // test's).
    std::string directory = "https://acme-v02.api.letsencrypt.org/directory";
    // The account's key, PEM, kept between runs (one account, reused). Empty:
    // a new one is made and handed back in account_key.
    std::string account_key;
    // How long to wait for the DNS record to be seen before asking the
    // authority to check it, at most (it checks itself; a lookup here only
    // keeps it from checking too soon).
    int dns_wait_s = 60;
};

// Puts up the TXT record name = value (name is "_acme-challenge.DOMAIN");
// false with why.
using SetTxt = std::function<bool(const std::string &name, const std::string &value, std::string &why)>;

// Gets a certificate for domain: cert_pem is the chain (the certificate
// first), key_pem its private key. o.account_key is set to the account's
// key, new or not. False with why (the authority's own words when it said
// no).
bool acme_certificate(AcmeOptions &o, const std::string &domain, const SetTxt &set_txt, std::string &cert_pem,
                      std::string &key_pem, std::string &why);

// When a PEM certificate (the first in a chain) stops being valid, in
// seconds since 1970 (UTC); 0 when it can't be read. names_ok: it is for
// domain.
long long cert_expiry(const std::string &cert_pem, const std::string &domain, bool *names_ok = nullptr);

}  // namespace ts5
