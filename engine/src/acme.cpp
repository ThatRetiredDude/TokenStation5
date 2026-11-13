#include "acme.h"

#include <arpa/nameser.h>
#include <netinet/in.h>
#include <resolv.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

#include <mbedtls/asn1write.h>
#include <mbedtls/ecdsa.h>
#include <mbedtls/ecp.h>
#include <mbedtls/oid.h>
#include <mbedtls/pk.h>
#include <mbedtls/sha256.h>
#include <mbedtls/x509_crt.h>
#include <mbedtls/x509_csr.h>

#include "json.h"
#include "log.h"
#include "net.h"

namespace ts5 {

namespace {

int rng(void *, unsigned char *out, size_t n) { return random_bytes(out, n) ? 0 : MBEDTLS_ERR_ECP_RANDOM_FAILED; }

std::string b64url(const unsigned char *p, size_t n) {
    static const char *t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    size_t i = 0;
    for (; i + 2 < n; i += 3) {
        const unsigned v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += t[(v >> 6) & 63];
        out += t[v & 63];
    }
    if (i + 1 == n) {
        const unsigned v = p[i] << 16;
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
    } else if (i + 2 == n) {
        const unsigned v = (p[i] << 16) | (p[i + 1] << 8);
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += t[(v >> 6) & 63];
    }
    return out;
}
std::string b64url(const std::string &s) { return b64url((const unsigned char *)s.data(), s.size()); }

std::string sha256(const std::string &s) {
    unsigned char h[32];
    mbedtls_sha256((const unsigned char *)s.data(), s.size(), h, 0);
    return std::string((const char *)h, 32);
}

std::string err_text(int rc) {
    char b[24];
    snprintf(b, sizeof b, "-0x%04x", (unsigned)-rc);
    return b;
}

// A P-256 key: made, or read from PEM.
struct Key {
    mbedtls_pk_context pk;
    Key() { mbedtls_pk_init(&pk); }
    ~Key() { mbedtls_pk_free(&pk); }
    Key(const Key &) = delete;
    Key &operator=(const Key &) = delete;

    bool make(std::string &why) {
        int rc = mbedtls_pk_setup(&pk, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY));
        if (rc == 0) rc = mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(pk), rng, nullptr);
        if (rc) why = "can't make a key (" + err_text(rc) + ")";
        return rc == 0;
    }
    bool read(const std::string &pem, std::string &why) {
        const int rc = mbedtls_pk_parse_key(&pk, (const unsigned char *)pem.c_str(), pem.size() + 1, nullptr, 0, rng, nullptr);
        if (rc || mbedtls_pk_get_type(&pk) != MBEDTLS_PK_ECKEY ||
            mbedtls_ecp_keypair_get_group_id(mbedtls_pk_ec(pk)) != MBEDTLS_ECP_DP_SECP256R1) {
            why = "the kept ACME account key can't be read";
            return false;
        }
        return true;
    }
    std::string pem() {
        std::vector<unsigned char> b(2048);
        if (mbedtls_pk_write_key_pem(&pk, b.data(), b.size()) != 0) return "";
        return std::string((const char *)b.data());
    }
    // The public key as a JWK, its members in the order RFC 7638 hashes them.
    std::string jwk() {
        unsigned char q[65];
        size_t n = 0;
        if (mbedtls_ecp_write_public_key(mbedtls_pk_ec(pk), MBEDTLS_ECP_PF_UNCOMPRESSED, &n, q, sizeof q) != 0 || n != 65)
            return "";
        return "{\"crv\":\"P-256\",\"kty\":\"EC\",\"x\":\"" + b64url(q + 1, 32) + "\",\"y\":\"" + b64url(q + 33, 32) + "\"}";
    }
    // ES256: the SHA-256 of data signed, r and s 32 bytes each.
    bool sign(const std::string &data, std::string &sig) {
        const std::string h = sha256(data);
        unsigned char der[MBEDTLS_ECDSA_MAX_LEN];
        size_t n = 0;
        if (mbedtls_pk_sign(&pk, MBEDTLS_MD_SHA256, (const unsigned char *)h.data(), 32, der, sizeof der, &n, rng, nullptr))
            return false;
        // SEQUENCE { INTEGER r, INTEGER s }
        unsigned char *p = der, *end = der + n;
        size_t len;
        if (mbedtls_asn1_get_tag(&p, end, &len, MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE)) return false;
        sig.clear();
        for (int i = 0; i < 2; i++) {
            if (mbedtls_asn1_get_tag(&p, end, &len, MBEDTLS_ASN1_INTEGER)) return false;
            while (len > 32 && *p == 0) {
                p++;
                len--;
            }
            if (len > 32) return false;
            sig += std::string(32 - len, '\0') + std::string((const char *)p, len);
            p += len;
        }
        return true;
    }
};

// An ACME exchange: the directory, a nonce kept from each answer, the account.
struct Acme {
    std::string new_nonce, new_account, new_order;
    std::string nonce, kid;
    Key *key = nullptr;
    std::string jwk;

    static std::string header(const std::string &head, const std::string &name) {
        std::string lower = head;
        for (char &c : lower) c = (char)tolower((unsigned char)c);
        std::string want = "\n" + name + ":";
        for (char &c : want) c = (char)tolower((unsigned char)c);
        const size_t at = lower.find(want);
        if (at == std::string::npos) return "";
        size_t b = at + want.size();
        const size_t e = head.find('\n', b);
        std::string v = head.substr(b, e == std::string::npos ? std::string::npos : e - b);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
        while (!v.empty() && (v.back() == '\r' || v.back() == ' ')) v.pop_back();
        return v;
    }

    // The authority's words for a refusal: its problem document.
    static std::string problem(const HttpResponse &r) {
        Json j;
        if (Json::parse(r.body, j) && j.type == Json::OBJ) {
            std::string d = j["detail"].str();
            const std::string t = j["type"].str();
            if (!t.empty()) d += (d.empty() ? "" : " (") + t.substr(t.rfind(':') + 1) + (d.empty() ? "" : ")");
            if (!d.empty()) return d;
        }
        return "HTTP " + std::to_string(r.status);
    }

    bool get_nonce(std::string &why) {
        HttpResponse r;
        std::string err;
        if (!http_request("GET", new_nonce, {}, "", r, err, 30)) {
            why = "the certificate authority doesn't answer: " + err;
            return false;
        }
        nonce = header(r.head, "Replay-Nonce");
        if (nonce.empty()) why = "the certificate authority gave no nonce (HTTP " + std::to_string(r.status) + ")";
        return !nonce.empty();
    }

    // A signed POST (payload "" for POST-as-GET); a stale nonce is tried once more.
    bool post(const std::string &url, const std::string &payload, HttpResponse &r, std::string &why, bool with_jwk = false,
              const std::string &accept = "") {
        for (int attempt = 0; attempt < 2; attempt++) {
            if (nonce.empty() && !get_nonce(why)) return false;
            std::string prot = "{\"alg\":\"ES256\",";
            prot += with_jwk ? "\"jwk\":" + jwk : "\"kid\":" + json_escape(kid);
            prot += ",\"nonce\":" + json_escape(nonce) + ",\"url\":" + json_escape(url) + "}";
            const std::string p64 = b64url(prot), b64 = b64url(payload);
            std::string sig;
            if (!key->sign(p64 + "." + b64, sig)) {
                why = "can't sign the request";
                return false;
            }
            const std::string body = "{\"protected\":\"" + p64 + "\",\"payload\":\"" + b64 + "\",\"signature\":\"" +
                                     b64url(sig) + "\"}";
            std::vector<std::string> h = {"Content-Type: application/jose+json"};
            if (!accept.empty()) h.push_back("Accept: " + accept);
            std::string err;
            nonce.clear();
            if (!http_request("POST", url, h, body, r, err, 60)) {
                why = "the certificate authority doesn't answer: " + err;
                return false;
            }
            nonce = header(r.head, "Replay-Nonce");
            if (r.status == 400 && r.body.find("badNonce") != std::string::npos && attempt == 0) continue;
            if (r.status >= 400) {
                why = problem(r);
                return false;
            }
            return true;
        }
        return false;
    }

    // POST-as-GET until the object's status is no longer pending/processing.
    bool wait_status(const std::string &url, Json &j, std::string &why, int max_s) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(max_s);
        while (true) {
            HttpResponse r;
            if (!post(url, "", r, why)) return false;
            if (!Json::parse(r.body, j)) {
                why = "the certificate authority's answer isn't JSON";
                return false;
            }
            const std::string st = j["status"].str();
            if (st != "pending" && st != "processing") return true;
            if (std::chrono::steady_clock::now() > until) {
                why = "the certificate authority is still checking after " + std::to_string(max_s) + " s";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
};

// Whether the public DNS has name's TXT record value yet.
bool txt_seen(const std::string &name, const std::string &value) {
    unsigned char ans[4096];
    const int n = res_query(name.c_str(), ns_c_in, ns_t_txt, ans, sizeof ans);
    if (n <= 0) return false;
    ns_msg msg;
    if (ns_initparse(ans, n, &msg) != 0) return false;
    for (int i = 0; i < ns_msg_count(msg, ns_s_an); i++) {
        ns_rr rr;
        if (ns_parserr(&msg, ns_s_an, i, &rr) != 0 || ns_rr_type(rr) != ns_t_txt) continue;
        const unsigned char *d = ns_rr_rdata(rr);
        const int len = ns_rr_rdlen(rr);
        std::string txt;
        for (int at = 0; at < len;) {
            const int l = d[at];
            txt.append((const char *)d + at + 1, std::min(l, len - at - 1));
            at += 1 + l;
        }
        if (txt == value) return true;
    }
    return false;
}

// A certificate request for domain, DER, with the key's public half.
bool make_csr(Key &k, const std::string &domain, std::string &der, std::string &why) {
    mbedtls_x509write_csr csr;
    mbedtls_x509write_csr_init(&csr);
    mbedtls_x509write_csr_set_md_alg(&csr, MBEDTLS_MD_SHA256);
    mbedtls_x509write_csr_set_key(&csr, &k.pk);
    int rc = mbedtls_x509write_csr_set_subject_name(&csr, ("CN=" + domain).c_str());
    // The name the certificate is for, as a subjectAltName.
    mbedtls_x509_san_list san{};
    san.node.type = MBEDTLS_X509_SAN_DNS_NAME;
    san.node.san.unstructured_name.tag = MBEDTLS_ASN1_CONTEXT_SPECIFIC | MBEDTLS_X509_SAN_DNS_NAME;
    san.node.san.unstructured_name.p = (unsigned char *)domain.data();
    san.node.san.unstructured_name.len = domain.size();
    if (rc == 0) rc = mbedtls_x509write_csr_set_subject_alternative_name(&csr, &san);
    unsigned char buf[4096];
    int n = rc == 0 ? mbedtls_x509write_csr_der(&csr, buf, sizeof buf, rng, nullptr) : rc;
    mbedtls_x509write_csr_free(&csr);
    if (n <= 0) {
        why = "can't make the certificate request (" + err_text(n) + ")";
        return false;
    }
    der.assign((const char *)buf + sizeof buf - n, (size_t)n);
    return true;
}

}  // namespace

bool acme_certificate(AcmeOptions &o, const std::string &domain, const SetTxt &set_txt, std::string &cert_pem,
                      std::string &key_pem, std::string &why) {
    Acme a;
    Key account;
    if (o.account_key.empty()) {
        if (!account.make(why)) return false;
        o.account_key = account.pem();
    } else if (!account.read(o.account_key, why)) {
        return false;
    }
    a.key = &account;
    a.jwk = account.jwk();
    if (a.jwk.empty()) {
        why = "can't read the account key";
        return false;
    }

    // The directory.
    {
        HttpResponse r;
        std::string err;
        Json d;
        if (!http_request("GET", o.directory, {}, "", r, err, 30) || r.status != 200 || !Json::parse(r.body, d)) {
            why = "the certificate authority's directory (" + o.directory + ") doesn't answer: " +
                  (err.empty() ? "HTTP " + std::to_string(r.status) : err);
            return false;
        }
        a.new_nonce = d["newNonce"].str();
        a.new_account = d["newAccount"].str();
        a.new_order = d["newOrder"].str();
        if (a.new_nonce.empty() || a.new_account.empty() || a.new_order.empty()) {
            why = "the certificate authority's directory lacks newNonce, newAccount or newOrder";
            return false;
        }
    }
    // The account: made the first time, found by its key after.
    {
        HttpResponse r;
        if (!a.post(a.new_account, "{\"termsOfServiceAgreed\":true}", r, why, true)) {
            why = "the certificate authority refused the account: " + why;
            return false;
        }
        a.kid = Acme::header(r.head, "Location");
        if (a.kid.empty()) {
            why = "the certificate authority gave the account no address";
            return false;
        }
    }
    // The order.
    std::string order_url;
    Json order;
    {
        HttpResponse r;
        if (!a.post(a.new_order, "{\"identifiers\":[{\"type\":\"dns\",\"value\":" + json_escape(domain) + "}]}", r, why)) {
            why = "the certificate authority refused the order for " + domain + ": " + why;
            return false;
        }
        order_url = Acme::header(r.head, "Location");
        if (!Json::parse(r.body, order) || order_url.empty()) {
            why = "the certificate authority's order makes no sense";
            return false;
        }
    }
    // Each authorization: the TXT record, then the authority checks it.
    for (const Json &au : order["authorizations"].a) {
        Json authz;
        HttpResponse r;
        if (!a.post(au.str(), "", r, why) || !Json::parse(r.body, authz)) {
            why = "can't read the authorization: " + why;
            return false;
        }
        if (authz["status"].str() == "valid") continue;
        std::string token, chal_url;
        for (const Json &c : authz["challenges"].a)
            if (c["type"].str() == "dns-01") {
                token = c["token"].str();
                chal_url = c["url"].str();
            }
        if (token.empty() || chal_url.empty()) {
            why = "the certificate authority offers no DNS check (dns-01) for " + domain;
            return false;
        }
        const std::string thumb = b64url(sha256(a.jwk));
        const std::string value = b64url(sha256(token + "." + thumb));
        const std::string name = "_acme-challenge." + domain;
        if (!set_txt(name, value, why)) {
            why = "can't put up the DNS record for the check: " + why;
            return false;
        }
        log_line("ts5: https: DNS record for the check is up; waiting until it can be seen (at most %d s)", o.dns_wait_s);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(o.dns_wait_s);
        bool seen = false;
        while (o.dns_wait_s > 0 && !(seen = txt_seen(name, value)) && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::seconds(3));
        if (o.dns_wait_s > 0 && !seen) log_line("ts5: https: the record isn't seen here yet; asking for the check anyway");
        if (!a.post(chal_url, "{}", r, why)) {
            why = "the certificate authority refused to check: " + why;
            return false;
        }
        Json done;
        if (!a.wait_status(au.str(), done, why, 120)) return false;
        if (done["status"].str() != "valid") {
            std::string detail;
            for (const Json &c : done["challenges"].a)
                if (c["type"].str() == "dns-01" && c["error"].type == Json::OBJ) detail = c["error"]["detail"].str();
            why = "the certificate authority's check of " + domain + " failed" + (detail.empty() ? "" : ": " + detail);
            return false;
        }
    }
    // The certificate's own key, the request, and the certificate.
    Key cert_key;
    if (!cert_key.make(why)) return false;
    std::string csr;
    if (!make_csr(cert_key, domain, csr, why)) return false;
    {
        HttpResponse r;
        if (!a.post(order["finalize"].str(), "{\"csr\":\"" + b64url(csr) + "\"}", r, why)) {
            why = "the certificate authority refused the certificate request: " + why;
            return false;
        }
    }
    Json fin;
    if (!a.wait_status(order_url, fin, why, 120)) return false;
    if (fin["status"].str() != "valid" || fin["certificate"].str().empty()) {
        why = "the certificate authority didn't issue the certificate (order " + fin["status"].str() + ")";
        return false;
    }
    {
        HttpResponse r;
        if (!a.post(fin["certificate"].str(), "", r, why, false, "application/pem-certificate-chain")) {
            why = "can't download the certificate: " + why;
            return false;
        }
        if (r.body.find("-----BEGIN CERTIFICATE-----") == std::string::npos) {
            why = "the certificate authority sent no certificate";
            return false;
        }
        cert_pem = r.body;
    }
    key_pem = cert_key.pem();
    if (key_pem.empty()) {
        why = "can't write the certificate's key";
        return false;
    }
    return true;
}

long long cert_expiry(const std::string &cert_pem, const std::string &domain, bool *names_ok) {
    if (names_ok) *names_ok = false;
    mbedtls_x509_crt crt;
    mbedtls_x509_crt_init(&crt);
    long long t = 0;
    if (mbedtls_x509_crt_parse(&crt, (const unsigned char *)cert_pem.c_str(), cert_pem.size() + 1) == 0) {
        const mbedtls_x509_time &v = crt.valid_to;
        struct tm tm {};
        tm.tm_year = v.year - 1900;
        tm.tm_mon = v.mon - 1;
        tm.tm_mday = v.day;
        tm.tm_hour = v.hour;
        tm.tm_min = v.min;
        tm.tm_sec = v.sec;
        t = (long long)timegm(&tm);
        if (names_ok) {
            for (const mbedtls_x509_sequence *s = &crt.subject_alt_names; s && s->buf.p; s = s->next) {
                mbedtls_x509_subject_alternative_name san;
                if (mbedtls_x509_parse_subject_alt_name(&s->buf, &san) == 0) {
                    if (san.type == MBEDTLS_X509_SAN_DNS_NAME &&
                        std::string((const char *)san.san.unstructured_name.p, san.san.unstructured_name.len) == domain)
                        *names_ok = true;
                    mbedtls_x509_free_subject_alt_name(&san);
                }
            }
        }
    }
    mbedtls_x509_crt_free(&crt);
    return t;
}

}  // namespace ts5
