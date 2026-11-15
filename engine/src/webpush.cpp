#include "webpush.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <vector>

#include <mbedtls/asn1.h>
#include <mbedtls/ecdh.h>
#include <mbedtls/ecdsa.h>
#include <mbedtls/ecp.h>
#include <mbedtls/gcm.h>
#include <mbedtls/md.h>
#include <mbedtls/pk.h>
#include <mbedtls/sha256.h>

#include "json.h"
#include "net.h"

namespace ts5 {

namespace {

int rng(void *, unsigned char *out, size_t n) { return random_bytes(out, n) ? 0 : MBEDTLS_ERR_ECP_RANDOM_FAILED; }

std::string hmac(const std::string &key, const std::string &data) {
    unsigned char out[32];
    mbedtls_md_hmac(mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), (const unsigned char *)key.data(), key.size(),
                    (const unsigned char *)data.data(), data.size(), out);
    return std::string((const char *)out, 32);
}

std::string err_text(int rc) {
    char b[24];
    snprintf(b, sizeof b, "-0x%04x", (unsigned)-rc);
    return b;
}

std::string read_file(const std::string &path) {
    std::string out;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return out;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) out.append(b, n);
    fclose(f);
    return out;
}

// The key in path, read into pk; made (and kept, 0600) when there is none.
bool load_key(const std::string &path, mbedtls_pk_context &pk, std::string &why) {
    const std::string pem = read_file(path);
    if (!pem.empty()) {
        if (mbedtls_pk_parse_key(&pk, (const unsigned char *)pem.c_str(), pem.size() + 1, nullptr, 0, rng, nullptr) == 0 &&
            mbedtls_pk_get_type(&pk) == MBEDTLS_PK_ECKEY &&
            mbedtls_ecp_keypair_get_group_id(mbedtls_pk_ec(pk)) == MBEDTLS_ECP_DP_SECP256R1)
            return true;
        why = "the notifications key (" + path + ") can't be read";
        return false;
    }
    int rc = mbedtls_pk_setup(&pk, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY));
    if (rc == 0) rc = mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(pk), rng, nullptr);
    std::vector<unsigned char> b(2048);
    if (rc == 0) rc = mbedtls_pk_write_key_pem(&pk, b.data(), b.size());
    if (rc) {
        why = "can't make the notifications key (" + err_text(rc) + ")";
        return false;
    }
    const std::string out((const char *)b.data());
    const std::string tmp = path + ".new";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    bool ok = fd >= 0 && write(fd, out.data(), out.size()) == (ssize_t)out.size();
    if (fd >= 0) ok = close(fd) == 0 && ok;
    // Another request may have made one meanwhile: the first one kept wins.
    if (ok && link(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        mbedtls_pk_free(&pk);
        mbedtls_pk_init(&pk);
        return load_key(path, pk, why);
    }
    unlink(tmp.c_str());
    if (!ok) why = "can't keep the notifications key in " + path;
    return ok;
}

std::string public_raw(mbedtls_pk_context &pk) {
    unsigned char q[65];
    size_t n = 0;
    if (mbedtls_ecp_write_public_key(mbedtls_pk_ec(pk), MBEDTLS_ECP_PF_UNCOMPRESSED, &n, q, sizeof q) != 0 || n != 65)
        return "";
    return std::string((const char *)q, 65);
}

// ES256: data's SHA-256 signed, r and s 32 bytes each.
bool sign(mbedtls_pk_context &pk, const std::string &data, std::string &sig) {
    unsigned char h[32];
    mbedtls_sha256((const unsigned char *)data.data(), data.size(), h, 0);
    unsigned char der[MBEDTLS_ECDSA_MAX_LEN];
    size_t n = 0;
    if (mbedtls_pk_sign(&pk, MBEDTLS_MD_SHA256, h, 32, der, sizeof der, &n, rng, nullptr)) return false;
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

// scheme://host[:port] of a URL.
std::string origin_of(const std::string &url) {
    const size_t s = url.find("://");
    if (s == std::string::npos) return "";
    const size_t p = url.find('/', s + 3);
    return url.substr(0, p);
}

// Who runs this sender, for the push service (RFC 8292's "sub"): a URL or a
// mailto:. Apple's refuses a request without one. The project's page, never
// anything of the owner's; TS5_PUSH_SUBJECT gives another.
std::string subject() {
    const char *e = getenv("TS5_PUSH_SUBJECT");
    return e && *e ? e : "https://github.com/ThatRetiredDude/TokenStation5";
}

}  // namespace

std::string b64url_encode(const std::string &bytes) {
    static const char *t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    const unsigned char *p = (const unsigned char *)bytes.data();
    const size_t n = bytes.size();
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

bool b64url_decode(const std::string &text, std::string &bytes) {
    bytes.clear();
    unsigned v = 0;
    int bits = 0;
    size_t i = 0;
    for (; i < text.size(); i++) {
        const char c = text[i];
        int d;
        if (c >= 'A' && c <= 'Z') d = c - 'A';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
        else if (c >= '0' && c <= '9') d = c - '0' + 52;
        else if (c == '-' || c == '+') d = 62;
        else if (c == '_' || c == '/') d = 63;
        else break;
        v = (v << 6) | (unsigned)d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            bytes += (char)((v >> bits) & 0xff);
        }
    }
    for (; i < text.size(); i++)
        if (text[i] != '=') return false;
    return true;
}

bool vapid_key(const std::string &path, std::string &public_b64, std::string &why) {
    mbedtls_pk_context pk;
    mbedtls_pk_init(&pk);
    const bool ok = load_key(path, pk, why);
    const std::string q = ok ? public_raw(pk) : "";
    mbedtls_pk_free(&pk);
    if (ok && q.empty()) why = "the notifications key has no public half";
    public_b64 = b64url_encode(q);
    return ok && !q.empty();
}

bool webpush_encrypt(const std::string &p256dh, const std::string &auth, const std::string &plain, std::string &body,
                     std::string &why, const std::string &as_private, const std::string &salt_in) {
    if (p256dh.size() != 65 || p256dh[0] != 4 || auth.size() != 16) {
        why = "the browser's keys aren't a P-256 key and a 16-byte secret";
        return false;
    }
    mbedtls_ecp_group grp;
    mbedtls_mpi d, z;
    mbedtls_ecp_point q, ua;
    mbedtls_ecp_group_init(&grp);
    mbedtls_mpi_init(&d);
    mbedtls_mpi_init(&z);
    mbedtls_ecp_point_init(&q);
    mbedtls_ecp_point_init(&ua);
    unsigned char as_pub[65], secret[32];
    size_t n = 0;
    int rc = mbedtls_ecp_group_load(&grp, MBEDTLS_ECP_DP_SECP256R1);
    if (rc == 0) rc = mbedtls_ecp_point_read_binary(&grp, &ua, (const unsigned char *)p256dh.data(), 65);
    if (rc == 0) rc = mbedtls_ecp_check_pubkey(&grp, &ua);
    // The sender's key for this message alone (or a test's).
    if (rc == 0 && as_private.size() == 32) {
        rc = mbedtls_mpi_read_binary(&d, (const unsigned char *)as_private.data(), 32);
        if (rc == 0) rc = mbedtls_ecp_mul(&grp, &q, &d, &grp.G, rng, nullptr);
    } else if (rc == 0) {
        rc = mbedtls_ecp_gen_keypair(&grp, &d, &q, rng, nullptr);
    }
    if (rc == 0) rc = mbedtls_ecp_point_write_binary(&grp, &q, MBEDTLS_ECP_PF_UNCOMPRESSED, &n, as_pub, sizeof as_pub);
    if (rc == 0) rc = mbedtls_ecdh_compute_shared(&grp, &z, &ua, &d, rng, nullptr);
    if (rc == 0) rc = mbedtls_mpi_write_binary(&z, secret, 32);
    mbedtls_ecp_group_free(&grp);
    mbedtls_mpi_free(&d);
    mbedtls_mpi_free(&z);
    mbedtls_ecp_point_free(&q);
    mbedtls_ecp_point_free(&ua);
    if (rc || n != 65) {
        why = "can't agree a key with the browser's (" + err_text(rc) + ")";
        return false;
    }
    std::string salt = salt_in;
    if (salt.size() != 16) {
        unsigned char s[16];
        if (!random_bytes(s, sizeof s)) {
            why = "no random bytes";
            return false;
        }
        salt.assign((const char *)s, 16);
    }
    const std::string as_public((const char *)as_pub, 65);
    // RFC 8291, 3.3 and 3.4: HKDF steps, each one HMAC block.
    const std::string prk_key = hmac(auth, std::string((const char *)secret, 32));
    const std::string key_info = std::string("WebPush: info") + '\0' + p256dh + as_public;
    const std::string ikm = hmac(prk_key, key_info + '\x01');
    const std::string prk = hmac(salt, ikm);
    const std::string cek = hmac(prk, std::string("Content-Encoding: aes128gcm") + '\0' + '\x01').substr(0, 16);
    const std::string nonce = hmac(prk, std::string("Content-Encoding: nonce") + '\0' + '\x01').substr(0, 12);
    // One record: the text, then 0x02 (the last record's delimiter).
    const std::string padded = plain + '\x02';
    std::string out(padded.size(), '\0');
    unsigned char tag[16];
    mbedtls_gcm_context gcm;
    mbedtls_gcm_init(&gcm);
    rc = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, (const unsigned char *)cek.data(), 128);
    if (rc == 0)
        rc = mbedtls_gcm_crypt_and_tag(&gcm, MBEDTLS_GCM_ENCRYPT, padded.size(), (const unsigned char *)nonce.data(), 12,
                                       nullptr, 0, (const unsigned char *)padded.data(), (unsigned char *)&out[0], 16, tag);
    mbedtls_gcm_free(&gcm);
    if (rc) {
        why = "can't encrypt (" + err_text(rc) + ")";
        return false;
    }
    // The header: salt, the record size (4096), the key id's length and the
    // sender's public key (its id).
    body = salt + std::string("\x00\x00\x10\x00", 4) + '\x41' + as_public + out + std::string((const char *)tag, 16);
    return true;
}

bool push_endpoint_ok(const std::string &endpoint) {
    const std::string origin = origin_of(endpoint);
    const size_t s = origin.find("://");
    if (s == std::string::npos || endpoint.size() > 2048) return false;
    const std::string scheme = origin.substr(0, s), hostport = origin.substr(s + 3);
    for (char c : hostport)
        if (!isalnum((unsigned char)c) && c != '.' && c != '-' && c != ':') return false;
    if (const char *e = getenv("TS5_PUSH_HOSTS"); e && *e) {
        const std::string list = std::string(",") + e + ",";
        if (list.find("," + hostport + ",") != std::string::npos) return scheme == "https" || scheme == "http";
    }
    if (scheme != "https" || hostport.find(':') != std::string::npos) return false;
    auto ends = [&](const std::string &tail) {
        return hostport == tail.substr(1) ||
               (hostport.size() > tail.size() && hostport.compare(hostport.size() - tail.size(), tail.size(), tail) == 0);
    };
    return ends(".push.apple.com") || hostport == "fcm.googleapis.com" || hostport == "android.googleapis.com" ||
           ends(".push.services.mozilla.com") || ends(".notify.windows.com");
}

bool webpush_send(const std::string &key_path, const PushSubscription &s, const std::string &plain, int &status,
                  std::string &why) {
    status = 0;
    if (!push_endpoint_ok(s.endpoint)) {
        why = "not a push service ts5 sends to: " + origin_of(s.endpoint);
        return false;
    }
    if (plain.size() > 3000) {
        why = "the message is too long for a notification";
        return false;
    }
    std::string p256dh, auth, body;
    if (!b64url_decode(s.p256dh, p256dh) || !b64url_decode(s.auth, auth) ||
        !webpush_encrypt(p256dh, auth, plain, body, why)) {
        if (why.empty()) why = "the browser's keys can't be read";
        return false;
    }
    mbedtls_pk_context pk;
    mbedtls_pk_init(&pk);
    if (!load_key(key_path, pk, why)) {
        mbedtls_pk_free(&pk);
        return false;
    }
    const std::string pub = b64url_encode(public_raw(pk));
    // The signed token, made again at most every hour for each push service
    // (Apple's asks for no more often), good for 12 hours (it takes 24 at most).
    const std::string aud = origin_of(s.endpoint);
    static std::mutex mu;
    static std::map<std::string, std::pair<std::string, time_t>> made;  // aud + key -> token, when
    const time_t now = time(nullptr);
    std::string jwt;
    {
        std::lock_guard<std::mutex> l(mu);
        auto it = made.find(aud + " " + pub);
        if (it != made.end() && now - it->second.second < 3600) jwt = it->second.first;
    }
    if (jwt.empty()) {
        const std::string head = b64url_encode("{\"typ\":\"JWT\",\"alg\":\"ES256\"}");
        const std::string claims = b64url_encode("{\"aud\":" + json_escape(aud) + ",\"exp\":" +
                                                 std::to_string((long long)now + 12 * 3600) +
                                                 ",\"sub\":" + json_escape(subject()) + "}");
        std::string sig;
        if (!sign(pk, head + "." + claims, sig)) {
            mbedtls_pk_free(&pk);
            why = "can't sign the request";
            return false;
        }
        jwt = head + "." + claims + "." + b64url_encode(sig);
        std::lock_guard<std::mutex> l(mu);
        made[aud + " " + pub] = {jwt, now};
    }
    mbedtls_pk_free(&pk);
    HttpResponse r;
    std::string err;
    const std::vector<std::string> h = {"TTL: 86400", "Urgency: high", "Content-Encoding: aes128gcm",
                                        "Content-Type: application/octet-stream", "Authorization: vapid t=" + jwt + ", k=" + pub};
    if (!http_request("POST", s.endpoint, h, body, r, err, 20)) {
        why = "the push service doesn't answer: " + err;
        return false;
    }
    status = r.status;
    if (r.status >= 200 && r.status < 300) return true;
    std::string reason;
    Json j;
    if (Json::parse(r.body, j) && j.type == Json::OBJ) reason = j["reason"].str();
    why = "the push service said " + std::to_string(r.status) + (reason.empty() ? "" : " (" + reason + ")");
    return false;
}

}  // namespace ts5
