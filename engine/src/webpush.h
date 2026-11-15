// Web Push: a notification to a browser, or to a phone's Home Screen web app
// (iOS 16.4 and later), through the push service that browser chose (Apple's,
// Google's, Mozilla's). The message is encrypted for that browser alone
// (RFC 8291, aes128gcm): the push service carries it without being able to
// read it. Each request is signed with this machine's own key (VAPID, RFC
// 8292), the key the browser was given when it subscribed.
#pragma once
#include <string>

namespace ts5 {

// A browser's subscription, as PushManager.subscribe() gives it: where to
// post, and its keys (base64url): p256dh (its P-256 public key, 65 bytes) and
// auth (16 bytes).
struct PushSubscription {
    std::string endpoint, p256dh, auth;
};

// base64url without padding; decoding takes either alphabet, with or without
// padding. False on anything else.
std::string b64url_encode(const std::string &bytes);
bool b64url_decode(const std::string &text, std::string &bytes);

// This machine's VAPID key, kept in path (PEM, 0600), made the first time.
// public_b64: its public key (65 bytes, uncompressed), base64url, for the
// page to subscribe with.
bool vapid_key(const std::string &path, std::string &public_b64, std::string &why);

// RFC 8291: plain encrypted for the browser (p256dh, auth: raw bytes), one
// record, as the request's body. as_private (32 bytes) and salt (16): fixed
// ones for a test; empty, new random ones (always, for a real message).
bool webpush_encrypt(const std::string &p256dh, const std::string &auth, const std::string &plain, std::string &body,
                     std::string &why, const std::string &as_private = "", const std::string &salt = "");

// Whether a subscription's endpoint is one ts5 posts to: https on a known push
// service (Apple's, Google's, Mozilla's, Microsoft's). TS5_PUSH_HOSTS
// (host[:port],...) adds others, http too, for tests: an endpoint is never an
// address on the machine's own network unless that is set.
bool push_endpoint_ok(const std::string &endpoint);

// Sends plain (at most 3 KB) to the subscription, signed with the key in
// key_path. status: the push service's answer (201: taken; 404 or 410: the
// subscription is gone, to be forgotten). False with why when it can't be
// sent or the service said no.
bool webpush_send(const std::string &key_path, const PushSubscription &s, const std::string &plain, int &status,
                  std::string &why);

}  // namespace ts5
