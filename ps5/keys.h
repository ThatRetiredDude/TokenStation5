// API keys, made on the Settings page (one per app, by name) and kept in
// keys.tsv in the title's folder. The file holds each key's SHA-256, never
// the key: a key shows once, when it's made. With "Require a key" on
// (TS5_KEYS_REQUIRED=1 in ts5.env), apps on the network must send a key that is
// turned on; the Chat page on the PS5 itself never needs one. With it off,
// anyone gets in, and a request that sends a key still has its name in
// usage.tsv, so Home's "by app" uses names.
//
// keys.tsv: "# ts5 keys v1", then "#" and the column names (id, name, sha256,
// on, created, used: Unix seconds, 0 for never), then a line per key.
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace app {

struct ApiKey {
    std::string id;    // 8 hex digits, for the page and the file
    std::string name;  // what the person called it, such as "Kilo Code on the laptop"
    std::string sha256;
    bool on = true;
    int64_t created = 0, used = 0;  // Unix seconds; used 0: never
};

class KeyStore {
public:
    // Reads path (none yet: no keys). Problems go to the log.
    void load(const std::string &path);
    std::vector<ApiKey> list() const;
    // A new key, turned on, saved at once: its text, such as
    // "ts5-k7qp2-wx9md-3hf8r-tz6na" (100 random bits). "" with err if the
    // system has no random numbers or the file can't be written.
    std::string create(const std::string &name, std::string &id, std::string &err);
    bool set_on(const std::string &id, bool on, std::string &err);
    bool remove(const std::string &id, std::string &err);

    void set_required(bool r) { required_ = r; }
    bool required() const { return required_; }
    // Server::Authorize: whether a request with this key from this address
    // gets in, and the key's name. Any thread.
    bool authorize(const std::string &key, const std::string &client, std::string &name);
    // Writes the last-used times if they changed, at most once a minute
    // (now: Unix seconds). The main loop calls it.
    void save_used(int64_t now);

private:
    bool save_locked(std::string &err);
    mutable std::mutex mu_;
    std::string path_;
    std::vector<ApiKey> keys_;
    std::atomic<bool> required_{false};
    bool used_dirty_ = false;
    int64_t used_saved_ = 0;
};

}  // namespace app
