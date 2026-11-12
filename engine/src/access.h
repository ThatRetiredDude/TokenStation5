// Who may use `ts5 serve` from another machine: API keys in a file, made
// and switched with `ts5 keys` (ts5-tui's Access page runs it).
//
// The file is the TV app's keys.tsv, line for line (ps5/keys.h): "# ts5
// keys v1", then "#" and the column names (id, name, sha256, on, created,
// used), then a line per key. It holds each key's SHA-256, never the key,
// which is shown once, when it is made. One more line, "# required=1",
// says that other machines need a key; the TV app reads past it.
//
// `ts5 serve` with TS5_KEYS=FILE asks the file about every request:
//   a key that is on      let in, under the key's name (usage log)
//   this machine          let in (127.0.0.1: ts5-tui, a program beside it)
//   another machine       let in without a key unless "required" is set
// The file is read again when it changes, so a key made or turned off
// holds from the next request, with the model still loaded. When a key
// was last used goes to FILE.used ("id<TAB>seconds", at most once a minute
// a key), so the file itself has one writer.
#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ts5 {

struct AccessKey {
    std::string id;    // 8 hex digits
    std::string name;  // what the person called it, such as "laptop"
    std::string sha256;
    bool on = true;
    int64_t created = 0, used = 0;  // Unix seconds; used 0: never
};

class KeyFile {
public:
    explicit KeyFile(std::string path) : path_(std::move(path)) {}
    const std::string &path() const { return path_; }

    // Reads the file and FILE.used. No file: no keys. False with err when
    // it is there and can't be read.
    bool load(std::string &err);
    std::vector<AccessKey> keys() const;
    bool required() const;

    // A new key, turned on and saved: its text, such as
    // "ts5-k7qp2-wx9md-3hf8r-tz6na" (100 random bits). "" with err.
    std::string create(const std::string &name, std::string &id, std::string &err);
    // which: a key's id, or its name when one key has it.
    bool set_on(const std::string &which, bool on, std::string &err);
    bool remove(const std::string &which, std::string &err);
    bool set_required(bool required, std::string &err);

    // Server::Authorize (server.h), on any thread.
    bool authorize(const std::string &key, const std::string &client, std::string &name);

private:
    bool load_locked(std::string &err);
    bool save_locked(std::string &err);
    int find_locked(const std::string &which, std::string &err) const;
    void note_used_locked(AccessKey &k, int64_t now);

    std::string path_;
    mutable std::mutex mu_;
    std::vector<AccessKey> keys_;
    bool required_ = false;
    // What the file was when it was read (authorize looks again a second later at the earliest).
    int64_t seen_mtime_ns_ = -1, seen_size_ = -1;
    double looked_ = 0;
};

// Where the keys are without TS5_KEYS: access/keys.tsv in the apps' data
// folder (TS5_DATA_DIR, else $XDG_DATA_HOME/tokenstation, else
// ~/.local/share/tokenstation). Not the TV app's own keys.tsv there: the
// TV app keeps its keys in memory and writes the whole file, so a key
// made beside it would be lost.
std::string default_keys_path();

// ts5 keys [--file FILE] [--json] list | new NAME | on KEY | off KEY | remove KEY | require on|off
int keys_main(int argc, char **argv);

}  // namespace ts5
