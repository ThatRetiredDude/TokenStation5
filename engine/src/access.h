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
//   a device on a tailnet only with a key (100.64.0.0/10: Tailscale, such
//                         as ts5's own, tailnet_serve.h)
// The file is read again when it changes, so a key made or turned off
// holds from the next request, with the model still loaded. When a key
// was last used goes to FILE.used ("id<TAB>seconds", at most once a minute
// a key), so the file itself has one writer.
//
// Users (users.tsv beside the keys): every key belongs to a user, and the
// user holds the permissions, so a lost phone's key is turned off without
// touching the rest, and a user's permissions change on all of its keys at
// once, from the next request. The owner is a user that is always there
// ("owner", never in the file): everything, the browser terminal too. A key
// without a user (every key made before users) is the owner's. Another user
// has the chat page and the API (an editor pointed at the address with its
// own key), never the browser terminal, and:
//   web_search  ts5's own web search in its chats
//   app_tools   the tools an app sends (an editor's, which the app runs)
//   tokens_day  tokens a day (the prompt read anew and the reply), 0 none
//   max_reply   the longest reply in tokens, 0 none
//   owner_reads the owner may read its chats (chosen when it is made; the
//               user sees it in its own settings on the chat page)
// Its chats from the chat page are its own (users/ID/chats beside access/).
// How many tokens each user had today goes to users.today (the server
// writes it; "ts5 users list" reads it).
#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "server.h"

namespace ts5 {

struct AccessKey {
    std::string id;    // 8 hex digits
    std::string name;  // what the person called it, such as "laptop"
    std::string sha256;
    bool on = true;
    int64_t created = 0, used = 0;  // Unix seconds; used 0: never
    std::string user = "owner";     // whose key it is
};

struct AccessUser {
    std::string id;    // "owner", or 8 hex digits
    std::string name;  // what the owner called them, such as "Sam"
    bool on = true;    // off: none of its keys gets in
    bool web_search = true, app_tools = true;
    bool owner_reads = false;
    int64_t tokens_day = 0, max_reply = 0;
    int64_t created = 0;
    int64_t today = 0;  // tokens so far today (users.today)
    bool owner() const { return id == "owner"; }
};

// The presets a user is made with: "chat" (chat only, no web search) and
// "chat-web" (with ts5's web search). The owner's own is "owner".
bool apply_preset(AccessUser &u, const std::string &preset);
// Which preset the user's switches are, or "custom".
std::string preset_of(const AccessUser &u);

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
    // "ts5-k7qp2-wx9md-3hf8r-tz6na" (100 random bits). "" with err. user:
    // whose (a user's id or name; "" the owner).
    std::string create(const std::string &name, std::string &id, std::string &err, const std::string &user = "");
    // which: a key's id, or its name when one key has it.
    bool set_on(const std::string &which, bool on, std::string &err);
    bool remove(const std::string &which, std::string &err);
    bool set_required(bool required, std::string &err);

    // Server::Authorize (server.h), on any thread.
    bool authorize(const std::string &key, const std::string &client, std::string &name);
    // Server::Admit: the same, with whose request it is and what it may do.
    bool admit(const std::string &key, const std::string &client, Server::Grant &g);

    // Users: the owner first, then the file's.
    std::vector<AccessUser> users() const;
    // A new user from a preset, saved; its id, or "" with err.
    std::string add_user(const std::string &name, const std::string &preset, std::string &err);
    // Changes a user (its id or name): "name=NAME", "preset=P", "on=0|1",
    // "web_search=0|1", "app_tools=0|1", "owner_reads=0|1", "tokens_day=N",
    // "max_reply=N".
    bool set_user(const std::string &which, const std::vector<std::string> &changes, std::string &err);
    // A user and its keys; its chats stay unless forget is set.
    bool remove_user(const std::string &which, bool forget, std::string &err);
    // Counts a request's tokens to its user for today (the server, after it).
    void count(const std::string &user, int64_t tokens);
    // Where a user's own things are (its chats): users/ID beside the
    // keys' folder.
    std::string user_dir(const std::string &user) const;

private:
    int find_user_locked(const std::string &which, std::string &err) const;
    bool save_users_locked(std::string &err);
    void load_today_locked();
    const AccessUser *user_locked(const std::string &id) const;
    bool fresh_locked();

    bool load_locked(std::string &err);
    bool save_locked(std::string &err);
    int find_locked(const std::string &which, std::string &err) const;
    void note_used_locked(AccessKey &k, int64_t now);

    std::string path_;
    mutable std::mutex mu_;
    std::vector<AccessKey> keys_;
    std::vector<AccessUser> users_;  // not the owner
    std::map<std::string, int64_t> today_;  // tokens today by user
    std::string today_date_;                // "2026-11-14", local
    bool required_ = false;
    int64_t users_mtime_ns_ = -1, users_size_ = -1;
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

// Whether a peer's address is on a tailnet (100.64.0.0/10, the range
// Tailscale gives its devices).
bool on_tailnet(const std::string &client);

// ts5 keys [--file FILE] [--json] list | new NAME [--user USER] | on KEY | off KEY | remove KEY | require on|off
int keys_main(int argc, char **argv);
// ts5 users [--file FILE] [--json] list | new NAME [--preset P] | set USER CHANGE... | remove USER [--forget]
int users_main(int argc, char **argv);

}  // namespace ts5
