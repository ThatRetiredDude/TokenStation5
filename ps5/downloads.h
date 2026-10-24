// Downloads from the catalog (catalog.h) into the title's models folder, one
// at a time on a thread of their own. Never a Runner job: the Runner does
// one job at a time, and serving, the GPU check and the speed test go on
// while a model downloads. A download writes .<file>.part, a dotfile the
// Library and the speech and voice scans skip, and every 64 MiB a
// checkpoint of its SHA-256 beside it (.<file>.sha), so going on after a
// stop, a failure or a restart of the title fetches at most 64 MiB again
// and hashes nothing. Once its size and SHA-256 agree with the catalog's,
// the part becomes <file>. A network failure is tried again 3 times, after
// 5, 15 and 45 s; anything else stops it with a reason in plain words.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "catalog.h"

namespace ts5 {
struct FetchProgress;
}

namespace app {

struct Download {
    enum Phase {
        None,      // nothing on the PS5 yet
        Queued,    // behind another download
        Running,   // held while a model loads (held)
        Retrying,  // after a network failure, until retry_at
        Stopped,   // by its button, or by closing the title; the part is kept
        Failed,    // problem says why
    } phase = None;
    uint64_t done = 0;   // bytes on the PS5 so far
    uint64_t total = 0;  // bytes it has: the server's once it answers, else the catalog's
    double speed = 0;    // bytes a second
    bool held = false;
    double retry_at = 0;   // seconds, app::now_s
    std::string problem;   // Failed, Retrying: what happened, in plain words
    bool active() const { return phase == Queued || phase == Running || phase == Retrying; }
};

// Bytes free on the volume dir is on. The title's one call to statvfs.
bool free_space(const std::string &dir, uint64_t &bytes);

// For the screen: whole percent (99 at most until the file is whole), and
// "1.9 GB of 4.7 GB · 38 MB a second", with "· about 2 minutes left" when
// asked for.
int percent(const Download &d);
std::string progress_text(const Download &d, bool left);
// Several downloads as one (Kokoro's model and a voice), for one row.
Download joined(const std::vector<std::string> &keys, const std::map<std::string, Download> &all);

class Downloads {
public:
    using Notify = std::function<void(const std::string &)>;
    // dir: the models folder. build: the title's version; a checkpoint from
    // another build isn't trusted (its hash state is that build's layout).
    Downloads(std::string dir, std::string build, Notify notify);
    ~Downloads();

    static const CatalogEntry *find(const std::string &key);
    // Where an entry downloads from: Hugging Face at its commit, or
    // TS5_CATALOG_URL/<file> when that is set (a server on the LAN, to try
    // downloads out with the console).
    static std::string url(const CatalogEntry &e);

    // Starts the entry's download, or goes on with it; after the one running.
    void start(const std::string &key);
    // Stops it, keeping the part.
    void stop(const std::string &key);
    // Removes a stopped download's part and checkpoint.
    void remove_part(const std::string &key);
    // While on, the transfer (or the hashing of a part) waits: a model loads.
    void hold(bool on);
    // Every catalog entry that has something to show, by key.
    std::map<std::string, Download> state();
    // Free space in the models folder, read once a minute and after each
    // change on the download thread; false until known (or if the console
    // can't tell).
    bool free_space(uint64_t &bytes);
    // Asks the download thread to read the free space again.
    void refresh();
    // Keys whose files arrived since the last call.
    std::vector<std::string> finished();
    // Bumped on every change worth a redraw.
    uint64_t version() const { return version_.load(); }

private:
    void loop();
    void run(const std::string &key);
    void set(const std::string &key, const Download &d);
    std::string part_of(const CatalogEntry &e) const;
    std::string sha_of(const CatalogEntry &e) const;
    void read_free();

    const std::string dir_, build_;
    Notify notify_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    std::map<std::string, Download> state_;
    std::vector<std::string> finished_;
    std::string current_;
    ts5::FetchProgress *progress_ = nullptr;  // the running transfer's, under mu_
    bool held_ = false, stop_current_ = false, quit_ = false, refresh_ = true;
    bool have_free_ = false, said_unknown_ = false;
    uint64_t free_ = 0;
    std::atomic<uint64_t> version_{0};
    std::thread thread_;
};

}  // namespace app
