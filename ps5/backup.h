// A copy of the title's own small files (settings, keys, chats, usage) in
// /download0, so they outlast a store update. The Prospero Store updates a
// title by swapping its whole folder (/app0) for the new version's, and
// deleting it takes the folder too; /download0 is the title's own space
// outside it (/user/download/PPSA99555/download0.dat, about 320 MB).
//
// The files stay where they are in the title's folder, where Settings writes
// them and scripts/ts5-ps5.sh reads them over FTP; the copy only follows them.
// At start, restore() puts back each one the folder lacks, so a new version
// opens with the last one's settings, keys and chats. Models are too big for
// /download0 and aren't copied.
#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace app {

class Backup {
public:
    // home: the title's folder; store: the copy's folder, made when it's
    // first used. files: names in home; dirs: folders in home, each copied one
    // level deep (chats/).
    Backup(std::string home, std::string store, std::vector<std::string> files, std::vector<std::string> dirs);
    // Copies back each file home lacks. How many it copied; problems go to the
    // log.
    int restore();
    // Copies each file that changed since the last call, and takes out of the
    // copy what was removed from home since (a chat removed on the Chat
    // page). The main loop calls it now and then, and leave() before the
    // title closes. Any thread.
    void sync();

private:
    struct Seen {
        int64_t size = -1, mtime = 0;
    };
    void sync_file(const std::string &rel);
    std::mutex mu_;
    std::string home_, store_;
    std::vector<std::string> files_, dirs_;
    std::map<std::string, Seen> seen_;  // relative path: home's file when it was last copied
    bool failed_ = false;               // a copy failed (said once in the log)
};

}  // namespace app
