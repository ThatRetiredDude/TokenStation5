#include "keep_models.h"

#include <sys/stat.h>

#include <cstring>

#include "ftp.h"
#include "log.h"

namespace app {

namespace {

// A file that's being written, or isn't a model: left where it is.
bool skip(const std::string &name) {
    auto ends = [&](const char *e) {
        const size_t n = strlen(e);
        return name.size() >= n && name.compare(name.size() - n, n, e) == 0;
    };
    return name[0] == '.' || ends(".part") || ends(".new") || ends(".tmp");
}

}  // namespace

BroughtBack bring_back_models(const std::string &models, const std::string &title_id, const std::vector<int> &ports) {
    BroughtBack r;
    // The folder the server calls /data/homebrew/<title ID> is this one.
    const std::string home = "/data/homebrew/" + title_id, old = "/data/prosperostore/previous/" + title_id + "/models";
    const int port = find_console_ftp(models, home + "/models", ports, &r.why);
    if (!port) return r;
    Ftp ftp;
    if (!ftp.open(port)) {
        r.why = "the FTP server on port " + std::to_string(port) + " stopped answering";
        return r;
    }
    ftp.login();
    r.looked = true;
    std::vector<std::string> names;
    if (!ftp.names(old, names)) {
        r.why = "no models from a last version (" + old + ": " + ftp.text() + ")";
        return r;
    }
    for (const std::string &name : names) {
        if (skip(name)) continue;
        struct stat here {};
        if (stat((models + "/" + name).c_str(), &here) == 0) continue;  // the new folder has it already
        if (ftp.size(old + "/" + name) < 0) continue;                    // a folder, or gone
        // RNTO's success is 250, or 226 from the console's ftpsrv.
        if (ftp.cmd("RNFR " + old + "/" + name) == 350 && ftp.cmd("RNTO " + home + "/models/" + name) / 100 == 2 &&
            stat((models + "/" + name).c_str(), &here) == 0) {
            r.moved.push_back(name);
        } else {
            r.left++;
            ts5::log_line("ts5-title: couldn't bring back %s: %s", name.c_str(), ftp.text().c_str());
        }
    }
    return r;
}

}  // namespace app
