// Models through a store update. The Prospero Store updates a title by
// swapping its whole folder for the new version's, so the models in it go
// with the old folder, which the store keeps in
// /data/prosperostore/previous/<title ID> until the next update. Models are
// too big for /download0 (backup.h), and the title can't see /data, but the
// console's FTP server can: it runs outside the title's sandbox (the
// jailbreak's), and a rename there puts a model back in the new folder at
// once, on the same drive, however big it is.
//
// So the first start of a new version asks the FTP server on the console
// itself to move back each model of the last version that the new folder
// lacks, and the same for the speech model and voices, which live with the
// models. Nothing goes over the network: the server is reached at 127.0.0.1.
#pragma once
#include <string>
#include <vector>

namespace app {

struct BroughtBack {
    bool looked = false;             // the FTP server showed this title's folder
    std::vector<std::string> moved;  // file names now in the models folder
    int left = 0;                    // files found there that couldn't be moved
    std::string why;                 // why none were looked for, for the log; "" if they were
};

// models: the title's models folder as the title sees it (/app0/models);
// title_id: its folder's name under /data/homebrew and the store's previous
// folder; eboot: the title's eboot.bin as the title sees it, whose size shows
// the FTP server's /data/homebrew/<title ID> is this folder. ports: where to
// look for the FTP server, in order.
BroughtBack bring_back_models(const std::string &models, const std::string &title_id, const std::string &eboot,
                              const std::vector<int> &ports);

}  // namespace app
