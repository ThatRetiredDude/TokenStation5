// The sandbox the terminal tool's commands run in (Settings → Terminal
// sandbox, /sandbox). It is bubblewrap (`bwrap`, the system's own program,
// not part of ts5): a command sees the system read-only, none of the
// user's home folder, and a home folder of its own instead, the toolbox,
// kept between chats (<data>/sandbox/home), where the model installs what
// it needs (pip, npm, a program in ~/bin); `/sandbox folder DIR` moves it
// to DIR/tokenstation-sandbox, on another drive say. The one folder of the
// user's it can change is the one ts5-tui was started in. Its environment
// holds none of the user's variables (keys in them stay out), the network
// is a switch, and what a command starts ends with it. Without bwrap, or where the
// system doesn't let it run (Ubuntu 24.04 and later stop it with
// AppArmor unless allowed), commands run as before, as the user, and
// Settings says why.
#pragma once
#include <string>
#include <vector>

namespace tui {

// Whether bwrap is here and runs. The first call tries it (a second at
// most) and remembers; again=true tries again. why: what stops it, in words.
bool sandbox_ready(std::string *why = nullptr, bool again = false);
// The toolbox: the sandbox's home folder on this machine.
std::string sandbox_home();
// Where the toolbox lives: "" for ts5's data folder (<data>/sandbox/home),
// else DIR/tokenstation-sandbox (config key sandbox_folder).
void sandbox_set_dir(const std::string &dir);
// The toolbox is there to be used: made in ts5's data folder if missing;
// one in a folder of the user's choosing must still be there (its drive
// mounted), or false and why: then no command runs.
bool sandbox_toolbox_ok(std::string &why);
// DIR as a place for the toolbox: the toolbox's path in it (made there),
// or "" and why not (no such folder, a drive that is read-only, mounted
// noexec or without Unix permissions, as exFAT and NTFS are).
std::string sandbox_dir_check(const std::string &dir, std::string &why);
// Moves the toolbox's contents from one place to another (to another
// drive: copied, then removed). 1: moved, or nothing to move; 2: the new
// place already holds a toolbox, both are kept; 0: failed, why.
int sandbox_move(const std::string &from, const std::string &to, std::string &why);
// Free space on the toolbox's drive, in bytes.
unsigned long long sandbox_free();
// The folder a command starts in and may change: the current folder, or,
// when that is the home folder, / or a folder above the home folder, the
// toolbox (shown inside as the home folder).
std::string sandbox_work();
// sandbox_work() is the home folder (shown so, holding the toolbox).
bool sandbox_starts_home();
// The program and its arguments that run `sh -c cmd` in the sandbox.
std::vector<std::string> sandbox_argv(const std::string &cmd, bool net);
// The environment a command gets in the sandbox, "NAME=value" each: none
// of the user's variables but the language, the time zone and, with the
// network, the proxy's and certificates' that reaching it may need.
std::vector<std::string> sandbox_env(bool net);
// What the toolbox holds, in bytes.
unsigned long long sandbox_size();
// Empties the toolbox. false with why.
bool sandbox_reset(std::string &why);

}  // namespace tui
