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

// The shell the model's commands run in: bash where the system has it
// (models write bash: arrays, [[ ]], `time`), else /bin/sh.
std::string command_shell();
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

// ---- The box: a Linux system of the sandbox's own (config sandbox_box).
// Debian's slim system, downloaded once and unpacked beside the toolbox
// (<data>/sandbox/box, or DIR/tokenstation-box). A command runs in it as
// its root, in the same bubblewrap: apt-get install works, and so does
// "sudo", which there is nothing to ask. Root in the box is the user
// outside it: every file it writes is the user's, and nothing of the
// machine's system can be changed. The user's home folder stays hidden,
// and the folder ts5-tui was started in is shared, as without the box.
// What an ordinary user can't do in a namespace with one user id (give a
// file to another user, become another user) is answered "done" by a
// small filter of system calls, so that packages' own scripts go through.

// Where the box is.
std::string box_root();
// It is set up (unpacked whole, its own files written).
bool box_there();
// The base system by name ("Debian 13"), for the words about it.
const char *box_name();
// The program and its arguments that set the box up: download (curl or
// wget), the SHA-256 checked, unpacked, its own few files written. It
// prints what it does, a line each; ends 0 when the box is there, 2 when
// it can't be done here, 3 when the download failed or was another file.
// TS5_BOX_URL and TS5_BOX_SHA256 name another base system (a test's).
std::vector<std::string> box_setup_argv();
// The command that gives a fresh box its basic tools, run in the box with
// the network (python3, pip, git, curl ...); a box without them works too.
std::string box_tools_command();
// The program and its arguments that run `bash -c cmd` in the box, and
// the environment a command gets there.
std::vector<std::string> box_argv(const std::string &cmd, bool net);
std::vector<std::string> box_env(bool net);
// What the box holds, in bytes (0 without one), and removing it whole:
// the next command sets up a fresh one.
unsigned long long box_size();
bool box_reset(std::string &why);
// This machine can run the box: its base system is built for x86-64.
// (TS5_SANDBOX_BOX=0 in the environment: never.)
bool box_can();
// A box, or what is left of one, is on the disk.
bool box_any();
// After the toolbox's place changed (sandbox_set_dir): the box that was at
// old_root goes with it. As sandbox_move: 1 moved or nothing to move, 2
// the new place already holds one, 0 failed (why).
int box_follow(const std::string &old_root, std::string &why);

}  // namespace tui
