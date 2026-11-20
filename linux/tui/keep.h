// The sandbox that stays up between commands. A command of the model's no
// longer gets a bubblewrap of its own that ends with it: one sandbox (the
// box, or the plain one) is started for the first command and kept, with a
// small shell of ts5-tui's inside it that runs each command it is handed.
// So a server, a database or a build that one command starts in the
// background is still there for the next. It ends when ts5-tui ends (its
// shell reads from a pipe only ts5-tui writes: when that closes, it ends,
// and every program in it), with a new chat, when the sandbox is reset or
// moved, when the network switch changes, or after it sits unused (30
// minutes; an unattended task keeps it for the task).
//
// Its memory has a cap (Settings → Sandbox memory, default 2 GiB): every
// program in it together. Where the user's systemd can give it a memory
// limit of its own (a cgroup), the kernel keeps to it; else ts5-tui adds
// up what the sandbox's programs use, four times a second, and ends the biggest
// when they go past it. The model is told which one was ended, and why.
//
// TS5_SANDBOX_KEEP=0 in the environment: a sandbox of its own for every
// command, as before.
#pragma once
#include <atomic>
#include <string>

namespace tui {

struct KeptRun {
    bool started = false;  // the command was handed to the sandbox
    std::string out;       // its output and errors together (start and end of a flood)
    int code = -1;
    bool timed_out = false, stopped = false;
    bool left_running = false;  // something it started is still running in the sandbox
    bool fresh = false;         // the sandbox was started for this command
    double secs = 0;
    // What happened to the sandbox meanwhile, in words for the model: a
    // program ended for the memory cap, the sandbox itself gone.
    std::string note;
};

// Whether commands go to a sandbox that stays up (TS5_SANDBOX_KEEP isn't 0).
bool kept_on();
// Runs cmd in the sandbox that stays up, the box (box) or the plain one,
// with or without the network, starting it first where it isn't up in
// that form. limit: seconds; cancel stops it. started false, and why,
// when the sandbox could not be started (the caller runs it the old way).
KeptRun kept_run(const std::string &cmd, bool box, bool net, double limit, const std::atomic<bool> &cancel,
                 std::string &why);
// Ends it, and every program in it.
void kept_end();
bool kept_up();
// The memory cap in bytes (0: none), for the next sandbox started.
void kept_set_cap(unsigned long long bytes);
unsigned long long kept_cap();
// Minutes unused after which it ends (0: never; an unattended task's).
void kept_set_idle(int minutes);
// What its programs use now, in bytes (0 when it isn't up).
unsigned long long kept_memory();
// How the cap is kept: "the kernel (a cgroup of its own)", "ts5-tui,
// four times a second", or "" when it isn't up.
std::string kept_cap_how();
// 2 GiB, 1.5 GiB, 512 MiB.
std::string bytes_words(unsigned long long b);

}  // namespace tui
