// /update: TokenStation5's newest build, installed from ts5-tui. ts5-tui
// closes (a model it loaded is stopped, so nothing runs the old program),
// one shell script downloads the newest build from the linux-latest release
// and installs it the way this copy was installed (.deb with apt, .rpm with
// dnf, the Arch package with pacman, the portable build with its
// install.sh; sudo asks for the password in the terminal), and ts5-tui
// opens again, the new one, with the model it had loaded back.
// `ts5-tui --update [--dry-run]` runs the script alone.
#pragma once
#include <string>

namespace tui {

// The shell command that updates this copy (self: the ts5-tui program's
// path). It writes one line for the user to note ("Updated: 1.3.0 to
// 1.3.1", "Already the newest: 1.3.0", or why it didn't) and ends 0
// updated, 10 already the newest, else not updated. dry: it only says
// what it would do.
std::string update_cmd(const std::string &self, const std::string &note, bool dry);

}  // namespace tui
