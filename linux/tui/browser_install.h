// Installing a browser for the model's browser tool, when the tool is
// switched on and none is here: Chromium from this system's packages (the
// administrator's password is asked for by sudo, in the terminal), or
// Obscura from its makers' GitHub release into the home folder (no
// administrator needed; the file is checked against a pinned SHA-256).
// One shell script does both; ts5-tui leaves its screen while it runs, and
// `ts5-tui --install-browser chromium|obscura [--dry-run]` runs it alone.
#pragma once
#include <string>

namespace tui {

// The shell command that installs what ("chromium" or "obscura"). dry: it
// only says what it would do.
std::string browser_install_cmd(const std::string &what, bool dry);

}  // namespace tui
