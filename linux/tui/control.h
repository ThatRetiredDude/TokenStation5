// ts5-tui's control sockets: the owner's phone controls (the chat page's
// Controls, /ts5/control in ts5 serve) reach the running ts5-tui here, so
// the phone loads models, sets the fan and changes Settings the way the
// keys on ts5-tui's pages do. One JSON request a connection, one JSON
// answer.
//
// Two sockets: data_dir()/control.sock (0600) for a ts5 serve this user
// runs, and, while ts5-tui works with the background service, the
// service's /run/tokenstation/control.sock (0660, group tokenstation), so
// the service (user tokenstation, which can't look in home folders)
// reaches the owner's ts5-tui. The peer's uid is checked on each: this
// user, and on the service's socket the service's user.
#pragma once

#include <functional>
#include <string>

namespace tui {

// data_dir()/control.sock (ts5 serve looks for it in the same place).
std::string control_socket_path();
// /run/tokenstation/control.sock (TS5_RUN_DIR for tests).
std::string control_shared_path();

// Listens (a thread takes the connections). False, and why, when the own
// socket can't be had (another ts5-tui answers there); the service's may
// still be taken later.
bool control_start(std::string &why);

// Whether to take the service's socket too (ts5-tui works with the
// service); taken, kept and made again by the accept thread.
void control_share(bool on);
// Why the service's socket isn't taken while wanted, or "".
std::string control_status();

// On the main thread, between frames: each request that came since gets
// handle's answer.
void control_pump(const std::function<std::string(const std::string &request)> &handle);

void control_stop();

}  // namespace tui
