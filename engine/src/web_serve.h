#pragma once
#include <functional>
#include <string>

#include "server.h"

namespace ts5 {

// ts5 serve's own pages, for a browser (Linux):
//   /                  the chat page: talks to /v1 with the key it is given,
//                      kept in that browser only.
//   /ts5/terminal      the browser terminal: ts5-tui in xterm.js, over a
//                      WebSocket at /ts5/terminal/socket. Only while its
//                      switch is on (terminal()), only for the tailnet's
//                      devices (100.64.0.0/10) with a key that is on, and
//                      this machine itself; never for the rest of the network.
//   /ts5/web/...       the pages' files (xterm.js, its CSS, its fit addon).
// The pages themselves are no secret and need no key; what they do with the
// API needs one, as from any app.
struct WebOptions {
    // Who gets in with which key (Server::Authorize): the same as /v1's.
    Server::Authorize authorize;
    // The browser terminal's switch, asked at each connection and every
    // second while one is open (off ends it). Without one: off.
    std::function<bool()> terminal;
    // The program the terminal runs, as a shell command; "" for ts5-tui
    // beside this ts5 (or on PATH). TS5_TERMINAL_COMMAND sets it too (tests).
    std::string command;
};

void attach_web(Server &s, const WebOptions &o);

// For tests: the WebSocket handshake's answer to a Sec-WebSocket-Key.
std::string websocket_accept(const std::string &key);

}  // namespace ts5
