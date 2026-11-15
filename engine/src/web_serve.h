#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "server.h"

namespace ts5 {

// ts5 serve's own pages, for a browser (Linux):
//   /                  the chat page: talks to /v1 with the key it is given,
//                      kept in that browser only.
//   /ts5/terminal      the browser terminal: ts5-tui in xterm.js, over a
//                      WebSocket at /ts5/terminal/socket. Only while its
//                      switch is on (terminal()), only for the tailnet's
//                      devices (100.64.0.0/10) with a key that is on, and
//                      this machine itself; never for the rest of the network,
//                      and never with another user's key than the owner's.
//   /ts5/web/...       the pages' files (xterm.js, its CSS, its fit addon).
// The pages themselves are no secret and need no key; what they do with the
// API needs one, as from any app.
//   /ts5/me            GET: whose key the request has, and what that user
//                      may do (Server::Grant), for the chat page.
//   /ts5/chats         GET: the key's user's chats ({"chats":[...]});
//                      DELETE: all of them. /ts5/chats/ID: PUT one (a JSON
//                      object, as the chat page keeps it), DELETE one. Only
//                      with a key: each user's own, never another's (the
//                      owner's included), in user_dir(USER)/chats.
//   /ts5/controls      the owner's phone controls (models, the machine,
//                      Settings): a page that asks /ts5/control.
//   /ts5/control       GET (the state) or POST {"do": ...}: handed to the
//                      running ts5-tui (linux/tui/control.h) and its answer
//                      handed back; the owner's key only (a key even on
//                      this machine), this machine and the tailnet only. ts5-tui not running:
//                      started in its tmux session (not by the service,
//                      which can't start the owner's: 503 says so).
//   /ts5/switch        GET: the models here the key's user may use, and
//                      the one loaded; POST {"model": M}: the owner's
//                      ts5-tui loads it. The owner, and a user who may
//                      switch (AccessUser::may_switch), only.
//   /ts5/users         GET, the owner's key only: the users whose chats the
//                      owner may read (each chose so when made, and sees it);
//                      /ts5/users/ID/chats: that user's chats, to read.
struct WebOptions {
    // Who gets in with which key (Server::Authorize): the same as /v1's.
    Server::Authorize authorize;
    // As whom (Server::Admit); when set, the browser terminal is the
    // owner's alone, and /ts5/me and /ts5/chats answer.
    Server::Admit admit;
    // Where a user's own things are (KeyFile::user_dir); without it the
    // chats stay in each browser.
    std::function<std::string(const std::string &user)> user_dir;
    // The users whose chats the owner may read (AccessUser::owner_reads),
    // as id and name, for /ts5/users.
    std::function<std::vector<std::pair<std::string, std::string>>()> readable_users;
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
