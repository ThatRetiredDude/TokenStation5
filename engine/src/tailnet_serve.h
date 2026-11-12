// Tailscale for `ts5 serve` on Linux: the server joins a tailnet as a device
// of its own, so phones and computers on the tailnet reach its API at home
// or away, with nothing forwarded on the router and nothing installed. It is
// the TV app's client (tailnet/tailscale.h) and tunnel (tailnet/wg.h), run in
// user space: no root, no network interface, nothing changed on the system,
// and only the API's port can be reached through it (and port 22, this
// machine's own SSH, while ssh is on). Every request to the API over it
// needs an API key that is on (access.h); SSH has its own sign-in.
//
// Its files sit beside the keys (TS5_KEYS's folder):
//   tailscale.tsv   the device's keys on the tailnet (0600), written by the client
//   tailscale.conf  on=0|1, name=NAME, server=URL, ssh=0|1, written here
// TS5_TS_NAME and TS5_TS_SERVER give the name and control server when
// tailscale.conf doesn't, TS5_TS_AUTH_KEY signs in without the QR code.
// TS5_NET_LOG=1 puts the tailnet's traffic in the log (its first lines), to
// find why a device gets no answer.
//
// This machine controls it through the server (from 127.0.0.1 only; ts5-tui's
// Access page does):
//   GET  /ts5/tailnet   what it is doing, as JSON (status_json; "peers" are
//                       the devices that reached it lately, "devices" all of them;
//                       "url" by its 100.x address, "name_url" by its name, which
//                       needs the tailnet's MagicDNS)
//   POST /ts5/tailnet   {"on": true|false}, {"name": "NAME"}, {"ssh": true|false},
//                       {"sign_out": true}
#pragma once
#include <memory>
#include <mutex>
#include <string>

#include "server.h"

namespace ts5 {

class Tailnet {
public:
    Tailnet();
    ~Tailnet();
    // Reads tailscale.conf in dir and starts it if it was on. api_port: the
    // server's, which the tunnel passes connections to.
    void start(const std::string &dir, int api_port);
    // Names tunnel connections by the device behind them, and answers
    // /ts5/tailnet. Before Server::start.
    void attach(Server &s);
    std::string status_json();

private:
    bool route(const std::string &method, const std::string &path, const std::string &body, const std::string &client,
               int &code, std::string &reply);
    void apply_locked();
    bool save_locked(std::string &err);

    struct Impl;
    std::unique_ptr<Impl> d_;
    std::mutex mu_;
    std::string dir_;
    int api_port_ = 8081;
    bool on_ = false, ssh_ = false;
    std::string name_, server_;
};

// The JSON a route answers with: {"error":"WHY"}.
std::string tailnet_error(const std::string &why);

}  // namespace ts5
