// Tailscale for `ts5 serve` on Linux: the server joins a tailnet as a device
// of its own, so phones and computers on the tailnet reach its API at home
// or away, with nothing forwarded on the router and nothing installed. It is
// the TV app's client (tailnet/tailscale.h) and tunnel (tailnet/wg.h), run in
// user space: no root, no network interface, nothing changed on the system,
// and only the API's port can be reached through it (and port 22, this
// machine's own SSH, while ssh is on). Every request to the API over it
// needs an API key that is on (access.h); SSH has its own sign-in.
//
// HTTPS on its tailnet name (https://NAME.TAILNET.ts.net/, port 443) once
// the tailnet's admin turns on HTTPS Certificates (with MagicDNS): it gets
// the name's certificate from Let's Encrypt, as Tailscale's own `tailscale
// cert` does (acme.h, the DNS record through the control server), renews it
// 30 days before it ends, and answers with it through a TLS front
// (https_front.h) that passes each connection on to the API. Phones' browsers
// then treat the pages as secure (their passwords fill the key in). The
// API's own port answers HTTPS too (https://NAME.TAILNET.ts.net:PORT/), so
// a browser that tries https:// first for a saved http:// address gets in.
// TS5_TS_HTTPS=0 leaves it off; TS5_ACME_DIRECTORY names another ACME server
// (Let's Encrypt's staging, say).
//
// Its files sit beside the keys (TS5_KEYS's folder):
//   tailscale.tsv   the device's keys on the tailnet (0600), written by the client
//   tailscale.conf  on=0|1, name=NAME, server=URL, ssh=0|1, written here
//   tailscale-https.crt, tailscale-https.key  the name's certificate (0600)
//   tailscale-acme.key  the Let's Encrypt account's key (0600)
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
//                       needs the tailnet's MagicDNS; "https" {state: off|waiting|
//                       admin|getting|on|problem, name, until, problem} and
//                       "https_url" once it answers HTTPS)
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
    // The browser terminal's switch (POST /ts5/tailnet {"terminal":true}),
    // kept in tailscale.conf: off until turned on.
    bool terminal() {
        std::lock_guard<std::mutex> lk(mu_);
        return terminal_;
    }

private:
    bool route(const std::string &method, const std::string &path, const std::string &body, const std::string &client,
               int &code, std::string &reply);
    void apply_locked(bool client);
    bool save_locked(std::string &err);
    void https_loop();
    void https_set(const std::string &state, const std::string &name = "", long long until = 0,
                   const std::string &problem = "");

    struct Impl;
    std::unique_ptr<Impl> d_;
    std::mutex mu_;
    std::string dir_;
    int api_port_ = 8081;
    bool on_ = false, ssh_ = false, terminal_ = false;
    std::string name_, server_;
};

// The JSON a route answers with: {"error":"WHY"}.
std::string tailnet_error(const std::string &why);

// Tests: ts5 https-test cert DOMAIN DIR (acme.h against a stand-in).
int https_test_main(int argc, char **argv);

}  // namespace ts5
