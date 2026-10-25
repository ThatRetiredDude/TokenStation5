// The page a device asks to join WireGuard from, served by the API on the
// home network at /wireguard (Server::set_pages). The page makes the
// device's key pair in its browser, asks the TV to approve the public half
// (wg_devices.h) and then gives the device a finished tunnel file; the
// private key stays in the browser. Or the device's WireGuard app makes the
// key pair, and the page takes its public key and shows what to put in the
// tunnel.
#pragma once
#include <string>

#include "server.h"
#include "wg_devices.h"

namespace app {

struct WgPageInfo {
    bool on = false;         // WireGuard is on in Settings
    bool running = true;     // and the tunnel started (it can't when its port is refused)
    std::string public_key;  // the PS5's
    int port = 9030;         // UDP
    int api_port = 8081;
};

// Server::Pages for /wireguard; false for any other path. client: the
// requester as the API names it (a device in the tunnel can't add devices).
bool wg_page(const std::string &method, const std::string &path, const std::string &body, const std::string &client,
             const WgPageInfo &info, WgStore &store, ts5::Server::Page &out);

}  // namespace app
