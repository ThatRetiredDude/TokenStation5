// The page a device asks to join WireGuard from, served by the API on the
// home network at /wireguard (Server::set_pages). It takes the device's
// name and public key (the device's WireGuard app makes its key pair; the
// private key never leaves it), asks the TV to approve it (wg_devices.h),
// and shows what to put in the device's tunnel.
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
