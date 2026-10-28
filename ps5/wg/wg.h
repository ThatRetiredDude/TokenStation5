// WireGuard inside the title: devices away from home reach the API through
// an encrypted tunnel to the PS5, with no port of the API open to the
// internet. The PS5 gives a title no network interface of its own, so the
// tunnel runs here: WireGuard (third_party/wireguard-lwip) over one UDP
// socket, and inside it a small TCP/IP stack (third_party/lwip) whose
// connections to the API's port are passed to the API on this PS5.
//
// One thread runs it all. Settings, keys and peers come from the title
// (Settings, WireGuard); a peer is one device, with its own key and address.
//
// The same thread runs the PS5's device on a tailnet (ps5/ts/tailscale.h):
// a second WireGuard device and network interface, whose packets come and
// go through Tailscale's relay rather than the UDP socket, and whose
// connections to the API's port are passed on the same way.
#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ts {
class Client;
}

namespace wg {

// How many peers the tunnel takes (third_party/wireguard-lwip's
// WIREGUARD_MAX_PEERS, set in scripts/vendor-wireguard.sh).
constexpr int kMaxPeers = 16;

// A key pair, base64 as WireGuard's apps write them. False when the system
// gave no random numbers.
bool make_keys(std::string &private_key, std::string &public_key);
// The public key of a private one, "" if it isn't one.
std::string public_key_of(const std::string &private_key);

struct Peer {
    std::string name;        // the device, as Settings shows it
    std::string public_key;  // base64
    std::string address;     // its address in the tunnel, 10.55.0.N
    // Optional: where to reach it, for a peer the PS5 calls out to (a
    // WireGuard server on the internet, say), with keepalives every
    // keepalive seconds so its replies get through home routers. Devices
    // that call in leave both empty.
    std::string endpoint;  // host:port, an IPv4 address
    int keepalive = 0;
};

struct Config {
    bool on = false;
    std::string private_key;          // base64
    int port = 9030;                  // UDP, on every address of the PS5 (it refuses 51820 and 40000)
    std::string address = "10.55.0.1";  // the PS5's in the tunnel (a /24)
    int api_port = 8081;              // passed to the API on this PS5
    std::vector<Peer> peers;
    bool tailscale = false;           // the PS5 on a tailnet (use_tailscale's client)
};

struct PeerState {
    std::string name, address;
    bool up = false;       // has a session now
    std::string from;      // the address its packets came from, "" if none yet; on the
                           // tailnet "direct" or "relay"
    double seen_s = -1;    // seconds since it last sent anything, -1 if never
};

class Tunnel {
public:
    ~Tunnel();
    // The tailnet's client, before the first apply(); its packets go
    // through this tunnel's thread while Config::tailscale is on.
    void use_tailscale(ts::Client *c) { ts_ = c; }
    // Starts the tunnel with c, or brings it in line with c: on and off, the
    // port, the key pair and the peers can change any time; the tunnel's
    // address can't once it has started (the title restarts for that).
    // False, with why, when c has no usable private key.
    bool apply(const Config &c, std::string &why);
    // What the tunnel is doing, for Settings: "" when it's fine (or off).
    std::string problem();
    std::vector<PeerState> peers();
    // The tailnet's devices that have talked to the PS5 since it started.
    std::vector<PeerState> tailnet_peers();
    // For the API: the tunnel address of the device behind a connection the
    // tunnel made to the API from local_port on this PS5, "" if it isn't one.
    // The API names requests by it, and asks it for a key when keys are on.
    std::string client_for(uint16_t local_port);

private:
    void loop();
    std::mutex mu_;
    Config want_;            // what apply() asked for; the thread follows it
    bool changed_ = false;
    std::string problem_;
    std::vector<PeerState> states_, ts_states_;
    ts::Client *ts_ = nullptr;
    // Local port: tunnel address, and when the entry lapses (0: its
    // connection is open). client_for takes an entry; one whose connection
    // reached the API stays until then, even closed, so a request never
    // loses its name and passes for the console's own (127.0.0.1).
    std::map<uint16_t, std::pair<std::string, double>> clients_;
    std::atomic<bool> quit_{false};
    std::thread thread_;
    bool started_ = false;
};

}  // namespace wg
