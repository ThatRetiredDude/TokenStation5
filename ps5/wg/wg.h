// WireGuard inside the title: devices away from home reach the API through
// an encrypted tunnel to the PS5, with no port of the API open to the
// internet. The PS5 gives a title no network interface of its own, so the
// tunnel runs here: WireGuard (third_party/wireguard-lwip) over one UDP
// socket, and inside it a small TCP/IP stack (third_party/lwip) whose
// connections to the API's port are passed to the API on this PS5.
//
// One thread runs it all. Settings, keys and peers come from the title
// (Settings, WireGuard); a peer is one device, with its own key and address.
#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    int port = 51820;                 // UDP, on every address of the PS5
    std::string address = "10.55.0.1";  // the PS5's in the tunnel (a /24)
    int api_port = 8081;              // passed to the API on this PS5
    std::vector<Peer> peers;
};

struct PeerState {
    std::string name, address;
    bool up = false;       // has a session now
    std::string from;      // the address its packets came from, "" if none yet
    double seen_s = -1;    // seconds since it last sent anything, -1 if never
};

class Tunnel {
public:
    ~Tunnel();
    // Starts the tunnel with c, or brings it in line with c: on and off, the
    // port, the key pair and the peers can change any time; the tunnel's
    // address can't once it has started (the title restarts for that).
    // False, with why, when c has no usable private key.
    bool apply(const Config &c, std::string &why);
    // What the tunnel is doing, for Settings: "" when it's fine (or off).
    std::string problem();
    std::vector<PeerState> peers();
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
    std::vector<PeerState> states_;
    std::map<uint16_t, std::string> clients_;  // local port: tunnel address
    std::atomic<bool> quit_{false};
    std::thread thread_;
    bool started_ = false;
};

}  // namespace wg
