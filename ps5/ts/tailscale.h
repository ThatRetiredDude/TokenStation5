// Tailscale inside the title: the PS5 joins a tailnet as a device of its
// own, so phones and computers on the tailnet reach the API at home or away,
// with nothing forwarded on the router. Tailscale's own app can't run on the
// PS5, so this is a small client of its own:
//
//   control  Tailscale's control server (or a Headscale): sign-in, and the
//            tailnet's devices, over its Noise protocol and HTTP/2 inside
//            (noise.h, h2.h). One thread.
//   relay    a Tailscale relay server (DERP), over TLS on port 443: every
//            packet to and from the other devices goes through it. One
//            thread. (No direct connections yet: the relay works on any
//            network, the building's double NAT included.)
//
// The packets are WireGuard's, as Tailscale's apps send them; WireGuard and
// the TCP/IP inside it run in the tunnel's thread (ps5/wg/wg.h) beside the
// PS5's own WireGuard server, which takes them from here and gives its
// answers back.
#pragma once
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ts {

struct Config {
    bool on = false;
    std::string control_url = "https://controlplane.tailscale.com";
    std::string hostname = "tokenstation5";
    std::string auth_key;    // optional: signs in without the QR code
    std::string state_file;  // the keys and who signed in, kept between starts
    std::string version;     // the title's, for the device's details
};

struct Status {
    enum State { Off, Starting, NeedsSignIn, Connecting, Connected, Problem } state = Off;
    std::string sign_in_url;  // NeedsSignIn: the page that approves the PS5
    std::string problem;      // what went wrong last, "" if nothing
    std::string name;         // the PS5 on the tailnet: tokenstation5.tailXXXX.ts.net
    std::string address;      // its 100.x.y.z
    std::string account;      // who signed it in
    std::string relay;        // the relay it uses, by name
    int devices = 0;          // the other devices on the tailnet
};

// A device on the tailnet, as the tunnel needs it.
struct Peer {
    uint8_t key[32] = {};   // its node key (its WireGuard public key)
    std::string name;       // its tailnet name, short
    std::string address;    // its IPv4 in the tailnet
};

struct Netmap {
    uint64_t version = 0;   // grows with each change
    uint8_t node_priv[32] = {};  // the PS5's WireGuard private key
    std::string address;    // "" until signed in
    std::vector<Peer> peers;
};

class Client {
public:
    ~Client();
    // Starts, stops or changes it: on, the control server, the name.
    void apply(const Config &c);
    // Signs the PS5 out of the tailnet and forgets its keys; the next
    // sign-in is a new device.
    void sign_out();
    Status status();

    // For the tunnel's thread.
    uint64_t netmap_version();
    Netmap netmap();
    // A packet to a device (by node key), through the relay.
    void send(const uint8_t to[32], const void *data, size_t len);
    // One packet that came through the relay; false when none.
    bool receive(uint8_t from[32], std::string &data);

private:
    void control_loop();
    void relay_loop();
    void set_state(Status::State s, const std::string &problem = "");

    std::mutex mu_;
    Config cfg_;
    uint64_t cfg_gen_ = 0;  // grows with each apply()
    bool sign_out_ = false;
    Status status_;
    Netmap map_;
    // The relays (region ID: its servers' host:port), from the control server.
    struct Region {
        int id = 0;
        std::string name, code;
        std::vector<std::pair<std::string, int>> nodes;  // host, port
        bool insecure = false;
    };
    std::vector<Region> regions_;
    int home_region_ = 0;  // the relay the PS5 uses; 0 until chosen
    uint8_t node_priv_[32] = {};
    bool have_node_key_ = false;
    std::deque<std::pair<std::string, std::string>> outbox_, inbox_;  // (node key, packet)

    std::atomic<bool> quit_{false};
    std::thread control_, relay_;
    bool started_ = false;
};

}  // namespace ts
