// Tailscale inside the title: the PS5 joins a tailnet as a device of its
// own, so phones and computers on the tailnet reach the API at home or away,
// with nothing forwarded on the router. Tailscale's own app can't run on the
// PS5, so this is a small client of its own:
//
//   control  Tailscale's control server (or a Headscale): sign-in, and the
//            tailnet's devices, over its Noise protocol and HTTP/2 inside
//            (noise.h, h2.h). One thread.
//   relay    a Tailscale relay server (DERP), over TLS on port 443: the
//            packets to and from the other devices go through it unless a
//            direct path works. One thread. It works on any network, the
//            building's double NAT included.
//   direct   a UDP socket of its own: devices that can reach the PS5
//            straight (on the home network, say) find it with Tailscale's
//            disco pings and then send there instead. One thread.
//
// The packets are WireGuard's, as Tailscale's apps send them; WireGuard and
// the TCP/IP inside it run in the tunnel's thread (ps5/wg/wg.h) beside the
// PS5's own WireGuard server, which takes them from here and gives its
// answers back.
#pragma once
#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
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
    uint8_t disco[32] = {}; // its disco key, for finding a direct path
    std::vector<std::string> endpoints;  // where it may be reached straight: "ip:port"
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
    // One packet that came in (through the relay or straight); false when
    // none.
    bool receive(uint8_t from[32], std::string &data);
    // Whether packets to a device go straight now, not through the relay.
    bool direct(const uint8_t key[32]);

private:
    void control_loop();
    void relay_loop();
    void direct_loop();
    // A disco message (ping, pong, call-me-maybe) from a device's disco
    // key: through the relay (relay_from, its node key) or straight (from
    // the address src). Called with mu_ held.
    void disco_in(const std::string &packet, const std::string *relay_from, uint64_t src);
    std::string disco_seal(const uint8_t to_disco[32], const std::string &plain);
    void ping_locked(const std::string &node_key, double now);
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

    // Direct paths, by node key.
    struct Path {
        std::string disco;               // its disco key (32 bytes)
        std::vector<uint64_t> endpoints; // where it may be reached (addr_key)
        uint64_t best = 0;               // the address that answered last
        double best_at = 0;              // when it last proved itself
        double wanted = 0;               // when the tunnel last sent it something
        double pinged = 0;               // when the PS5 last pinged its endpoints
        std::map<std::string, std::pair<uint64_t, double>> pings;  // TxID: where to, when
    };
    std::map<std::string, Path> paths_;
    std::map<uint64_t, std::string> by_addr_;  // address: node key
    uint8_t disco_priv_[32] = {};
    bool have_disco_ = false;
    int udp_ = -1;
    std::vector<std::string> endpoints_;  // the PS5's own: "ip:port"
    uint64_t endpoints_gen_ = 0;          // grows when they change

    std::atomic<bool> quit_{false};
    std::thread control_, relay_, direct_;
    bool started_ = false;
};

}  // namespace ts
