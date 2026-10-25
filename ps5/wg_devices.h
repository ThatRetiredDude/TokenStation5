// The WireGuard devices this PS5 lets in, and its own key, kept in
// wireguard.tsv in the title's folder (wg/wg.h runs the tunnel). Each
// device makes its own key pair in its WireGuard app and gives the PS5 only
// the public half: the PS5 never makes, sees or keeps a device's private
// key. A device asks to join from a page the API serves on the home network
// (/wireguard), and joins only once someone approves it on the TV.
//
// wireguard.tsv: "# ts5 wireguard v1", "@private_key\tBASE64" (the PS5's),
// then "#" and the column names (id, name, public_key, address, on, added:
// Unix seconds), then a line per device.
#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "wg/wg.h"

namespace app {

struct WgDevice {
    std::string id;          // 8 hex digits, for the page and the file
    std::string name;        // what the person called it, such as "Laptop"
    std::string public_key;  // base64, as its app shows it
    std::string address;     // its address in the tunnel, 10.55.0.N
    bool on = true;
    int64_t added = 0;  // Unix seconds
};

// A device asking to join, waiting for the TV.
struct WgRequest {
    std::string name, public_key, address, from;  // from: where the page was asked, on the home network
    double at = 0;                                  // when (steady seconds)
};

class WgStore {
public:
    static constexpr const char *kAddress = "10.55.0.1";  // the PS5's in the tunnel
    // Reads path (none yet: no devices, no key). Problems go to the log.
    void load(const std::string &path);
    // The PS5's public key, making its key pair first if it has none; "" with
    // err if it can't (no random numbers, the file can't be written).
    std::string public_key(std::string &err);
    // Whether the PS5 has its key pair yet.
    bool has_key() const;
    // A new key pair for the PS5: every device then needs the new public key.
    bool new_key(std::string &err);
    std::vector<WgDevice> devices() const;
    bool set_on(const std::string &id, bool on, std::string &err);
    bool remove(const std::string &id, std::string &err);

    // From the page: a device asking to join. False with err (in plain words,
    // for the page) when the key isn't a WireGuard public key, is already a
    // device's or the PS5's, all addresses are taken, or another device is
    // waiting already. The request then waits for answer().
    bool ask(const std::string &name, const std::string &public_key, const std::string &from, WgRequest &out,
             std::string &err);
    // The waiting request, if there is one (they lapse after 10 minutes).
    bool waiting(WgRequest &out);
    // The TV's answer: yes adds the device, turned on. False with err if it
    // can't be saved.
    bool answer(bool yes, std::string &err);

    // The tunnel's settings with the devices that are on.
    wg::Config config(bool on, int port, int api_port) const;

private:
    bool save_locked(std::string &err);
    std::string free_address_locked() const;
    mutable std::mutex mu_;
    std::string path_, private_key_;
    std::vector<WgDevice> devices_;
    bool has_request_ = false;
    WgRequest request_;
};

}  // namespace app
