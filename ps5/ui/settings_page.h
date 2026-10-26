// The Settings page: the title's settings, kept in ts5.env (settings.h), on
// the left; the chosen one's choices and what it does on the right. Up and
// down choose a setting, left and right change it, cross steps through its
// choices, square puts back its default. A key is typed on the console's
// keyboard (cross opens it, text_input.h) and only shows whether it's set;
// square removes it. On Speech model and Voice with no file on the PS5 yet,
// cross downloads one (downloads.h): Base (English), or Kokoro with the
// Heart voice; cross again stops the download. The APPS section on top
// has the instructions (cross opens their list on the right, up and down
// choose a guide, cross opens it and cross again goes back), turns "Require a key" on and off, makes API
// keys (keys.h; cross names a new one, and it shows once on its row) and
// turns each key on and off (cross) or removes it (square twice). The
// WIREGUARD section turns the tunnel on and off (wg/wg.h), approves (cross)
// or refuses (square) a device asking to join, shows the address a device
// asks from, turns each device on and off (cross) or removes it (square
// twice), shows this PS5's public key and makes it a new key pair (square
// twice), and sets the port.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "canvas.h"
#include "downloads.h"

namespace ui {

// An API key as the page shows it (keys.h's ApiKey without its hash).
struct ApiKeyInfo {
    std::string id, name;
    bool on = true;
    int64_t created = 0, used = 0;  // Unix seconds; used 0: never
};

// A WireGuard device as the page shows it (wg_devices.h, with what the
// tunnel says of it).
struct WgDeviceInfo {
    std::string id, name, address;  // address: its address in the tunnel
    bool on = true;
    int64_t added = 0;    // Unix seconds
    bool up = false;      // connected now
    double seen_s = -1;   // seconds since it last sent anything, -1: not this run
};

struct WgInfo {
    std::string public_key;  // the PS5's; "" until WireGuard is first turned on
    std::string problem;     // what's wrong with the tunnel, in plain words; "" if nothing
    std::vector<WgDeviceInfo> devices;
    bool waiting = false;  // a device asks to join
    std::string waiting_name, waiting_address, waiting_from;  // from: its address on the home network
    std::string home;            // the tunnel's endpoint host: Home address, or the PS5's public address
    bool home_looked_up = false;  // home is the public address the PS5 looked up
    // A phone or tablet just added on the TV: its whole tunnel (private key
    // and all) for the code, until the title forgets it.
    std::string tunnel, tunnel_name;
};

// What the page shows besides the settings themselves.
struct SettingsInfo {
    std::vector<ApiKeyInfo> keys;
    std::string fresh_key, fresh_key_id;  // a key just made, shown on its row until the title forgets it
    std::string url;                      // the address apps use
    bool on_network = false;
    std::string model_id;  // the served model's id in the API ("" with none)
    int n_ctx = 0;         // its context
    int api_port = 8081;   // the API's port, which WireGuard passes on
    std::vector<std::string> speech_models;  // ggml-*.bin in the models folder
    std::string speech_model;                // the one talking uses
    std::vector<std::string> voices;         // Kokoro's voices there (kokoro-voice-NAME.gguf), by NAME
    bool kokoro_model = false;               // Kokoro's model is there too
    std::map<std::string, app::Download> downloads;  // by catalog key
    WgInfo wg;
    std::string note;                        // the last change: saved, or why not
    bool note_failed = false;
    double note_at = -1e9;  // when (now's clock)
    double now = 0;
};

// A change to save: NAME=VALUE; an empty value puts back the default. Or,
// with type, the key NAME to type on the console's keyboard. Or downloads
// to start (or, with stop, to stop): catalog keys.
struct SettingChange {
    bool any = false, type = false;
    std::string name, value;
    std::string label;  // type: the setting's name on the page
    std::vector<std::string> download;
    bool stop = false;
    bool new_key = false;  // name a new API key on the console's keyboard
    std::string key_id;    // an API key to turn on or off (key_on), or to remove
    bool key_on = false, key_remove = false;
    std::string wg_device;  // a WireGuard device to turn on or off (wg_on), or to remove
    bool wg_on = false, wg_remove = false;
    bool wg_new_key = false;  // a new key pair for this PS5
    bool wg_phone = false;    // name a phone or tablet on the console's keyboard, to add it with keys made here
    int wg_answer = 0;        // the device asking to join: 1 let it in, -1 refuse it
};

class SettingsPage {
public:
    explicit SettingsPage(const Fonts &fonts);
    SettingChange press(uint32_t buttons, const SettingsInfo &info);
    void draw(Canvas &c, const SettingsInfo &info, int x, int y, int w, int h);
    // The panel on the right, for the chosen setting.
    void draw_details(Canvas &c, const SettingsInfo &info, int x, int y, int w, int h);
    // What cross does on the chosen setting, for the footer: "Next choice",
    // "Type", "Download" or "Stop the download"; "" when nothing.
    std::string cross_does(const SettingsInfo &info) const;
    // The left and right arrows' hint for the footer ("" when they do nothing).
    std::string arrows_do(const SettingsInfo &info) const;
    // A guide open in the instructions: up and down do nothing there.
    bool guide_open() const { return in_guide_ && open_ >= 0; }
    // What square does: "Default", "Remove" (an API key or a typed key), or
    // "" when nothing.
    std::string square_does(const SettingsInfo &info) const;

private:
    const Font &h2_, &body_, &bodyb_, &small_, &smallb_;
    int sel_ = 0;
    int scroll_ = 0;
    double confirm_at_ = -1e9;  // square once on a key: press again to remove it
    std::string fresh_seen_;    // the fresh key the selection last moved to
    std::string asking_seen_;   // the WireGuard request the selection last moved to
    bool in_guide_ = false;     // inside the Instructions row's panel
    int link_ = 0, open_ = -1;  // the guide chosen there, and the one open (-1: the list)
};

}  // namespace ui
