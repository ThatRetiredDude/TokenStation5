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
// turns each key on and off (cross) or removes it (square twice).
//
// AWAY FROM HOME has one row, WireGuard (wg/wg.h); cross opens its own list
// on the right, where left goes back. There, cross turns the tunnel on and
// off and steps the port; on the rest it opens the item: what WireGuard
// needs, a device asking to join (cross lets it in, square refuses it), adding
// a phone or tablet (cross names it, and its tunnel shows as a code) or a
// computer (a page in its browser), each device (cross turns it on and off,
// square twice removes it), this PS5's key (square twice makes a new pair) and
// the home address (cross types it). Left goes back to the list. A device
// asking to join opens its item wherever the page was.
//
// Tailscale, above it, turns on and off with cross; its whole setup is on
// the right: the code to scan to sign the PS5 in, then its tailnet name and
// the devices that reached it. Square twice signs the PS5 out.
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
    std::string added_id;  // the device last added on the TV, until Settings is left (its code may be gone)
};

// The PS5 on a tailnet (ts/tailscale.h), as the page shows it.
struct TsDeviceInfo {
    std::string name, address;  // its tailnet name and 100.x address
    bool up = false;            // has a session with the PS5 now
    double seen_s = -1;         // seconds since it last sent anything
};

struct TsInfo {
    // ts::Status::State: 0 off, 1 starting, 2 needs sign-in, 3 connecting,
    // 4 connected, 5 a problem.
    int state = 0;
    std::string sign_in_url;  // state 2: the page that approves the PS5
    std::string problem;      // what went wrong last, plain words; "" if nothing
    std::string name, address, account, relay;
    int devices = 0;          // on the tailnet besides the PS5
    std::vector<TsDeviceInfo> connected;  // the ones that have reached it since it started
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
    TsInfo ts;
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
    bool ts_sign_out = false;  // sign this PS5 out of its tailnet
};

struct Row;  // one row of the page (settings_page.cpp)

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
    // A guide open in the instructions, or an item of WireGuard's list: up
    // and down do nothing there.
    bool guide_open() const { return (in_guide_ && open_ >= 0) || (in_wg_ && wg_open_); }
    // What square does: "Default", "Remove" (an API key or a typed key), or
    // "" when nothing.
    std::string square_does(const SettingsInfo &info) const;

private:
    // Tailscale's panel, from ty down.
    void draw_tailscale(Canvas &c, const SettingsInfo &info, int dx, int &ty, int dw);
    SettingChange press_wg(uint32_t buttons, const SettingsInfo &info);
    SettingChange act(const Row &r, uint32_t buttons, const SettingsInfo &info);
    void follow_wg(const std::vector<Row> &rs, const SettingsInfo &info);
    void draw_wg(Canvas &c, const SettingsInfo &info, const std::vector<Row> &ws, int x, int y, int w, int h);
    void draw_panel(Canvas &c, const SettingsInfo &info, const Row &r, int x, int y, int w, int h);
    Row acting(const SettingsInfo &info, bool *listing) const;
    const Font &h2_, &body_, &bodyb_, &small_, &smallb_;
    int sel_ = 0;
    int scroll_ = 0;
    double confirm_at_ = -1e9;  // square once on a key: press again to remove it
    std::string fresh_seen_;    // the fresh key the selection last moved to
    std::string asking_seen_;   // the WireGuard request the selection last moved to
    bool in_guide_ = false;     // inside the Instructions row's panel
    int link_ = 0, open_ = -1;  // the guide chosen there, and the one open (-1: the list)
    bool in_wg_ = false;        // inside WireGuard's list, on the right
    bool wg_open_ = false;      // with its chosen item open
    std::string wg_sel_;        // that item (row_id), so it stays chosen as the list changes
    int wg_pos_ = 0, wg_scroll_ = 0;
};

}  // namespace ui
