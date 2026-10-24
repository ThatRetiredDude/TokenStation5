// The Settings page: the title's settings, kept in ts5.env (settings.h), on
// the left; the chosen one's choices and what it does on the right. Up and
// down choose a setting, left and right change it, cross steps through its
// choices, square puts back its default. A key is typed on the console's
// keyboard (cross opens it, text_input.h) and only shows whether it's set;
// square removes it. On Speech model and Voice with no file on the PS5 yet,
// cross downloads one (downloads.h): Base (English), or Kokoro with the
// Heart voice; cross again stops the download. The APPS section on top
// shows how to set up an app, turns "Require a key" on and off, makes API
// keys (keys.h; cross names a new one, and it shows once on its row) and
// turns each key on and off (cross) or removes it (square twice).
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

// What the page shows besides the settings themselves.
struct SettingsInfo {
    std::vector<ApiKeyInfo> keys;
    std::string fresh_key, fresh_key_id;  // a key just made, shown on its row until the title forgets it
    std::string url;                      // the address apps use
    bool on_network = false;
    std::string model_id;  // the served model's id in the API ("" with none)
    int n_ctx = 0;         // its context
    std::vector<std::string> speech_models;  // ggml-*.bin in the models folder
    std::string speech_model;                // the one talking uses
    std::vector<std::string> voices;         // Kokoro's voices there (kokoro-voice-NAME.gguf), by NAME
    bool kokoro_model = false;               // Kokoro's model is there too
    std::map<std::string, app::Download> downloads;  // by catalog key
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
    // What square does: "Default", "Remove" (an API key or a typed key), or
    // "" when nothing.
    std::string square_does(const SettingsInfo &info) const;

private:
    const Font &h2_, &body_, &bodyb_, &small_, &smallb_;
    int sel_ = 0;
    int scroll_ = 0;
    double confirm_at_ = -1e9;  // square once on a key: press again to remove it
    std::string fresh_seen_;    // the fresh key the selection last moved to
};

}  // namespace ui
