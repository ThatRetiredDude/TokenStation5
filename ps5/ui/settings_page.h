// The Settings page: the title's settings, kept in ts5.env (settings.h), on
// the left; the chosen one's choices and what it does on the right. Up and
// down choose a setting, left and right change it, cross steps through its
// choices, square puts back its default. A key is typed on the console's
// keyboard (cross opens it, text_input.h) and only shows whether it's set;
// square removes it.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "canvas.h"

namespace ui {

// What the page shows besides the settings themselves.
struct SettingsInfo {
    std::vector<std::string> speech_models;  // ggml-*.bin in the models folder
    std::string speech_model;                // the one talking uses
    std::vector<std::string> voices;         // Kokoro's voices there (kokoro-voice-NAME.gguf), by NAME
    std::string note;                        // the last change: saved, or why not
    bool note_failed = false;
    double note_at = -1e9;  // when (now's clock)
    double now = 0;
};

// A change to save: NAME=VALUE; an empty value puts back the default. Or,
// with type, the key NAME to type on the console's keyboard.
struct SettingChange {
    bool any = false, type = false;
    std::string name, value;
    std::string label;  // type: the setting's name on the page
};

class SettingsPage {
public:
    explicit SettingsPage(const Fonts &fonts);
    SettingChange press(uint32_t buttons, const SettingsInfo &info);
    void draw(Canvas &c, const SettingsInfo &info, int x, int y, int w, int h);
    // The panel on the right, for the chosen setting.
    void draw_details(Canvas &c, const SettingsInfo &info, int x, int y, int w, int h);

private:
    const Font &h2_, &body_, &bodyb_, &small_, &smallb_;
    int sel_ = 0;
    int scroll_ = 0;
    double confirm_at_ = -1e9;  // square once on a key: press again to remove it
};

}  // namespace ui
