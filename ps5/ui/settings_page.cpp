#include "settings_page.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "library.h"
#include "local_time.h"
#include "pad.h"
#include "plain.h"
#include "qrcodegen.hpp"
#include "settings.h"
#include "style.h"

namespace ui {

namespace {

struct Choice {
    std::string value, shown;
};

// One setting: NAME in ts5.env, and what reads it does with it. No choices:
// a key, typed on the console's keyboard, or files in the models folder.
// The APPS section also has rows that aren't settings: the instructions,
// making a key, and a row per API key (name: its id). AWAY FROM HOME has
// Tailscale, a setting whose whole setup is its panel.
struct Row {
    std::string section, name, label, help, applies;
    std::vector<Choice> choices;
    std::string def;  // the value with nothing in ts5.env, as each reader's default
    bool files = false;
    enum Kind { Setting, Guide, NewKey, ApiKey, Tailscale } kind = Setting;
};

constexpr const char *kSpeechModel = "TS5_WHISPER_MODEL", *kVoice = "TS5_TTS_VOICE";
// Not ..._KEY: a name ending in _KEY is a secret, typed in (settings.h).
constexpr const char *kRequireKey = "TS5_KEYS_REQUIRED";
constexpr const char *kTailscale = "TS5_TAILSCALE";
// The voice with no TS5_TTS_VOICE, if it's there (tts/synth.h).
constexpr const char *kDefaultVoice = "af_heart";
constexpr int kRowH = 64, kHeadH = 44, kGap = 8;

// The guides the Instructions row links to, in order.
struct GuideLink {
    const char *label, *blurb, *help;
};
const GuideLink kGuides[] = {
    {"Set up an app", "Use the models from Kilo Code, Continue, Cline, Open WebUI and more.",
     "Any app that takes an OpenAI-compatible provider can use the models on this PS5. Give it these:"},
    {"At home and away with Tailscale", "This PS5 joins your tailnet itself. Nothing to set up on the router.",
     "Tailscale connects your devices into a private network of their own (a tailnet), wherever each one is. "
     "This PS5 joins it as a device of its own, so it works at home and away, even where no port can be opened."},
    {"Add models from a computer", "Copy models over your network with FTP, much faster than downloading here.",
     "Big downloads on the PS5 slow down after the first few hundred MB: the PS5 seems to limit how fast apps "
     "can save. FTP from a computer is much faster."},
};
constexpr int kGuideCount = sizeof kGuides / sizeof kGuides[0];
constexpr int kLinkH = 104;
constexpr double kNoteFor = 6, kConfirmFor = 4;

const std::vector<Row> &setting_rows() {
    static const std::vector<Row> r = {
        {"CHAT", "TS5_CHAT_MAX_TOKENS", "Longest reply",
         "How long a reply on the Chat page may run before it stops, in tokens (a token is about three "
         "quarters of a word).",
         "From the next reply",
         {{"512", "512 tokens"}, {"1024", "1,024 tokens"}, {"2048", "2,048 tokens"}, {"4096", "4,096 tokens"},
          {"8192", "8,192 tokens"}},
         "2048"},
        {"CHAT", "TAVILY_API_KEY", "Web search key",
         "With a key from Tavily, a model that calls tools can search the web, on the Chat page and from your "
         "apps. Each search uses one of the account's credits.",
         "From the next request", {}, ""},
        {"MODELS", "TS5_CTX_MAX", "Longest context",
         "How much text a model keeps in mind at once: the chat or the app's request, files included, and the "
         "reply. As much as fits suits most uses. A cap leaves the rest of the memory free and makes the model "
         "load a little faster.",
         "Now: the model loads again, which takes a few seconds",
         {{"", "As much as fits"}, {"4096", "4,096 tokens"}, {"8192", "8,192 tokens"}, {"16384", "16,384 tokens"},
          {"32768", "32,768 tokens"}, {"65536", "65,536 tokens"}},
         ""},
        {"MODELS", "TS5_KV", "Room for longer contexts",
         "Keeps what the model has read in half the memory, so about twice the context fits. It's used only when "
         "a model's whole context doesn't fit otherwise. Answers can come out very slightly different.",
         "Now: the model loads again, which takes a few seconds", {{"f16", "Off"}, {"q8", "On"}}, "f16"},
        {"VOICE", kSpeechModel, "Speech model",
         "The model that turns what you say into text when you hold R1 on the Chat page. Base (English) is a "
         "good start; Small (English) understands more and takes longer.",
         "The next time you hold R1", {}, "", true},
        {"VOICE", "TS5_SPEAK", "Read replies aloud",
         "Reads each reply on the Chat page out loud as it comes in, a sentence at a time, in a voice made on "
         "the PS5. Triangle stops it. The voice is loaded only while this is on, so off gives its memory back. "
         "It needs a voice: see Voice below.",
         "Now", {{"1", "On"}, {"0", "Off"}}, "1"},
        {"VOICE", "TS5_SPEAK_OUT", "Speak on",
         "Where the voice comes out: the TV, or the speaker in the controller.", "From the next sentence",
         {{"tv", "TV"}, {"controller", "Controller speaker"}}, "tv"},
        {"VOICE", kVoice, "Voice",
         "The voice that reads replies. Heart (US English) is the clearest; others include Michael (US "
         "English) and Emma (British English).",
         "From the next sentence", {}, "", true},
        {"VOICE", "TS5_TTS_SPEED", "Speaking speed", "How fast the voice talks.", "From the next sentence",
         {{"0.8", "Slower (0.8x)"}, {"0.9", "0.9x"}, {"1", "Normal"}, {"1.1", "1.1x"}, {"1.25", "1.25x"},
          {"1.5", "1.5x"}},
         "1"},
        {"CONSOLE", "TS5_KEEP_AWAKE", "Keep the console awake",
         "Keeps the console out of rest mode while TokenStation5 is open, as a video player does. Off: the "
         "console's own power saving settings decide.",
         "Within a minute", {{"1", "On"}, {"0", "Off"}}, "1"},
        {"CONSOLE", "TS5_DIM_AFTER", "Dark screen after",
         "With no button pressed for this long, the TV shows a dark screen instead, so the same picture doesn't "
         "sit on it for hours. Any button brings it back.",
         "Now",
         {{"120", "2 minutes"}, {"300", "5 minutes"}, {"600", "10 minutes"}, {"1200", "20 minutes"},
          {"1800", "30 minutes"}, {"3600", "1 hour"}, {"0", "Never"}},
         "600"},
        // Rarely touched: the defaults suit the PS5.
        {"TUNING", "TS5_CHAT_FPS", "Frames while a reply comes in",
         "How often the Chat page redraws while a reply streams in. Each frame is a copy on the GPU the model "
         "runs on, so fewer frames leave the model a little more speed.",
         "Now",
         {{"1", "1 a second"}, {"2", "2 a second"}, {"5", "5 a second"}, {"10", "10 a second"},
          {"15", "15 a second"}, {"30", "30 a second"}},
         "10"},
        {"TUNING", "TS5_WHISPER_THREADS", "Speech threads",
         "CPU cores that turn what you said into text. More is faster while nothing else runs on the CPU.",
         "The next time you hold R1", {{"4", "4"}, {"6", "6"}, {"8", "8"}, {"10", "10"}, {"12", "12"}}, "6"},
        {"TUNING", "TS5_TTS_THREADS", "Voice threads",
         "CPU cores that make the voice. More is faster while nothing else runs on the CPU.",
         "From the next sentence", {{"4", "4"}, {"6", "6"}, {"8", "8"}, {"10", "10"}, {"12", "12"}}, "6"},
        {"TUNING", "TS5_NET_LOG", "Detailed log",
         "Adds every step of each download and web search to the log, for tracking down a problem. Off keeps "
         "the log short: what happened, problems, and a line for each request.",
         "Now", {{"0", "Off"}, {"1", "On"}}, "0"},
    };
    return r;
}

const ApiKeyInfo *key_of(const Row &r, const SettingsInfo &info) {
    if (r.kind != Row::ApiKey) return nullptr;
    for (const ApiKeyInfo &k : info.keys)
        if (k.id == r.name) return &k;
    return nullptr;
}

// The page's rows: APPS (the instructions, the key setting, making a key and
// the keys), AWAY FROM HOME (Tailscale), then the settings: CHAT, MODELS,
// VOICE, CONSOLE, and TUNING last, for what's rarely changed.
std::vector<Row> rows(const SettingsInfo &info) {
    std::vector<Row> r;
    r.push_back({"APPS", "", "Instructions", "How to use the models on this PS5 from your apps, at home and away, and how to add more.",
                 "", {}, "", false, Row::Guide});
    r.push_back({"APPS", kRequireKey, "Require a key",
                 "With this on, apps on your network need one of your keys below to use the models. The Chat page "
                 "on this PS5 never needs one. Off: anyone on your network can, and an app that sends a key still "
                 "shows by its name on Home.",
                 "From the next request", {{"1", "On"}, {"0", "Off"}}, "0"});
    r.push_back({"APPS", "", "New key",
                 "Make a key for each app, named after it, such as \"Kilo Code on the laptop\". The key shows once, "
                 "right after you make it: type it into the app then.",
                 "", {}, "", false, Row::NewKey});
    for (const ApiKeyInfo &k : info.keys)
        r.push_back({"APPS", k.id, k.name.empty() ? "Key " + k.id : k.name,
                     "An app sends this key to use the models. Turn it off to shut that app out without removing "
                     "the key.",
                     "From the next request", {{"1", "On"}, {"0", "Off"}}, "", false, Row::ApiKey});
    r.push_back({"AWAY FROM HOME", kTailscale, "Tailscale",
                 "Reach this PS5 from your phone, tablet or computer, at home or away, through Tailscale. This PS5 "
                 "joins your tailnet as a device of its own: nothing to set up on your router, and only your "
                 "tailnet's devices can reach it.",
                 "", {{"1", "On"}, {"0", "Off"}}, "0", false, Row::Tailscale});
    const auto &s = setting_rows();
    r.insert(r.end(), s.begin(), s.end());
    return r;
}

// "24 Oct 2026", in the console's own time zone; "Never" for 0.
std::string date_of(int64_t t) {
    if (t <= 0) return "Never";
    static const char *months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    struct tm g;
    if (!app::local_tm(t, &g)) return "";
    return std::to_string(g.tm_mday) + " " + months[g.tm_mon % 12] + " " + std::to_string(g.tm_year + 1900);
}

// "Just now", "12 minutes ago"; s < 0: not since the title opened.
std::string ago(double s) {
    if (s < 0) return "Not since TokenStation5 opened";
    if (s < 90) return "Just now";
    if (s < 90 * 60) return std::to_string((int)(s / 60 + 0.5)) + " minutes ago";
    const int h = (int)(s / 3600 + 0.5);
    return std::to_string(h) + (h == 1 ? " hour ago" : " hours ago");
}

// TEXT as a QR code, dark on white with its quiet zone, at most SIZE pixels
// square at (x, y); returns the side it drew (0 if TEXT doesn't fit one).
int draw_qr(Canvas &c, int x, int y, const std::string &text, int size) {
    static std::string last;
    static std::vector<bool> dark;  // the last code's modules, row by row
    static int n = 0;
    if (text != last) {
        last = text;
        dark.clear();
        n = 0;
        try {
            const auto qr = qrcodegen::QrCode::encodeText(text.c_str(), qrcodegen::QrCode::Ecc::MEDIUM);
            n = qr.getSize();
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++) dark.push_back(qr.getModule(i, j));
        } catch (...) {
        }
    }
    if (n == 0) return 0;
    const int m = size / (n + 8);  // 4 modules of quiet zone each side
    if (m < 1) return 0;
    const int side = m * (n + 8);
    c.fill(x, y, side, side, rgb(0xFFFFFF));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (dark[(size_t)j * n + i]) c.fill(x + (i + 4) * m, y + (j + 4) * m, m, m, rgb(0x000000));
    return side;
}

bool is_key(const Row &r) { return r.kind == Row::Setting && app::secret_setting(r.name); }

// For a row of files, what its default picks; "" if there are none.
std::string default_file(const Row &r, const SettingsInfo &info) {
    const auto &files = std::string(r.name) == kVoice ? info.voices : info.speech_models;
    if (files.empty()) return "";
    if (std::string(r.name) == kVoice && std::find(files.begin(), files.end(), kDefaultVoice) != files.end())
        return kDefaultVoice;
    return files.front();
}

// A file's name on the page: a speech model's or a voice's.
std::string file_name(const Row &r, const std::string &file) {
    return std::string(r.name) == kVoice ? plain::voice_name(file) : plain::speech_name(file);
}

// The row's choices; a row of files has the files there are.
std::vector<Choice> choices(const Row &r, const SettingsInfo &info) {
    if (!r.files) return r.choices;
    const auto &files = std::string(r.name) == kVoice ? info.voices : info.speech_models;
    std::vector<Choice> c;
    for (const std::string &m : files) c.push_back({m, file_name(r, m)});
    return c;
}

std::string current(const Row &r) { return app::setting(r.name.c_str(), r.def); }

// On Speech model or Voice: the catalog files (catalog.h) the row needs on
// the PS5 before it has something to use. Voice needs Kokoro's model and a
// voice; Heart is the one it downloads.
std::vector<std::string> missing(const Row &r, const SettingsInfo &info) {
    std::vector<std::string> keys;
    if (!r.files) return keys;
    if (std::string(r.name) == kSpeechModel) {
        if (info.speech_models.empty()) keys.push_back("speech-base-en");
        return keys;
    }
    if (!info.kokoro_model) keys.push_back("kokoro");
    if (info.voices.empty()) keys.push_back("voice-af-heart");
    return keys;
}

// What cross downloads there: "Base (English), 148 MB", "the Heart voice,
// 171 MB" (with Kokoro's model), "Kokoro's model, 170 MB".
std::string download_what(const std::vector<std::string> &keys, const SettingsInfo &info) {
    const app::Download d = app::joined(keys, info.downloads);
    std::string what;
    if (keys.size() == 1 && keys[0] == "speech-base-en") what = "Base (English)";
    else if (std::find(keys.begin(), keys.end(), "voice-af-heart") != keys.end()) what = "the Heart voice";
    else what = "Kokoro's model";
    return what + ", " + app::format_bytes(d.total);
}

// What the row is set to in effect: for a row of files left unset, the file
// used by default.
std::string in_effect(const Row &r, const SettingsInfo &info) {
    const std::string v = current(r);
    return r.files && v.empty() ? default_file(r, info) : v;
}

// The choice marked "default": the row's default, or for a row of files the
// file used when nothing is chosen.
bool is_default(const Row &r, const SettingsInfo &info, const Choice &ch) {
    return r.files ? ch.value == default_file(r, info) : ch.value == r.def;
}

// The current value's choice, or -1 (a value set some other way).
int index_of(const std::vector<Choice> &cs, const std::string &v) {
    for (size_t i = 0; i < cs.size(); i++)
        if (cs[i].value == v) return (int)i;
    return -1;
}

// A value that isn't one of the row's choices (set from a computer), by name.
std::string other(const Row &r) {
    if (!r.files) return current(r);
    return std::string(r.name) == kVoice ? "A voice not on your PS5" : "A speech model not on your PS5";
}

std::string shown(const Row &r, const SettingsInfo &info) {
    if (r.kind == Row::Guide) return "Open";
    if (r.kind == Row::NewKey) return "Make one";
    if (r.kind == Row::Tailscale && current(r) == "1") {
        switch (info.ts.state) {
            case 2: return "Sign in";
            case 4: return "Connected";
            case 5: return "Trying again";
            default: return "Connecting";
        }
    }
    if (const ApiKeyInfo *k = key_of(r, info)) return k->on ? "On" : "Off";
    if (is_key(r)) return current(r).empty() ? "Not set" : "Set";
    const auto cs = choices(r, info);
    if (r.files && cs.empty()) {
        const app::Download d = app::joined(missing(r, info), info.downloads);
        switch (d.phase) {
            case app::Download::Running:
                return d.held ? "Waiting" : "Downloading " + std::to_string(app::percent(d)) + "%";
            case app::Download::Queued:
            case app::Download::Retrying: return "Waiting";
            case app::Download::Stopped: return "Stopped at " + std::to_string(app::percent(d)) + "%";
            case app::Download::Failed: return "Didn't download";
            case app::Download::None: break;
        }
        return "None yet";
    }
    const int i = index_of(cs, in_effect(r, info));
    if (i < 0) return other(r);
    return cs[i].shown;
}

// Keeps sel in range, and moves it to a key just made (once), so its row
// shows the key.
void follow(const std::vector<Row> &rs, const SettingsInfo &info, int &sel, std::string &seen) {
    if (info.fresh_key_id != seen) {
        seen = info.fresh_key_id;
        for (size_t i = 0; i < rs.size(); i++)
            if (rs[i].kind == Row::ApiKey && rs[i].name == seen) sel = (int)i;
    }
    sel = std::max(0, std::min(sel, (int)rs.size() - 1));
}

std::vector<std::string> wrap(const Font &f, const std::string &text, int max_w) {
    std::vector<std::string> lines;
    std::string line;
    size_t i = 0;
    while (i < text.size()) {
        size_t e = text.find(' ', i);
        if (e == std::string::npos) e = text.size();
        const std::string word = text.substr(i, e - i);
        const std::string tryline = line.empty() ? word : line + " " + word;
        if (!line.empty() && Canvas::measure(f, tryline) > max_w) {
            lines.push_back(line);
            line = word;
        } else {
            line = tryline;
        }
        i = e + 1;
    }
    if (!line.empty()) lines.push_back(line);
    return lines;
}

}  // namespace

SettingsPage::SettingsPage(const Fonts &fonts)
    : h2_(fonts.get("h2")), body_(fonts.get("body")), bodyb_(fonts.get("bodyb")), small_(fonts.get("small")),
      smallb_(fonts.get("smallb")) {}

SettingChange SettingsPage::press(uint32_t b, const SettingsInfo &info) {
    SettingChange ch;
    const auto rs = rows(info);
    follow(rs, info, sel_, fresh_seen_);
    if (rs[sel_].kind != Row::Guide) in_guide_ = false;
    if (in_guide_) {
        // In the instructions: up and down choose a guide, cross or right
        // opens it; on a guide, cross or left goes back to the list, and left
        // on the list back to the rows.
        if (open_ >= 0) {
            if (b & (kCross | kLeft)) open_ = -1;
        } else if (b & (kCross | kRight)) {
            open_ = link_;
        } else if (b & kLeft) {
            in_guide_ = false;
        } else {
            if (b & kUp) link_ = std::max(0, link_ - 1);
            if (b & kDown) link_ = std::min(kGuideCount - 1, link_ + 1);
        }
        return ch;
    }
    if (b & kUp) sel_ = std::max(0, sel_ - 1);
    if (b & kDown) sel_ = std::min((int)rs.size() - 1, sel_ + 1);
    if (b & (kUp | kDown)) confirm_at_ = -1e9;
    const Row &r = rs[sel_];
    if (r.kind == Row::Guide) {
        if (b & (kCross | kRight)) {
            in_guide_ = true;
            open_ = -1;
        }
        return ch;
    }
    if (r.kind == Row::NewKey) {
        ch.new_key = (b & kCross) != 0;
        return ch;
    }
    if (const ApiKeyInfo *k = key_of(r, info)) {
        // On is the first choice, Off the second.
        const bool want = (b & kCross) ? !k->on : (b & kLeft) ? true : (b & kRight) ? false : k->on;
        if (want != k->on) {
            ch.key_id = k->id;
            ch.key_on = want;
        } else if (b & kSquare) {
            if (info.now - confirm_at_ < kConfirmFor) {
                ch.key_id = k->id;
                ch.key_remove = true;
                confirm_at_ = -1e9;
                // The row above stays chosen (another key, or New key): the
                // removed key's place would go to the next section's row.
                sel_ = std::max(0, sel_ - 1);
            } else {
                confirm_at_ = info.now;
            }
        }
        return ch;
    }
    if (r.kind == Row::Tailscale && (b & kSquare)) {
        // Square signs it out (twice); it puts back no default.
        if (!info.ts.account.empty() || !info.ts.address.empty()) {
            if (info.now - confirm_at_ < kConfirmFor) {
                ch.ts_sign_out = true;
                confirm_at_ = -1e9;
            } else {
                confirm_at_ = info.now;
            }
        }
        return ch;
    }
    auto set = [&](const std::string &v) {
        if (v == current(r) || (v == in_effect(r, info) && v != r.def)) return;
        ch.any = true;
        ch.name = r.name;
        // The default goes back to nothing in ts5.env.
        ch.value = v == r.def ? "" : v;
    };
    if (is_key(r)) {
        if (b & kCross) {
            ch.type = true;
            ch.name = r.name;
            ch.label = r.label;
            return ch;
        }
        if ((b & kSquare) && !current(r).empty()) {
            if (info.now - confirm_at_ < kConfirmFor) {
                set("");
                confirm_at_ = -1e9;
            } else {
                confirm_at_ = info.now;
            }
        }
        return ch;
    }
    // Nothing there to choose yet: cross downloads it, or stops that.
    const auto keys = missing(r, info);
    if (!keys.empty() && (b & kCross)) {
        ch.download = keys;
        ch.stop = app::joined(keys, info.downloads).active();
        return ch;
    }
    const auto cs = choices(r, info);
    if (cs.empty()) return ch;
    int i = index_of(cs, in_effect(r, info));
    if (i < 0) i = std::max(0, index_of(cs, r.def));
    if (b & kLeft) set(cs[std::max(0, i - 1)].value);
    else if (b & kRight) set(cs[std::min((int)cs.size() - 1, i + 1)].value);
    else if (b & kCross) set(cs[(i + 1) % cs.size()].value);
    else if (b & kSquare) set(r.def);
    return ch;
}

void SettingsPage::draw(Canvas &c, const SettingsInfo &info, int x, int y, int w, int h) {
    c.text(h2_, x + 32, y + 50, "Settings", kText);
    const auto rs = rows(info);
    follow(rs, info, sel_, fresh_seen_);

    // Rows under their sections, scrolled so the chosen one shows.
    struct Place {
        int y;
        bool head;
        int row;
    };
    std::vector<Place> places;
    int py = 0, sel_top = 0, sel_bottom = 0;
    for (size_t i = 0; i < rs.size(); i++) {
        if (i == 0 || std::string(rs[i].section) != rs[i - 1].section) {
            places.push_back({py, true, (int)i});
            py += kHeadH;
        }
        if ((int)i == sel_) sel_top = py - (i == 0 || std::string(rs[i].section) != rs[i - 1].section ? kHeadH : 0);
        places.push_back({py, false, (int)i});
        if ((int)i == sel_) sel_bottom = py + kRowH;
        py += kRowH + kGap;
    }
    const int lx = x + 16, lw = w - 32, top = y + 76, view_h = y + h - 16 - top;
    if (sel_top - scroll_ < 0) scroll_ = sel_top;
    if (sel_bottom - scroll_ > view_h) scroll_ = sel_bottom - view_h;
    scroll_ = std::max(0, std::min(scroll_, std::max(0, py - kGap - view_h)));

    c.clip(lx - 4, top, lw + 8, view_h);
    for (const Place &p : places) {
        const int ry = top + p.y - scroll_;
        if (ry > top + view_h || ry + kRowH < top) continue;
        const Row &r = rs[p.row];
        if (p.head) {
            c.text(smallb_, lx + 8, ry + 32, r.section, kMuted);
            continue;
        }
        const bool sel = p.row == sel_;
        c.round_rect((float)lx, (float)ry, (float)lw, (float)kRowH, 14, sel ? kRowSel : kRow);
        if (sel) c.round_rect_line((float)lx, (float)ry, (float)lw, (float)kRowH, 14, 2.5f, kAccent);
        const std::string v = shown(r, info);
        const bool changeable = !is_key(r) && !choices(r, info).empty();
        int right = lx + lw - 24;
        if (sel && changeable) {
            // The arrows that change it.
            const int cy = ry + kRowH / 2;
            c.segment((float)right - 4, (float)cy, (float)right - 12, (float)cy - 8, 2.5f, kAccent);
            c.segment((float)right - 4, (float)cy, (float)right - 12, (float)cy + 8, 2.5f, kAccent);
            right -= 28;
        }
        const Color vc = is_key(r) ? (current(r).empty() ? kText2 : kGreen) : sel ? kAccent : kText2;
        const int vx = c.text_right(bodyb_, right, ry + 42, v, vc);
        if (sel && changeable) {
            const int ax = vx - 16, cy = ry + kRowH / 2;
            c.segment((float)ax, (float)cy, (float)ax + 8, (float)cy - 8, 2.5f, kAccent);
            c.segment((float)ax, (float)cy, (float)ax + 8, (float)cy + 8, 2.5f, kAccent);
        }
        c.text(bodyb_, lx + 24, ry + 42, r.label, kText, vx - 48 - (lx + 24));
    }
    c.unclip();
}

void SettingsPage::draw_details(Canvas &c, const SettingsInfo &info, int x, int y, int w, int h) {
    const auto rs = rows(info);
    const Row &r = rs[std::max(0, std::min(sel_, (int)rs.size() - 1))];
    const int dx = x + 32, dw = w - 64;
    const int open = r.kind == Row::Guide && in_guide_ ? open_ : -1;  // the guide on show, if any
    c.text(h2_, dx, y + 52, open >= 0 ? kGuides[open].label : r.label, kText, dw);
    int ty = y + 100;
    for (const std::string &l : wrap(body_, open >= 0 ? kGuides[open].help : r.help, dw)) {
        c.text(body_, dx, ty, l, kText2);
        ty += 38;
    }
    ty += 24;
    const std::string cur = current(r);
    // A label and its value, one under the other.
    auto field = [&](const char *label, const std::string &value, Color col, const Font &f) {
        c.text(smallb_, dx, ty, label, kMuted);
        c.text(f, dx, ty + 38, value, col, dw);
        ty += 84;
    };
    auto para = [&](const std::string &text, Color col) {
        for (const std::string &l : wrap(small_, text, dw)) {
            c.text(small_, dx, ty, l, col);
            ty += 32;
        }
    };

    if (r.kind == Row::Guide && open < 0) {
        // The guides, as links.
        for (int i = 0; i < kGuideCount; i++) {
            const bool on = in_guide_ && i == link_;
            c.round_rect((float)dx, (float)ty, (float)dw, (float)kLinkH, 14, on ? kRowSel : kRow);
            if (on) c.round_rect_line((float)dx, (float)ty, (float)dw, (float)kLinkH, 14, 2.5f, kAccent);
            const int cx = dx + dw - 28, cy = ty + kLinkH / 2;
            c.segment((float)cx, (float)cy, (float)cx - 9, (float)cy - 9, 2.5f, on ? kAccent : kText2);
            c.segment((float)cx, (float)cy, (float)cx - 9, (float)cy + 9, 2.5f, on ? kAccent : kText2);
            c.text(bodyb_, dx + 24, ty + 44, kGuides[i].label, on ? kAccent : kText, dw - 96);
            c.text(small_, dx + 24, ty + 80, kGuides[i].blurb, kText2, dw - 96);
            ty += kLinkH + kGap * 2;
        }
        ty += 8;
        if (!in_guide_) text_glyph(c, small_, dx, ty + 8, "Press", kGlyphCross, "to choose one.", kText2);
    } else if (r.kind == Row::Guide) {
        // Back to the list, at the foot of the panel.
        const int bw = std::min(dw, 460), bh = 72, by = y + h - bh - 32;
        c.round_rect((float)dx, (float)by, (float)bw, (float)bh, 14, kRowSel);
        c.round_rect_line((float)dx, (float)by, (float)bw, (float)bh, 14, 2.5f, kAccent);
        text_glyph(c, bodyb_, dx + 24, by + 46, "", kGlyphCross, "Back to instructions", kText);
    }
    // The PS5's address alone, from http://ADDRESS:8081/v1.
    std::string host = info.url;
    if (host.rfind("http://", 0) == 0) host = host.substr(7);
    host = host.substr(0, host.find(':'));
    if (open == 0) {
        ty -= 12;
        if (info.on_network) field("BASE URL", info.url, kAccent, bodyb_);
        else field("BASE URL", "Connect the PS5 to your network first", kAmber, body_);
        if (!info.model_id.empty()) field("MODEL", info.model_id, kText, bodyb_);
        else field("MODEL", "Load a model on the Models page first", kAmber, body_);
        field("API KEY",
              app::setting(kRequireKey, "0") == "1" ? "One of your keys, under Apps on the left"
                                                    : "Any text, or one of your keys under Apps on the left",
              kText, body_);
        if (info.n_ctx > 0)
            field("CONTEXT WINDOW", plain::count(info.n_ctx) + " tokens", kText, body_);
        para("In Kilo Code, add a provider of the type OpenAI Compatible with these. Use the address as it is, "
             "starting http and ending /v1. Where the app asks for a context window, give it the number above, "
             "so it shortens long conversations before the PS5 has to refuse them. For coding agents that edit "
             "files and run commands, Qwen 3 8B does best: Qwen 2.5 Coder 7B writes the code but often stops "
             "before the next step.",
             kText2);
    } else if (open == 1) {
        ty -= 12;
        para("1. Install Tailscale (free, tailscale.com) on your phone, tablet or computer, and sign in.", kText);
        ty += 8;
        para("2. On the left, under Away from home, turn on Tailscale. A code shows: scan it with your phone, sign "
             "in to the same Tailscale account, and approve TokenStation5.",
             kText);
        ty += 8;
        para("3. With Tailscale on, apps use http://" + (info.ts.address.empty() ? std::string("ADDRESS") : info.ts.address) +
                 ":" + std::to_string(info.api_port) +
                 "/v1 and one of your keys, at home and away. The ADDRESS is the PS5's on your tailnet, shown under "
                 "Tailscale once it's on.",
             kText);
        ty += 8;
        para("Over Tailscale a key is always needed. Devices at home connect directly; elsewhere Tailscale's relays "
             "carry the traffic, which works even where no port can be opened. TokenStation5 has to be open on the "
             "PS5.",
             kText2);
    } else if (open == 2) {
        ty -= 12;
        // ftpsrv, which the jailbreak runs, listens on 2121.
        if (info.on_network) field("FTP ADDRESS", "ftp://" + host + ":2121", kAccent, bodyb_);
        else field("FTP ADDRESS", "Connect the PS5 to your network first", kAmber, body_);
        field("FOLDER", "/data/homebrew/PPSA99555/models", kText, bodyb_);
        para("1. On your computer, download the model's .gguf file, from Hugging Face for example. For 7B and "
             "8B models, Q4_K_M is a good choice.",
             kText);
        ty += 8;
        para("2. Connect an FTP app, such as FileZilla or Cyberduck, to the address above, with no user name "
             "or password.",
             kText);
        ty += 8;
        para("3. Copy the file into the folder above. The Models page lists it within seconds, as COPYING "
             "until the whole file is there.",
             kText);
        ty += 8;
        para("Keep a copy on your computer: updating or deleting the app removes its models.", kText2);
    } else if (r.kind == Row::Guide) {
        // The list of guides, drawn above.
    } else if (r.kind == Row::Tailscale) {
        // Its last lines stay above the bottom line, where a note ("Saved",
        // "Removed") shows.
        draw_tailscale(c, info, dx, ty, dw, y + h - 120);
    } else if (r.kind == Row::NewKey) {
        text_glyph(c, body_, dx, ty + 8, "Press", kGlyphCross, "to name a new key.", kText2);
        ty += 56;
        if (info.keys.empty()) para("You have no keys yet.", kMuted);
        else para("You have " + std::to_string(info.keys.size()) + (info.keys.size() == 1 ? " key." : " keys."), kMuted);
    } else if (const ApiKeyInfo *k = key_of(r, info)) {
        if (k->id == info.fresh_key_id && !info.fresh_key.empty()) {
            field("YOUR KEY", info.fresh_key, kAccent, h2_);
            para("Type it into the app now: it won't show again. Lost it? Make a new one and remove this one.",
                 kAmber);
            ty += 16;
        }
        field("NOW", k->on ? "On" : "Off", k->on ? kGreen : kText2, bodyb_);
        c.text(smallb_, dx, ty, "MADE", kMuted);
        c.text(smallb_, dx + dw / 2, ty, "LAST USED", kMuted);
        c.text(body_, dx, ty + 38, date_of(k->created), kText2, dw / 2 - 16);
        c.text(body_, dx + dw / 2, ty + 38, date_of(k->used), kText2, dw / 2);
        ty += 92;
        text_glyph(c, small_, dx, ty, "Press", kGlyphCross, k->on ? "to turn it off." : "to turn it on.", kText2);
        ty += 40;
        const bool confirming = info.now - confirm_at_ < kConfirmFor;
        text_glyph(c, small_, dx, ty, confirming ? "Press" : "", kGlyphSquare,
                   confirming ? "again to remove the key" : "Remove the key", confirming ? kAmber : kText2);
        ty += 40;
    } else if (is_key(r)) {
        c.text(smallb_, dx, ty, "NOW", kMuted);
        c.text(bodyb_, dx, ty + 38, cur.empty() ? "Not set" : "Set", cur.empty() ? kText2 : kGreen);
        ty += 84;
        c.text(smallb_, dx, ty, "TO SET IT", kMuted);
        text_glyph(c, body_, dx, ty + 38, "Press", kGlyphCross,
                   "and type it, with the controller or a keyboard.", kText2);
        c.text(small_, dx, ty + 84, "Get a key at app.tavily.com. It stays on your PS5.", kMuted, dw);
        ty += 120;
        if (!cur.empty()) {
            const bool confirming = info.now - confirm_at_ < kConfirmFor;
            text_glyph(c, small_, dx, ty + 8, confirming ? "Press" : "", kGlyphSquare,
                       confirming ? "again to remove the key" : "Remove the key", confirming ? kAmber : kText2);
            ty += 40;
        }
    } else {
        const auto cs = choices(r, info);
        const auto keys = missing(r, info);
        const bool fetching = !keys.empty() && app::joined(keys, info.downloads).phase != app::Download::None;
        c.text(smallb_, dx, ty, cs.empty() && fetching ? "DOWNLOAD" : "CHOICES", kMuted);
        ty += 20;
        if (!keys.empty()) {
            const app::Download d = app::joined(keys, info.downloads);
            using D = app::Download;
            const std::string head = cs.empty() ? "None on your PS5 yet." : "Kokoro's model isn't on your PS5 yet.";
            if (d.phase == D::None) {
                c.text(body_, dx, ty + 30, head, kText2, dw);
                text_glyph(c, body_, dx, ty + 68, "Press", kGlyphCross, "to download " + download_what(keys, info) + ".",
                           kText2);
                ty += 100;
            } else {
                c.round_rect((float)dx, (float)ty + 14, (float)dw, 12, 6, kTrack);
                const float f = d.total ? std::min(1.0f, (float)d.done / (float)d.total) : 0;
                const Color mc = d.phase == D::Failed ? kRed : d.phase == D::Running && !d.held ? kAccent : kText2;
                if (f > 0) c.round_rect((float)dx, (float)ty + 14, std::max(12.0f, dw * f), 12, 6, mc);
                c.text(small_, dx, ty + 58, std::to_string(app::percent(d)) + "%  \xC2\xB7  " + app::progress_text(d, true),
                       kText2, dw);
                ty += 94;
                auto para = [&](const std::string &text, Color col) {
                    for (const std::string &l : wrap(small_, text, dw)) {
                        c.text(small_, dx, ty, l, col);
                        ty += 32;
                    }
                };
                switch (d.phase) {
                    case D::Running:
                        para(d.held ? "It waits while a model loads, then goes on by itself."
                                    : "It's ready to use when the download finishes.",
                             kText2);
                        break;
                    case D::Queued: para("It starts when one of the downloads running now ends.", kText2); break;
                    case D::Retrying:
                        para(d.problem + " Trying again in " +
                                 std::to_string(std::max(1, (int)std::ceil(d.retry_at - info.now))) + " s.",
                             kAmber);
                        break;
                    case D::Stopped:
                        text_glyph(c, small_, dx, ty, "Stopped. Press", kGlyphCross, "to go on.", kText2);
                        ty += 32;
                        break;
                    case D::Failed: para(d.problem, kRed); break;
                    case D::None: break;
                }
                ty += 8;
            }
        }
        const int at = index_of(cs, in_effect(r, info));
        for (size_t i = 0; i < cs.size(); i++) {
            const int cy = ty + 26;
            const bool on = (int)i == at;
            c.ring((float)dx + 12, (float)cy - 8, 11, 2.5f, on ? kAccent : kMuted);
            if (on) c.circle((float)dx + 12, (float)cy - 8, 5.5f, kAccent);
            const int tw = c.text(body_, dx + 40, cy, cs[i].shown, on ? kText : kText2, dw - 40);
            if (is_default(r, info, cs[i])) c.text(small_, dx + 40 + tw + 14, cy, "default", kMuted);
            ty += 42;
        }
        if (at < 0 && !cs.empty()) {
            c.text(small_, dx + 40, ty + 26, other(r) + " (set from a computer)", kAmber, dw - 40);
            ty += 42;
        }
        ty += 20;
    }

    // Bottom: when a change counts.
    const int by = y + h - 90;
    if (!r.applies.empty()) {
        c.text(smallb_, dx, by, "TAKES EFFECT", kMuted);
        c.text(body_, dx, by + 36, r.applies, kText2, dw);
    }
    if (info.now - info.note_at < kNoteFor && !info.note.empty())
        c.text_right(smallb_, dx + dw, by, info.note, info.note_failed ? kRed : kGreen);
    (void)ty;
}

void SettingsPage::draw_tailscale(Canvas &c, const SettingsInfo &info, int dx, int &ty, int dw, int bottom) {
    auto field = [&](const char *label, const std::string &value, Color col, const Font &f) {
        c.text(smallb_, dx, ty, label, kMuted);
        c.text(f, dx, ty + 38, value, col, dw);
        ty += 84;
    };
    auto para = [&](const std::string &text, Color col) {
        for (const std::string &l : wrap(small_, text, dw)) {
            c.text(small_, dx, ty, l, col);
            ty += 32;
        }
    };
    const TsInfo &t = info.ts;
    const std::string port = std::to_string(info.api_port);
    const std::string short_name = t.name.empty() ? "tokenstation5" : t.name.substr(0, t.name.find('.'));
    if (app::setting(kTailscale, "0") != "1") {
        field("NOW", "Off", kText2, bodyb_);
        para("1. Install Tailscale (free, tailscale.com) on your phone, tablet or computer, and sign in.", kText);
        ty += 8;
        text_glyph(c, small_, dx, ty, "2. Press", kGlyphCross, "here to turn it on. A code shows here:", kText);
        ty += 32;
        para("scan it with your phone and sign in to the same Tailscale account, to add this PS5.", kText);
        ty += 8;
        para("3. Apps then use http://" + (t.address.empty() ? std::string("ADDRESS") : t.address) + ":" + port +
                 "/v1 with one of your keys, at home or away: the PS5's address on your tailnet shows here once "
                 "it's on.",
             kText);
        ty += 16;
        para("Over Tailscale a key is always needed. Devices at home connect directly; elsewhere Tailscale's relays "
             "carry the traffic, so it works even where no port can be opened.",
             kText2);
        return;
    }
    if (t.state == 2 && !t.sign_in_url.empty()) {
        // The code to scan, with what to do beside it.
        field("SCAN THIS TO ADD THIS PS5", "", kAccent, bodyb_);
        ty -= 40;
        const int side = draw_qr(c, dx, ty, t.sign_in_url, 360);
        const int sx = dx + side + 32, sw = dx + dw - sx;
        int sy = ty + 4;
        auto step = [&](const std::string &text, Color col) {
            for (const std::string &l : wrap(small_, text, sw)) {
                c.text(small_, sx, sy, l, col);
                sy += 32;
            }
            sy += 8;
        };
        step("1. Scan it with your phone's camera.", kText);
        step("2. Sign in to Tailscale, with the account your other devices use.", kText);
        step("3. Approve TokenStation5. This page shows it on your tailnet then.", kText);
        step("No camera handy? Open this on any device:", kText2);
        for (const std::string &l : wrap(small_, t.sign_in_url, sw)) {
            c.text(small_, sx, sy, l, kAccent);
            sy += 32;
        }
        ty = std::max(sy, ty + side + 24);
        if (!t.problem.empty()) para(t.problem, kAmber);
        return;
    }
    if (t.state == 2) {
        field("NOW", "Waiting for approval", kAmber, bodyb_);
        para(t.problem.empty() ? "Your tailnet's admin has to approve this PS5 in Tailscale's admin page, under Machines."
                               : t.problem,
             kText);
        return;
    }
    if (t.state != 4) {
        field("NOW", t.state == 5 ? "Not connected: trying again" : "Connecting to Tailscale", t.state == 5 ? kAmber : kText,
              bodyb_);
        if (!t.problem.empty()) para(t.problem, t.state == 5 ? kRed : kText2);
        ty += 8;
        para(info.on_network ? "It goes on trying by itself." : "Connect the PS5 to your network first.", kText2);
        return;
    }
    field("ON YOUR TAILNET AS", t.name.empty() ? short_name : t.name, kAccent, bodyb_);
    c.text(smallb_, dx, ty, "ADDRESS", kMuted);
    c.text(smallb_, dx + dw / 2, ty, "SIGNED IN AS", kMuted);
    c.text(body_, dx, ty + 38, t.address, kText2, dw / 2 - 16);
    c.text(body_, dx + dw / 2, ty + 38, t.account.empty() ? "Your tailnet" : t.account, kText2, dw / 2);
    ty += 92;
    // The address works from every device on the tailnet; the name only
    // where the device uses Tailscale's DNS (MagicDNS).
    field("APPS USE", "http://" + (t.address.empty() ? short_name : t.address) + ":" + port + "/v1", kText, bodyb_);
    para("With one of your keys, from any device with Tailscale on. Or http://" + short_name + ":" + port +
             "/v1 where the device has Tailscale's DNS on.",
         kText2);
    ty += 16;
    c.text(smallb_, dx, ty, "DEVICES THAT REACHED IT", kMuted);
    ty += 40;
    if (t.connected.empty()) {
        para("None yet since TokenStation5 opened. " + std::to_string(t.devices) +
                 (t.devices == 1 ? " other device is" : " other devices are") + " on your tailnet.",
             kText2);
    }
    // As many as fit above the note and the sign-out line, connected ones
    // first, then the latest.
    // Its keys are in tailscale.tsv, which FTP on the home network can read,
    // as it can keys.tsv.
    const std::string note =
        (t.relay.empty() ? std::string() : "Devices at home connect directly; the rest go through Tailscale's relay in " +
                                               t.relay + ". ") +
        "Its Tailscale keys are in tailscale.tsv, which anyone using FTP on your network can read.";
    const int note_h = note.empty() ? 0 : 16 + 32 * (int)wrap(small_, note, dw).size();
    std::vector<TsDeviceInfo> shown_devices = t.connected;
    std::stable_sort(shown_devices.begin(), shown_devices.end(), [](const TsDeviceInfo &a, const TsDeviceInfo &b) {
        if (a.up != b.up) return a.up;
        return a.seen_s >= 0 && (b.seen_s < 0 || a.seen_s < b.seen_s);
    });
    const int fit = std::max(1, (bottom - ty - note_h - 56) / 40);
    // Too many: the last row that fits says how many more.
    const size_t n = shown_devices.size() <= (size_t)fit ? shown_devices.size() : (size_t)(fit - 1);
    for (size_t i = 0; i < n; i++) {
        const TsDeviceInfo &d = shown_devices[i];
        c.text(body_, dx, ty, d.name, kText, dw / 2 - 16);
        c.text_right(small_, dx + dw, ty, d.up ? (d.direct ? "Connected directly" : "Connected") : ago(d.seen_s),
                     d.up ? kGreen : kText2);
        ty += 40;
    }
    if (n < shown_devices.size()) {
        const std::string more = std::to_string(shown_devices.size() - n);
        c.text(small_, dx, ty, n ? "and " + more + " more" : more + " devices", kText2);
        ty += 40;
    }
    if (!note.empty()) {
        ty += 8;
        para(note, kMuted);
    }
    ty += 8;
    const bool confirming = info.now - confirm_at_ < kConfirmFor;
    text_glyph(c, small_, dx, ty, confirming ? "Press" : "", kGlyphSquare,
               confirming ? "again to sign this PS5 out" : "Sign this PS5 out of your tailnet", confirming ? kAmber : kText2);
    ty += 40;
}

std::string SettingsPage::cross_does(const SettingsInfo &info) const {
    const auto rs = rows(info);
    const Row &r = rs[std::max(0, std::min(sel_, (int)rs.size() - 1))];
    if (r.kind == Row::Guide) return in_guide_ && open_ >= 0 ? "Back" : "Open";
    if (r.kind == Row::NewKey) return "New key";
    if (r.kind == Row::Tailscale) return current(r) == "1" ? "Turn off" : "Turn on";
    if (const ApiKeyInfo *k = key_of(r, info)) return k->on ? "Turn off" : "Turn on";
    if (is_key(r)) return "Type";
    const auto keys = missing(r, info);
    if (!keys.empty()) {
        const app::Download d = app::joined(keys, info.downloads);
        return d.active() ? "Stop the download" : d.done > 0 ? "Go on" : "Download";
    }
    return choices(r, info).empty() ? "" : "Next choice";
}

std::string SettingsPage::arrows_do(const SettingsInfo &info) const {
    const auto rs = rows(info);
    const Row &r = rs[std::max(0, std::min(sel_, (int)rs.size() - 1))];
    if (r.kind != Row::Guide) return "\xE2\x86\x90\xE2\x86\x92  Change";
    if (!in_guide_) return "";
    return open_ >= 0 ? "" : "\xE2\x86\x90  Back to the settings";  // on a guide, cross says Back
}

std::string SettingsPage::square_does(const SettingsInfo &info) const {
    const auto rs = rows(info);
    const Row &r = rs[std::max(0, std::min(sel_, (int)rs.size() - 1))];
    if (r.kind == Row::Guide || r.kind == Row::NewKey) return "";
    if (r.kind == Row::Tailscale) return info.ts.account.empty() && info.ts.address.empty() ? "" : "Sign out";
    if (r.kind == Row::ApiKey || is_key(r)) return "Remove";
    return "Default";
}

}  // namespace ui
