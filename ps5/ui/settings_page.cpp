#include "settings_page.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "library.h"
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
// making a key, and a row per API key (name: its id). So has WIREGUARD: a
// device asking to join, how to add one, a row per device (name: its id)
// and this PS5's key.
struct Row {
    std::string section, name, label, help, applies;
    std::vector<Choice> choices;
    std::string def;  // the value with nothing in ts5.env, as each reader's default
    bool files = false;
    enum Kind { Setting, Guide, NewKey, ApiKey, WgRequest, WgPhone, WgAdd, WgDevice, WgKey } kind = Setting;
};

constexpr const char *kSpeechModel = "TS5_WHISPER_MODEL", *kVoice = "TS5_TTS_VOICE";
// Not ..._KEY: a name ending in _KEY is a secret, typed in (settings.h).
constexpr const char *kRequireKey = "TS5_KEYS_REQUIRED";
constexpr const char *kWireGuard = "TS5_WIREGUARD", *kWgPort = "TS5_WG_PORT", *kWgHost = "TS5_WG_HOST";
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
    {"Away from home with WireGuard", "An encrypted tunnel from your computer, phone or tablet.",
     "Only devices you add here can use the tunnel, and it reaches TokenStation5 alone. There are two ways to "
     "add one: pick by the device."},
    {"Away from home with Tailscale", "Through a computer at home, with no port open on the router.",
     "Never forward the API's own port on your router: anyone could find it, and keys would cross the internet "
     "unprotected. Tailscale keeps it private, through a computer that stays on at home."},
    {"Add models from a computer", "Copy models over your network with FTP, much faster than downloading here.",
     "Big downloads on the PS5 slow down after the first few hundred MB: the PS5 seems to limit how fast apps "
     "can save. FTP from a computer is much faster."},
};
constexpr int kGuideCount = sizeof kGuides / sizeof kGuides[0];
enum { kGuideApp, kGuideWireGuard, kGuideTailscale, kGuideFtp };
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
        {"CHAT", "TS5_CHAT_FPS", "Frames while a reply comes in",
         "How often the Chat page redraws while a reply streams in. Each frame is a copy on the GPU the model "
         "runs on, so fewer frames leave the model a little more speed.",
         "Now",
         {{"1", "1 a second"}, {"2", "2 a second"}, {"5", "5 a second"}, {"10", "10 a second"},
          {"15", "15 a second"}, {"30", "30 a second"}},
         "10"},
        {"TALK", kSpeechModel, "Speech model",
         "The model that turns what you say into text when you hold R1 on the Chat page. Base (English) is a "
         "good start; Small (English) understands more and takes longer.",
         "The next time you hold R1", {}, "", true},
        {"TALK", "TS5_WHISPER_THREADS", "Speech threads",
         "CPU cores that turn what you said into text. More is faster while nothing else runs on the CPU.",
         "The next time you hold R1", {{"4", "4"}, {"6", "6"}, {"8", "8"}, {"10", "10"}, {"12", "12"}}, "6"},
        {"SPOKEN REPLIES", "TS5_SPEAK", "Read replies aloud",
         "Reads each reply on the Chat page out loud as it comes in, a sentence at a time, in a voice made on "
         "the PS5. Triangle stops it. The voice is loaded only while this is on, so off gives its memory back. "
         "It needs a voice: see Voice below.",
         "Now", {{"1", "On"}, {"0", "Off"}}, "1"},
        {"SPOKEN REPLIES", "TS5_SPEAK_OUT", "Speak on",
         "Where the voice comes out: the TV, or the speaker in the controller.", "From the next sentence",
         {{"tv", "TV"}, {"controller", "Controller speaker"}}, "tv"},
        {"SPOKEN REPLIES", kVoice, "Voice",
         "The voice that reads replies. Heart (US English) is the clearest; others include Michael (US "
         "English) and Emma (British English).",
         "From the next sentence", {}, "", true},
        {"SPOKEN REPLIES", "TS5_TTS_SPEED", "Speaking speed", "How fast the voice talks.", "From the next sentence",
         {{"0.8", "Slower (0.8x)"}, {"0.9", "0.9x"}, {"1", "Normal"}, {"1.1", "1.1x"}, {"1.25", "1.25x"},
          {"1.5", "1.5x"}},
         "1"},
        {"SPOKEN REPLIES", "TS5_TTS_THREADS", "Voice threads",
         "CPU cores that make the voice. More is faster while nothing else runs on the CPU.",
         "From the next sentence", {{"4", "4"}, {"6", "6"}, {"8", "8"}, {"10", "10"}, {"12", "12"}}, "6"},
        {"WEB SEARCH", "TAVILY_API_KEY", "Tavily key",
         "With a key, a model that calls tools can search the web through Tavily. Each search uses one of the "
         "account's credits.",
         "From the next request", {}, ""},
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
        {"LOGS", "TS5_NET_LOG", "Detailed log",
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

const WgDeviceInfo *device_of(const Row &r, const SettingsInfo &info) {
    if (r.kind != Row::WgDevice) return nullptr;
    for (const WgDeviceInfo &d : info.wg.devices)
        if (d.id == r.name) return &d;
    return nullptr;
}

// What tells one WireGuard request from the next, "" with none.
std::string asking(const SettingsInfo &info) {
    return info.wg.waiting ? info.wg.waiting_name + "\t" + info.wg.waiting_address + "\t" + info.wg.waiting_from : "";
}

// The page's rows: APPS (the instructions, the key setting, making a key and
// the keys), WIREGUARD, then the settings.
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
    r.push_back({"WIREGUARD", kWireGuard, "WireGuard",
                 "Reach this PS5 away from home from your computer, phone or tablet, through an encrypted tunnel "
                 "that only devices you approve can use. It needs a port forwarded on your router: see "
                 "Instructions, at the top. Off: nothing listens for it.",
                 "Now", {{"1", "On"}, {"0", "Off"}}, "0"});
    if (info.wg.waiting)
        r.push_back({"WIREGUARD", "", info.wg.waiting_name + " asks to join",
                     "A device asked to join from the Add a computer page on your home network. Let it in only if "
                     "you asked from it yourself, just now.",
                     "", {}, "", false, Row::WgRequest});
    r.push_back({"WIREGUARD", "", "Add a phone or tablet",
                 "The quickest way for a phone or tablet: this PS5 makes its key pair and shows its tunnel as a "
                 "code to scan in WireGuard. It keeps only the public half.",
                 "", {}, "", false, Row::WgPhone});
    r.push_back({"WIREGUARD", "", "Add a computer",
                 "For a computer, or any device whose private key should never leave it: a page in its browser "
                 "makes the key pair on the device and gives this PS5 only the public half. Then you let it in here.",
                 "", {}, "", false, Row::WgAdd});
    for (const WgDeviceInfo &d : info.wg.devices)
        r.push_back({"WIREGUARD", d.id, d.name,
                     "A device that can reach this PS5 through WireGuard. Turn it off to shut it out without "
                     "removing it.",
                     "Now", {{"1", "On"}, {"0", "Off"}}, "", false, Row::WgDevice});
    r.push_back({"WIREGUARD", "", "This PS5's key",
                 "Devices know this PS5 by its public key. Its private key is kept in the app's folder, which FTP "
                 "can reach from your home network: if you think it got out, make a new key pair, and give each "
                 "device the new public key.",
                 "", {}, "", false, Row::WgKey});
    r.push_back({"WIREGUARD", kWgPort, "Port",
                 "The UDP port WireGuard listens on. Forward it on your router to this PS5. Change it only if "
                 "something else uses it. (Not WireGuard's usual 51820: the PS5 doesn't let apps use ports that "
                 "high.)",
                 "Now", {{"9030", "9030"}, {"9031", "9031"}, {"9032", "9032"}, {"9033", "9033"}, {"9034", "9034"}},
                 "9030"});
    r.push_back({"WIREGUARD", kWgHost, "Home address",
                 "Where devices find your home on the internet: a dynamic DNS name for it, if your router has one, "
                 "or its public address. Not set: this PS5 looks up its public address itself. Away from home, "
                 "devices added from now on use this.",
                 "For devices added from now on", {}, ""});
    const auto &s = setting_rows();
    r.insert(r.end(), s.begin(), s.end());
    return r;
}

// "24 Oct 2026" (local to the PS5's clock, UTC); "never" for 0.
std::string date_of(int64_t t) {
    if (t <= 0) return "Never";
    static const char *months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    const time_t tt = (time_t)t;
    struct tm g;
    if (!gmtime_r(&tt, &g)) return "";
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

// The page a device asks to join from: the base URL with /wireguard for /v1.
std::string join_url(const SettingsInfo &info) {
    std::string u = info.url;
    const size_t v1 = u.rfind("/v1");
    if (v1 != std::string::npos && v1 + 3 == u.size()) u.resize(v1);
    return u + "/wireguard";
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
// Typed on the console's keyboard: a key, or Home address.
bool typed(const Row &r) { return is_key(r) || (r.kind == Row::Setting && r.name == kWgHost); }

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
    if (const ApiKeyInfo *k = key_of(r, info)) return k->on ? "On" : "Off";
    if (r.kind == Row::WgRequest) return "Waiting";
    if (r.kind == Row::WgPhone) {
        if (!info.wg.tunnel.empty()) return "Scan the code";
        for (const WgDeviceInfo &d : info.wg.devices)
            if (d.id == info.wg.added_id) return d.up ? "Connected" : "Added";
        return "Add one";
    }
    if (r.kind == Row::WgAdd) return "How";
    if (r.kind == Row::WgKey) return info.wg.public_key.empty() ? "None yet" : "Show";
    if (const WgDeviceInfo *d = device_of(r, info)) return !d->on ? "Off" : d->up ? "Connected" : "On";
    if (is_key(r)) return current(r).empty() ? "Not set" : "Set";
    if (typed(r)) return !current(r).empty() ? current(r) : info.wg.home_looked_up ? "Looked up" : "Not set";
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
// shows the key, and to a device asking to join WireGuard (once).
void follow(const std::vector<Row> &rs, const SettingsInfo &info, int &sel, std::string &seen,
            std::string &asking_seen) {
    if (info.fresh_key_id != seen) {
        seen = info.fresh_key_id;
        for (size_t i = 0; i < rs.size(); i++)
            if (rs[i].kind == Row::ApiKey && rs[i].name == seen) sel = (int)i;
    }
    if (asking(info) != asking_seen) {
        asking_seen = asking(info);
        for (size_t i = 0; i < rs.size(); i++)
            if (rs[i].kind == Row::WgRequest) sel = (int)i;
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
    follow(rs, info, sel_, fresh_seen_, asking_seen_);
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
    if (r.kind == Row::WgAdd) return ch;
    if (r.kind == Row::WgPhone) {
        ch.wg_phone = (b & kCross) != 0;
        return ch;
    }
    if (r.kind == Row::WgRequest) {
        ch.wg_answer = (b & kCross) ? 1 : (b & kSquare) ? -1 : 0;
        return ch;
    }
    if (r.kind == Row::WgKey) {
        if ((b & kSquare) && !info.wg.public_key.empty()) {
            if (info.now - confirm_at_ < kConfirmFor) {
                ch.wg_new_key = true;
                confirm_at_ = -1e9;
            } else {
                confirm_at_ = info.now;
            }
        }
        return ch;
    }
    if (const WgDeviceInfo *d = device_of(r, info)) {
        const bool want = (b & kCross) ? !d->on : (b & kLeft) ? true : (b & kRight) ? false : d->on;
        if (want != d->on) {
            ch.wg_device = d->id;
            ch.wg_on = want;
        } else if (b & kSquare) {
            if (info.now - confirm_at_ < kConfirmFor) {
                ch.wg_device = d->id;
                ch.wg_remove = true;
                confirm_at_ = -1e9;
            } else {
                confirm_at_ = info.now;
            }
        }
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
    if (typed(r)) {
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
    follow(rs, info, sel_, fresh_seen_, asking_seen_);

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
        const bool changeable = !typed(r) && !choices(r, info).empty();
        int right = lx + lw - 24;
        if (sel && changeable) {
            // The arrows that change it.
            const int cy = ry + kRowH / 2;
            c.segment((float)right - 4, (float)cy, (float)right - 12, (float)cy - 8, 2.5f, kAccent);
            c.segment((float)right - 4, (float)cy, (float)right - 12, (float)cy + 8, 2.5f, kAccent);
            right -= 28;
        }
        const Color vc = is_key(r)                  ? (current(r).empty() ? kText2 : kGreen)
                         : r.kind == Row::WgRequest ? kAmber
                         : sel                      ? kAccent
                                                    : kText2;
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
    if (open == kGuideApp) {
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
    } else if (open == kGuideWireGuard) {
        ty -= 12;
        const std::string port = app::setting(kWgPort, "9030");
        para("1. Turn on WireGuard, on the left. On your router, forward UDP port " + port +
                 " to this PS5, and keep its address fixed with a DHCP reservation.",
             kText);
        ty += 8;
        para("2. A phone or tablet: Add a phone or tablet. Name it, and scan the code the TV shows in its "
             "WireGuard app. Quickest: this PS5 makes its keys.",
             kText);
        ty += 8;
        para("3. A computer: Add a computer. Open the page it shows in the computer's browser, let it in here, "
             "and import the tunnel file. Its key never leaves it, so phones can use this way too.",
             kText);
        ty += 8;
        para("Away from home, with the tunnel on, apps use http://10.55.0.1:" + std::to_string(info.api_port) +
                 "/v1 and one of your keys.",
             kText);
        ty += 8;
        para("The page stays blank on a device? Turn off its VPN while you add it.", kText2);
        ty += 8;
        para("If your internet provider shares one address among many homes (CGNAT), a forwarded port can't be "
             "reached: use Tailscale instead.",
             kText2);
    } else if (open == kGuideTailscale) {
        ty -= 12;
        if (info.on_network) field("THIS PS5'S ADDRESS", host, kAccent, bodyb_);
        else field("THIS PS5'S ADDRESS", "Connect the PS5 to your network first", kAmber, body_);
        para("1. Install Tailscale (free, tailscale.com) on a computer that stays on at home, and on your phone "
             "or laptop, signed in to the same account.",
             kText);
        ty += 8;
        para("2. On the home computer, share this PS5's address as a subnet route: in a terminal, tailscale up "
             "--advertise-routes=" + (info.on_network ? host : std::string("ADDRESS")) + "/32",
             kText);
        ty += 8;
        para("3. In Tailscale's admin page, under Machines, approve the route for the home computer.", kText);
        ty += 8;
        para("4. Away from home, with Tailscale on, apps use the same base URL as at home. Turn on Require a key "
             "under Apps on the left too, and keep the PS5's address fixed with a DHCP reservation in your "
             "router.",
             kText);
    } else if (open == kGuideFtp) {
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
    } else if (r.kind == Row::NewKey) {
        text_glyph(c, body_, dx, ty + 8, "Press", kGlyphCross, "to name a new key.", kText2);
        ty += 56;
        if (info.keys.empty()) para("You have no keys yet.", kMuted);
        else para("You have " + std::to_string(info.keys.size()) + (info.keys.size() == 1 ? " key." : " keys."), kMuted);
    } else if (r.kind == Row::WgRequest) {
        field("ASKED FROM", info.wg.waiting_from + " on your home network", kText, bodyb_);
        field("ITS ADDRESS IN THE TUNNEL", info.wg.waiting_address, kText, bodyb_);
        text_glyph(c, body_, dx, ty + 8, "Press", kGlyphCross, "to let it in.", kText);
        ty += 48;
        text_glyph(c, body_, dx, ty + 8, "Press", kGlyphSquare, "to refuse it.", kText2);
        ty += 56;
        para("The request lapses after 10 minutes.", kMuted);
    } else if (r.kind == Row::WgPhone) {
        const std::string port = app::setting(kWgPort, "9030");
        const WgDeviceInfo *added = nullptr;
        for (const WgDeviceInfo &d : info.wg.devices)
            if (d.id == info.wg.added_id) added = &d;
        if (info.wg.tunnel.empty() && added) {
            // Added just now, its code gone: whether it has connected.
            field("ADDED", added->name, kAccent, bodyb_);
            if (added->up) {
                field("NOW", "Connected", kGreen, bodyb_);
                para(added->name + " can reach TokenStation5 through WireGuard now. It's in the list below, where you "
                     "can turn it off or remove it.",
                     kText);
            } else {
                field("NOW", "Not connected yet", kAmber, bodyb_);
                para("Turn the TokenStation5 tunnel on in WireGuard on the device. At home, some routers don't "
                     "pass a device's own public address back in: if it doesn't connect, try it on mobile data.",
                     kText);
            }
            ty += 16;
            text_glyph(c, small_, dx, ty, "Press", kGlyphCross, "to add another.", kText2);
            ty += 40;
        } else if (!info.wg.tunnel.empty()) {
            // The tunnel just made: the code, with what to do beside it.
            field("SCAN THIS IN WIREGUARD", info.wg.tunnel_name, kAccent, bodyb_);
            const int side = draw_qr(c, dx, ty - 8, info.wg.tunnel, 414);
            const int sx = dx + side + 32, sw = dx + dw - sx;
            int sy = ty - 4;
            auto step = [&](const std::string &text, Color col) {
                for (const std::string &l : wrap(small_, text, sw)) {
                    c.text(small_, sx, sy, l, col);
                    sy += 32;
                }
                sy += 8;
            };
            step("In WireGuard on the device: +, then Create from QR code. Scan this, and name the tunnel "
                 "TokenStation5.",
                 kText);
            step("Turn it on: this says Connected once it reaches the PS5. Apps use http://10.55.0.1:" +
                     std::to_string(info.api_port) + "/v1 and one of your keys.",
                 kText);
            step("The code holds the device's private key: keep it to yourself. It goes once the device connects "
                 "or you leave Settings, and this PS5 doesn't keep it.",
                 kText2);
            ty = std::max(sy, ty - 8 + side + 24);
        } else {
            const char *label = "ITS TUNNEL GOES TO";
            if (!info.on_network) field(label, "Connect the PS5 to your network first", kAmber, body_);
            else if (app::setting(kWireGuard, "0") != "1") field(label, "Turn on WireGuard above first", kAmber, body_);
            else if (info.wg.home.empty())
                field(label, "Your home's address isn't known: type it in Home address, below", kAmber, body_);
            else field(label, info.wg.home + ":" + port, kAccent, bodyb_);
            para("1. Install WireGuard on the phone or tablet, from its app store.", kText);
            ty += 8;
            text_glyph(c, small_, dx, ty, "2. Press", kGlyphCross, "here and type a name for it, such as Phone.", kText);
            ty += 40;
            para("3. In WireGuard, tap +, then Create from QR code, and scan the code that shows here.", kText);
            ty += 8;
            para("The press here lets it in. Anyone who sees the code could copy the tunnel, so show it only to "
                 "you. To keep a device's private key off this PS5 altogether, use Add a computer instead: it "
                 "works on phones too.",
                 kText2);
        }
    } else if (r.kind == Row::WgAdd) {
        const char *open_label = "IN THE COMPUTER'S BROWSER, OPEN";
        const bool ready = info.on_network && app::setting(kWireGuard, "0") == "1";
        if (!info.on_network) field(open_label, "Connect the PS5 to your network first", kAmber, body_);
        else if (!ready) field(open_label, "Turn on WireGuard above first", kAmber, body_);
        else field(open_label, join_url(info), kAccent, bodyb_);
        // The code, with the steps beside it.
        int side = 0;
        if (ready) side = draw_qr(c, dx, ty - 8, join_url(info), 264);
        const int sx = side > 0 ? dx + side + 32 : dx, sw = dx + dw - sx;
        int sy = side > 0 ? ty - 4 : ty;
        auto step = [&](const std::string &text, Color col) {
            for (const std::string &l : wrap(small_, text, sw)) {
                c.text(small_, sx, sy, l, col);
                sy += 32;
            }
            sy += 8;
        };
        step("1. Install WireGuard on the computer, from wireguard.com/install.", kText);
        step("2. At home, type the address in its browser (on a phone, scan the code with the camera app). Name "
             "it and ask to join.",
             kText);
        step("3. Let it in here. The page then gives it its tunnel file: in WireGuard, Import tunnel(s) from file.",
             kText);
        ty = std::max(sy, side > 0 ? ty - 8 + side + 24 : 0);
        para("The page stays blank? Turn off the device's VPN while you add it, or let the VPN reach your home "
             "network.",
             kAmber);
        ty += 8;
        para("Instructions, at the top, has the rest: the port on your router, and the address apps use.", kText2);
    } else if (const WgDeviceInfo *d = device_of(r, info)) {
        field("NOW", !d->on ? "Off" : d->up ? "Connected" : "On, not connected", !d->on ? kText2 : d->up ? kGreen : kText,
              bodyb_);
        c.text(smallb_, dx, ty, "ADDRESS IN THE TUNNEL", kMuted);
        c.text(smallb_, dx + dw / 2, ty, "ADDED", kMuted);
        c.text(body_, dx, ty + 38, d->address, kText2, dw / 2 - 16);
        c.text(body_, dx + dw / 2, ty + 38, date_of(d->added), kText2, dw / 2);
        ty += 92;
        field("LAST HEARD FROM", ago(d->seen_s), kText2, body_);
        text_glyph(c, small_, dx, ty, "Press", kGlyphCross, d->on ? "to turn it off." : "to turn it on.", kText2);
        ty += 40;
        const bool confirming = info.now - confirm_at_ < kConfirmFor;
        text_glyph(c, small_, dx, ty, confirming ? "Press" : "", kGlyphSquare,
                   confirming ? "again to remove the device" : "Remove the device", confirming ? kAmber : kText2);
        ty += 40;
    } else if (r.kind == Row::WgKey) {
        if (info.wg.public_key.empty()) {
            field("PUBLIC KEY", "Made when you first turn on WireGuard", kText2, body_);
        } else {
            field("PUBLIC KEY", info.wg.public_key, kAccent, bodyb_);
            para("Each new device gets it in its tunnel. Its private half never leaves this PS5.", kText2);
            ty += 16;
            const bool confirming = info.now - confirm_at_ < kConfirmFor;
            text_glyph(c, small_, dx, ty, confirming ? "Press" : "", kGlyphSquare,
                       confirming ? "again to make a new key pair" : "Make a new key pair", confirming ? kAmber : kText2);
            ty += 40;
            if (confirming) para("Every device then needs the new public key before it can connect again.", kAmber);
        }
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
    } else if (typed(r)) {
        // Home address.
        if (!cur.empty()) field("NOW", cur, kAccent, bodyb_);
        else if (info.wg.home_looked_up) field("NOW, LOOKED UP", info.wg.home, kText, bodyb_);
        else field("NOW", "Not known yet: type it in", kAmber, body_);
        text_glyph(c, body_, dx, ty, "Press", kGlyphCross, cur.empty() ? "to type it." : "to change it.", kText2);
        ty += 48;
        para("Your router's settings page shows its public address. A dynamic DNS name keeps working when "
             "that address changes.",
             kText2);
        if (!cur.empty()) {
            ty += 16;
            const bool confirming = info.now - confirm_at_ < kConfirmFor;
            text_glyph(c, small_, dx, ty, confirming ? "Press" : "", kGlyphSquare,
                       confirming ? "again to look it up instead" : "Look it up instead", confirming ? kAmber : kText2);
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
        if (r.name == kWireGuard && !info.wg.problem.empty()) para(info.wg.problem, kRed);
        else if (r.name == kWireGuard && current(r) == "1") {
            const long on = std::count_if(info.wg.devices.begin(), info.wg.devices.end(),
                                          [](const WgDeviceInfo &d) { return d.on; });
            para(info.wg.devices.empty() ? "No devices yet: add one below."
                 : on == 0               ? "Every device is turned off."
                                         : std::to_string(on) + (on == 1 ? " device can use it." : " devices can use it."),
                 kText2);
        }
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

std::string SettingsPage::cross_does(const SettingsInfo &info) const {
    const auto rs = rows(info);
    const Row &r = rs[std::max(0, std::min(sel_, (int)rs.size() - 1))];
    if (r.kind == Row::Guide) return in_guide_ && open_ >= 0 ? "Back" : "Open";
    if (r.kind == Row::NewKey) return "New key";
    if (const ApiKeyInfo *k = key_of(r, info)) return k->on ? "Turn off" : "Turn on";
    if (r.kind == Row::WgRequest) return "Let it in";
    if (r.kind == Row::WgPhone) return info.wg.tunnel.empty() && info.wg.added_id.empty() ? "Name it" : "Add another";
    if (r.kind == Row::WgAdd || r.kind == Row::WgKey) return "";
    if (const WgDeviceInfo *d = device_of(r, info)) return d->on ? "Turn off" : "Turn on";
    if (typed(r)) return "Type";
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
    if (r.kind == Row::WgRequest || r.kind == Row::WgPhone || r.kind == Row::WgAdd || r.kind == Row::WgKey ||
        typed(r))
        return "";
    if (r.kind != Row::Guide) return "\xE2\x86\x90\xE2\x86\x92  Change";
    if (!in_guide_) return "";
    return open_ >= 0 ? "" : "\xE2\x86\x90  Back to the settings";  // on a guide, cross says Back
}

std::string SettingsPage::square_does(const SettingsInfo &info) const {
    const auto rs = rows(info);
    const Row &r = rs[std::max(0, std::min(sel_, (int)rs.size() - 1))];
    if (r.kind == Row::Guide || r.kind == Row::NewKey || r.kind == Row::WgAdd || r.kind == Row::WgPhone) return "";
    if (typed(r) && !is_key(r)) return current(r).empty() ? "" : "Remove";
    if (r.kind == Row::WgRequest) return "Refuse";
    if (r.kind == Row::WgKey) return info.wg.public_key.empty() ? "" : "New key pair";
    if (r.kind == Row::ApiKey || r.kind == Row::WgDevice || is_key(r)) return "Remove";
    return "Default";
}

}  // namespace ui
