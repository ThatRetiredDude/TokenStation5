#include "settings_page.h"

#include <algorithm>
#include <cstdlib>

#include "pad.h"
#include "settings.h"
#include "style.h"

namespace ui {

namespace {

struct Choice {
    std::string value, shown;
};

// One setting: NAME in ts5.env, and what reads it does with it. No choices:
// a key, typed on the console's keyboard, or files in the models folder (get:
// the command that fetches them).
struct Row {
    const char *section, *name, *label, *help, *applies;
    std::vector<Choice> choices;
    std::string def;  // the value with nothing in ts5.env, as each reader's default
    const char *get = nullptr;
};

constexpr const char *kSpeechModel = "TS5_WHISPER_MODEL", *kVoice = "TS5_TTS_VOICE";
// The voice with no TS5_TTS_VOICE, if it's there (tts/synth.h).
constexpr const char *kDefaultVoice = "af_heart";
constexpr int kRowH = 64, kHeadH = 44, kGap = 8;
constexpr double kNoteFor = 6, kConfirmFor = 4;

const std::vector<Row> &rows() {
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
         "The speech-to-text model that holding R1 on the Chat page uses: a ggml-*.bin file in the models "
         "folder. base.en is a good start; small.en understands more, and takes longer.",
         "The next time you hold R1", {}, "", "scripts/ts5-ps5.sh speech base.en"},
        {"TALK", "TS5_WHISPER_THREADS", "Speech threads",
         "CPU cores that turn what you said into text. More is faster while nothing else runs on the CPU.",
         "The next time you hold R1", {{"4", "4"}, {"6", "6"}, {"8", "8"}, {"10", "10"}, {"12", "12"}}, "6"},
        {"SPOKEN REPLIES", "TS5_SPEAK", "Read replies aloud",
         "Reads each reply on the Chat page out loud as it comes in, a sentence at a time, in a Kokoro voice "
         "made on the console's CPU. Triangle stops it. The voice is loaded only while this is on: off gives "
         "its memory back.",
         "Now", {{"1", "On"}, {"0", "Off"}}, "1"},
        {"SPOKEN REPLIES", "TS5_SPEAK_OUT", "Speak on",
         "Where the voice comes out: the TV, or the speaker in the controller.", "From the next sentence",
         {{"tv", "TV"}, {"controller", "Controller speaker"}}, "tv"},
        {"SPOKEN REPLIES", kVoice, "Voice",
         "Kokoro's voice: a kokoro-voice-*.gguf file in the models folder. af_heart is the clearest; "
         "scripts/ts5-ps5.sh voice NAME adds others, such as am_michael or bf_emma.",
         "From the next sentence", {}, "", "scripts/ts5-ps5.sh voice af_heart"},
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
    };
    return r;
}

bool is_key(const Row &r) { return app::secret_setting(r.name); }

// For a row of files, what its default picks; "" if there are none.
std::string default_file(const Row &r, const SettingsInfo &info) {
    const auto &files = std::string(r.name) == kVoice ? info.voices : info.speech_models;
    if (files.empty()) return "";
    if (std::string(r.name) == kVoice && std::find(files.begin(), files.end(), kDefaultVoice) != files.end())
        return kDefaultVoice;
    return files.front();
}

// The row's choices; a row of files has the files there are.
std::vector<Choice> choices(const Row &r, const SettingsInfo &info) {
    if (!r.get) return r.choices;
    const auto &files = std::string(r.name) == kVoice ? info.voices : info.speech_models;
    std::vector<Choice> c;
    if (files.empty()) return c;
    const std::string def = default_file(r, info);
    c.push_back({"", (std::string(r.name) == kVoice ? "The default (" : "The first one (") + def + ")"});
    for (const std::string &m : files) c.push_back({m, m});
    return c;
}

std::string current(const Row &r) { return app::setting(r.name, r.def); }

// The current value's choice, or -1 (a value set some other way).
int index_of(const std::vector<Choice> &cs, const std::string &v) {
    for (size_t i = 0; i < cs.size(); i++)
        if (cs[i].value == v) return (int)i;
    return -1;
}

std::string shown(const Row &r, const SettingsInfo &info) {
    if (is_key(r)) return current(r).empty() ? "Not set" : "Set";
    const auto cs = choices(r, info);
    if (r.get && cs.empty()) return "None yet";
    const int i = index_of(cs, current(r));
    if (i < 0) return current(r);
    if (r.get && i == 0) return default_file(r, info);
    return cs[i].shown;
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
      smallb_(fonts.get("smallb")), mono_(fonts.get("mono")) {}

SettingChange SettingsPage::press(uint32_t b, const SettingsInfo &info) {
    SettingChange ch;
    const auto &rs = rows();
    if (b & kUp) sel_ = std::max(0, sel_ - 1);
    if (b & kDown) sel_ = std::min((int)rs.size() - 1, sel_ + 1);
    if (b & (kUp | kDown)) confirm_at_ = -1e9;
    const Row &r = rs[sel_];
    auto set = [&](const std::string &v) {
        if (v == current(r)) return;
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
    const auto cs = choices(r, info);
    if (cs.empty()) return ch;
    int i = index_of(cs, current(r));
    if (i < 0) i = std::max(0, index_of(cs, r.def));
    if (b & kLeft) set(cs[std::max(0, i - 1)].value);
    else if (b & kRight) set(cs[std::min((int)cs.size() - 1, i + 1)].value);
    else if (b & kCross) set(cs[(i + 1) % cs.size()].value);
    else if (b & kSquare) set(r.def);
    return ch;
}

void SettingsPage::draw(Canvas &c, const SettingsInfo &info, int x, int y, int w, int h) {
    c.text(h2_, x + 32, y + 50, "Settings", kText);
    c.text_right(small_, x + w - 32, y + 50, "kept in ts5.env", kText2);
    const auto &rs = rows();

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
    const Row &r = rows()[sel_];
    const int dx = x + 32, dw = w - 64;
    c.text(h2_, dx, y + 52, r.label, kText, dw);
    int ty = y + 100;
    for (const std::string &l : wrap(body_, r.help, dw)) {
        c.text(body_, dx, ty, l, kText2);
        ty += 38;
    }
    ty += 24;
    const std::string cur = current(r);

    if (is_key(r)) {
        c.text(smallb_, dx, ty, "NOW", kMuted);
        c.text(bodyb_, dx, ty + 38, cur.empty() ? "Not set" : "Set", cur.empty() ? kText2 : kGreen);
        ty += 84;
        c.text(smallb_, dx, ty, "TO SET IT", kMuted);
        text_glyph(c, body_, dx, ty + 38, "Press", kGlyphCross,
                   "and type it, with the controller or a keyboard.", kText2);
        c.text(small_, dx, ty + 84, "Get a key at app.tavily.com. It stays on your PS5, out of the log.", kMuted, dw);
        ty += 120;
        if (!cur.empty()) {
            const bool confirming = info.now - confirm_at_ < kConfirmFor;
            text_glyph(c, small_, dx, ty + 8, confirming ? "Press" : "", kGlyphSquare,
                       confirming ? "again to remove the key" : "Remove the key", confirming ? kAmber : kText2);
            ty += 40;
        }
    } else {
        const auto cs = choices(r, info);
        c.text(smallb_, dx, ty, "CHOICES", kMuted);
        ty += 20;
        if (cs.empty()) {
            c.text(body_, dx, ty + 30, "None in the models folder yet. On your computer, run", kText2, dw);
            c.round_rect((float)dx, (float)ty + 48, (float)dw, 50, 10, rgb(0x0A1024));
            c.text(mono_, dx + 18, ty + 80, r.get ? r.get : "", kAccent, dw - 36);
            ty += 120;
        }
        const int at = index_of(cs, cur);
        for (size_t i = 0; i < cs.size(); i++) {
            const int cy = ty + 26;
            const bool on = (int)i == at;
            c.ring((float)dx + 12, (float)cy - 8, 11, 2.5f, on ? kAccent : kMuted);
            if (on) c.circle((float)dx + 12, (float)cy - 8, 5.5f, kAccent);
            const int tw = c.text(body_, dx + 40, cy, cs[i].shown, on ? kText : kText2, dw - 40);
            if (cs[i].value == r.def && !r.get)
                c.text(small_, dx + 40 + tw + 14, cy, "default", kMuted);
            ty += 42;
        }
        if (at < 0 && !cs.empty()) {
            c.text(small_, dx + 40, ty + 26, cur + " (set in ts5.env)", kAmber, dw - 40);
            ty += 42;
        }
        ty += 20;
    }

    // Bottom: when a change counts, and the line in ts5.env.
    const int by = y + h - 150;
    c.text(smallb_, dx, by, "TAKES EFFECT", kMuted);
    c.text(body_, dx, by + 36, r.applies, kText2, dw);
    std::string line = cur.empty() || getenv(r.name) == nullptr ? std::string("nothing (the default)")
                       : is_key(r)                               ? std::string(r.name) + "=(set)"
                                                                 : std::string(r.name) + "=" + cur;
    c.text(smallb_, dx, by + 84, "IN TS5.ENV", kMuted);
    c.text(mono_, dx, by + 116, line, kAccent, dw);
    if (info.now - info.note_at < kNoteFor && !info.note.empty())
        c.text_right(smallb_, dx + dw, by, info.note, info.note_failed ? kRed : kGreen);
    (void)ty;
}

}  // namespace ui
