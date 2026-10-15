#include "dashboard.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "pad.h"
#include "plain.h"
#include "style.h"

namespace ui {

namespace {

constexpr int W = 1920, H = 1080;
// Panels: models on the left, details on the right, activity below.
constexpr int PT = 192, PB = 844;
constexpr int LX = 72, LW = 900, RX = 1000, RW = 848;
constexpr int AY = 868, AH = 136;
constexpr int LIST_X = LX + 16, LIST_W = LW - 32, LIST_Y0 = 268, LIST_Y1 = PB - 16;
constexpr int DX = RX + 32, DW = RW - 64;
// The chat page: the conversation, and the line for a message under it.
constexpr int CX = LX, CW = 1848 - LX, CB = 884, IY = 900, IH = AY + AH - 900;
// The settings page: two panels down to where the activity panel ends.
constexpr int SB = AY + AH;

std::string lower(std::string s) {
    for (auto &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

// The catalog model a file is, in whatever precision; null if none.
const app::CatalogEntry *catalog_of(const std::string &file) {
    const std::string f = lower(file);
    for (const app::CatalogEntry &e : app::kCatalog)
        if (e.kind == app::CatalogEntry::Model && f.find(e.same) != std::string::npos) return &e;
    return nullptr;
}

// What a model is good for: its catalog entry's words, found by its file name
// or its own name (a copy under another file name is the same model), else a
// guess from its name.
std::string good_for(const app::ModelInfo &m) {
    std::string name = lower(m.name);
    for (auto &ch : name)
        if (ch == ' ') ch = '-';
    if (const app::CatalogEntry *e = catalog_of(m.file)) return e->good_for;
    if (const app::CatalogEntry *e = catalog_of(name)) return e->good_for;
    // A file with no model in it: nothing to guess from.
    if (m.arch.empty()) return "\xE2\x80\x94";
    return name.find("coder") != std::string::npos ? "Coding" : "General chat";
}

// A model's name in the list: with its precision when another model there
// has the same name (two downloads of one model, 4-bit and 8-bit).
std::string shown_name(const std::vector<app::ModelInfo> &models, const app::ModelInfo &m) {
    int same = 0;
    for (const app::ModelInfo &o : models) same += o.name == m.name;
    if (same < 2 || m.quant.empty()) return m.name;
    return m.name + " (" + plain::quant_detail(m.quant) + ")";
}

// A catalog model is on the PS5 already, in some precision.
bool installed(const View &v, const app::CatalogEntry &e) {
    return std::any_of(v.models.begin(), v.models.end(),
                       [&](const app::ModelInfo &m) { return lower(m.file).find(e.same) != std::string::npos; });
}

app::Download download_of(const View &v, const std::string &key) {
    const auto it = v.downloads.find(key);
    return it == v.downloads.end() ? app::Download() : it->second;
}

// The download that is running or waiting to try again, for "waiting for".
std::string downloading(const View &v) {
    for (const auto &[key, d] : v.downloads)
        if (d.phase == app::Download::Running || d.phase == app::Download::Retrying)
            if (const app::CatalogEntry *e = app::Downloads::find(key)) return e->name;
    return "";
}

// Whole percent, 99 at most until the file is whole.
int percent(const app::Download &d) {
    if (!d.total) return 0;
    return std::min(99, (int)(100.0 * (double)d.done / (double)d.total));
}

// "about 3 minutes left"
std::string time_left(double s) {
    char b[64];
    if (s < 60) return "less than a minute left";
    const int min = (int)std::ceil(s / 60);
    if (min < 60) snprintf(b, sizeof b, "about %d minute%s left", min, min == 1 ? "" : "s");
    else if (min % 60 == 0) snprintf(b, sizeof b, "about %d hour%s left", min / 60, min / 60 == 1 ? "" : "s");
    else snprintf(b, sizeof b, "about %d h %d min left", min / 60, min % 60);
    return b;
}

// "38 MB a second"
std::string per_second(double bytes) {
    char b[40];
    if (bytes >= 1e6) snprintf(b, sizeof b, "%.0f MB a second", bytes / 1e6);
    else snprintf(b, sizeof b, "%.0f KB a second", bytes / 1e3);
    return b;
}

// "1.9 GB of 4.7 GB · 38 MB a second", and the time left when asked.
std::string progress_text(const app::Download &d, bool left) {
    std::string s = app::format_bytes(d.done) + " of " + app::format_bytes(d.total);
    if (d.phase == app::Download::Running && !d.held && d.speed > 0) {
        s += "  \xC2\xB7  " + per_second(d.speed);
        if (left && d.total > d.done) s += "  \xC2\xB7  " + time_left((double)(d.total - d.done) / d.speed);
    }
    return s;
}

// Whether right on Models removes this model: not the one in use or a job's,
// not one still copying, and only from the title's own folder (another
// app's models stay that app's).
bool removable(const View &v, const app::ModelInfo &m) {
    if (m.path == v.status.serving_path || m.copying) return false;
    if (m.file == v.status.job_model && v.status.phase != app::Status::Idle && v.status.phase != app::Status::Serving)
        return false;
    return !v.models_dir.empty() && m.path.compare(0, v.models_dir.size() + 1, v.models_dir + "/") == 0;
}

// "Still copying: 2.1 GB of 4.7 GB so far"
std::string copied(const app::ModelInfo &m) {
    return "Still copying: " + app::format_bytes(m.size) +
           (m.full_size > m.size ? " of " + app::format_bytes(m.full_size) : std::string()) + " so far";
}

std::string ago(int64_t then, int64_t now) {
    const int64_t s = std::max<int64_t>(0, now - then);
    char b[48];
    if (s < 60) return "just now";
    if (s < 3600) snprintf(b, sizeof b, "%lld min ago", (long long)(s / 60));
    else if (s < 86400) snprintf(b, sizeof b, "%lld h ago", (long long)(s / 3600));
    else if (s < 2 * 86400) return "a day ago";
    else snprintf(b, sizeof b, "%lld days ago", (long long)(s / 86400));
    return b;
}

// A line with its time after it ("... · 5 min ago"): a line too long for
// max_w is cut, never the time.
int text_with_time(Canvas &c, const Font &f, int x, int y, const std::string &s, const std::string &when, Color col,
                   int max_w) {
    const std::string tail = "  \xC2\xB7  " + when;
    const int w = c.text(f, x, y, s, col, std::max(1, max_w - Canvas::measure(f, tail)));
    return w + c.text(f, x + w, y, tail, col, std::max(1, max_w - w));
}

std::string elapsed(double s) {
    char b[32];
    if (s < 60) snprintf(b, sizeof b, "%.0f s", s);
    else snprintf(b, sizeof b, "%d min %02d s", (int)s / 60, (int)s % 60);
    return b;
}

std::string tps(double v) { return plain::tokens_a_second(v); }

// Words onto lines no wider than max_w.
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

void label(Canvas &c, const Font &f, int x, int y, const std::string &s) { c.text(f, x, y, s, kMuted); }

// A model's name (general.name) for its file name.
std::string name_of(const View &v, const std::string &file) {
    for (const auto &m : v.models)
        if (m.file == file) return m.name;
    return plain::model_name(file);
}

const char *page_name(Dashboard::Page p) {
    return p == Dashboard::Home     ? "Home"
           : p == Dashboard::Models ? "Models"
           : p == Dashboard::Chat   ? "Chat"
           : p == Dashboard::Settings ? "Settings"
                                      : "Logs";
}

// Some request over Home's range wasn't answered: square shows why.
bool home_problems(const View &v) { return v.usage && v.usage->ready && v.usage->total.failed > 0; }

// A model that can't run, and isn't just still copying: square shows why.
bool cant_run(const app::ModelInfo &m) { return !m.runnable && !m.copying; }

// Cross types a message: a model serves, and no reply is coming.
bool can_type(const View &v) { return v.status.phase == app::Status::Serving && !v.chat.busy; }

// No model runs or loads: cross on Chat goes to Models.
bool nothing_running(const View &v) { return v.status.serving.empty() && v.status.phase != app::Status::Loading; }

// How a model has been used over the last 7 days, from the usage log.
const app::UsageSums *week_of(const View &v, const std::string &name) {
    if (!v.usage) return nullptr;
    const auto it = v.usage->week.find(name);
    return it == v.usage->week.end() ? nullptr : &it->second;
}

}  // namespace

struct Dashboard::Item {
    enum Kind { Model, Header, Catalog, Empty } kind;
    int index;  // into View::models or app::kCatalog
    int h;
    std::string key;
};

Dashboard::Dashboard(const Fonts &fonts, const uint8_t *logo, size_t logo_size)
    : title_(fonts.get("title")), h2_(fonts.get("h2")), body_(fonts.get("body")),
      bodyb_(fonts.get("bodyb")), small_(fonts.get("small")), smallb_(fonts.get("smallb")),
      home_(fonts), logs_(fonts), chat_(fonts), settings_(fonts) {
    if (logo && logo_size >= 8 && memcmp(logo, "TS5L", 4) == 0) {
        uint16_t w, h;
        memcpy(&w, logo + 4, 2);
        memcpy(&h, logo + 6, 2);
        if (logo_size >= 8 + (size_t)w * h * 4) {
            logo_w_ = w;
            logo_h_ = h;
            logo_px_.resize((size_t)w * h);
            memcpy(logo_px_.data(), logo + 8, logo_px_.size() * 4);
            logo_ = logo_px_.data();
        }
    }
}

std::vector<Dashboard::Item> Dashboard::items(const View &v) const {
    std::vector<Item> out;
    for (size_t i = 0; i < v.models.size(); i++) out.push_back({Item::Model, (int)i, 88, v.models[i].path});
    if (v.models.empty()) out.push_back({Item::Empty, 0, 128, ""});
    // The catalog's models that aren't on the PS5, to download; a row with
    // a part or a download going on is taller, for its progress bar.
    std::vector<Item> more;
    for (size_t i = 0; i < sizeof app::kCatalog / sizeof app::kCatalog[0]; i++) {
        const app::CatalogEntry &e = app::kCatalog[i];
        if (e.kind != app::CatalogEntry::Model || installed(v, e)) continue;
        const app::Download d = download_of(v, e.key);
        const bool bar = d.done > 0 || d.active();
        more.push_back({Item::Catalog, (int)i, bar ? 100 : 88, std::string("catalog:") + e.key});
    }
    if (!more.empty()) {
        out.push_back({Item::Header, 0, 56, ""});
        out.insert(out.end(), more.begin(), more.end());
    }
    return out;
}

int Dashboard::selected(const std::vector<Item> &items) const {
    int first = -1;
    for (size_t i = 0; i < items.size(); i++) {
        if (items[i].kind != Item::Model && items[i].kind != Item::Catalog) continue;
        if (first < 0) first = (int)i;
        if (items[i].key == selected_key_) return (int)i;
    }
    // A download that just finished: its model, now on the PS5.
    if (selected_key_.compare(0, 8, "catalog:") == 0)
        if (const app::CatalogEntry *e = app::Downloads::find(selected_key_.substr(8)))
            for (size_t i = 0; i < items.size(); i++)
                if (items[i].kind == Item::Model && lower(items[i].key).find(e->same) != std::string::npos) return (int)i;
    return first;
}

void Dashboard::show(Page p) {
    if (p == Chat && page_ != Chat) chat_.to_bottom();
    if (p == Logs && page_ != Logs) logs_.open(false);
    if (p != Logs && page_ == Logs) logs_.close();
    page_ = p;
}

void Dashboard::show_problems() {
    show(Logs);
    logs_.open(true);
}

SettingsInfo Dashboard::settings_info(const View &v) const {
    SettingsInfo s = v.settings;
    s.speech_models = v.voice.models;
    s.speech_model = v.voice.model;
    s.now = v.now;
    return s;
}

bool Dashboard::animating(const View &v) const {
    if (page_ == Settings || page_ == Home || page_ == Logs) return false;
    if (page_ == Chat)
        return v.chat.busy || v.voice.phase != app::VoiceState::Idle || v.status.phase == app::Status::Loading;
    return v.status.phase == app::Status::Loading || v.status.phase == app::Status::Checking ||
           v.status.phase == app::Status::Benchmarking || v.api.generating;
}

Action Dashboard::press(uint32_t b, const View &v) {
    Action a;
    if (!b) return a;
    if (confirm_ != NoConfirm) {
        if (b & kCross) {
            a.kind = confirm_ == ConfirmClose ? Action::Close : confirm_ == ConfirmStop ? Action::Stop : Action::Remove;
            if (confirm_ == ConfirmRemove) a.path = confirm_path_;
        }
        if (b & (kCross | kCircle | kOptions)) confirm_ = NoConfirm;
        return a;
    }
    if (b & kOptions) {
        confirm_ = ConfirmClose;
        return a;
    }
    if (b & kL1) {
        show(next(page_));
        return a;
    }
    if (page_ == Home) {
        home_.press(b);
        if (b & kCross) show(v.status.phase == app::Status::Serving ? Chat : Models);
        else if ((b & kSquare) && home_problems(v)) show_problems();
        return a;
    }
    if (page_ == Logs) {
        logs_.press(b);
        if (b & kCircle) show(Home);
        return a;
    }
    if (page_ == Settings) {
        const SettingChange ch = settings_.press(b, settings_info(v));
        if (ch.any || ch.type) {
            a.kind = ch.type ? Action::TypeSetting : Action::Set;
            a.name = ch.name;
            a.value = ch.value;
            a.label = ch.label;
        }
        if (b & kCircle) show(Home);
        return a;
    }
    if (page_ == Chat) {
        chat_.press(b);
        if ((b & kTriangle) && (v.chat.busy || v.speaking)) a.kind = Action::ChatStop;
        else if ((b & kSquare) && !v.chat.turns.empty()) a.kind = Action::ChatClear;
        else if ((b & kCross) && can_type(v)) a.kind = Action::ChatType;
        else if ((b & kCross) && nothing_running(v)) show(Models);
        if (a.kind == Action::ChatClear) chat_.to_bottom();
        if (b & kCircle) show(Home);
        return a;
    }
    if (b & kCircle) {
        show(Home);
        return a;
    }
    const auto list = items(v);
    int sel = selected(list);
    if (sel < 0) return a;
    if (selected_key_.empty()) {
        // The serving model, else the first, until a choice is made.
        for (auto &it : list)
            if (it.kind == Item::Model && v.models[it.index].path == v.status.serving_path) selected_key_ = it.key;
        sel = selected(list);
    }
    auto step = [&](int d) {
        for (int i = sel + d; i >= 0 && i < (int)list.size(); i += d)
            if (list[i].kind == Item::Model || list[i].kind == Item::Catalog) return i;
        return sel;
    };
    if (b & kUp) sel = step(-1);
    if (b & kDown) sel = step(+1);
    selected_key_ = list[sel].key;
    const Item &it = list[sel];
    // Downloads go on beside jobs, so these come before the busy check.
    if (it.kind == Item::Catalog) {
        const app::CatalogEntry &e = app::kCatalog[it.index];
        const app::Download d = download_of(v, e.key);
        a.key = e.key;
        if (b & kCross) a.kind = d.active() ? Action::StopDownload : Action::Download;
        else if ((b & kSquare) && !d.active() && d.done > 0) a.kind = Action::RemovePart;
        return a;
    }
    if ((b & kSquare) && it.kind == Item::Model && cant_run(v.models[it.index])) {
        show_problems();
        return a;
    }
    if ((b & kRight) && it.kind == Item::Model && removable(v, v.models[it.index])) {
        const app::ModelInfo &m = v.models[it.index];
        confirm_ = ConfirmRemove;
        confirm_name_ = m.name;
        confirm_path_ = m.path;
        confirm_bytes_ = m.size;
        return a;
    }
    const bool busy = v.status.phase == app::Status::Loading || v.status.phase == app::Status::Checking ||
                      v.status.phase == app::Status::Benchmarking || v.status.queued > 0;
    if (busy) return a;
    if (it.kind == Item::Model && v.models[it.index].runnable) {
        a.path = v.models[it.index].path;
        if ((b & kCross) && a.path == v.status.serving_path) {
            // The model apps use: stop it, once asked.
            confirm_ = ConfirmStop;
            confirm_name_ = v.models[it.index].name;
        } else if (b & kCross) {
            a.kind = Action::Serve;
        } else if (b & kSquare) {
            a.kind = Action::Check;
        } else if (b & kTriangle) {
            a.kind = Action::Bench;
        }
    }
    return a;
}

void Dashboard::draw_background(Canvas &c) {
    if (background_page_ != page_) {
        std::vector<std::array<int, 4>> panels;
        switch (page_) {
            case Home: HomePage::panels(panels); break;
            case Models: panels = {{LX, PT, LW, PB - PT}, {RX, PT, RW, PB - PT}, {LX, AY, 1848 - LX, AH}}; break;
            case Chat: panels = {{CX, PT, CW, CB - PT}, {CX, IY, CW, IH}}; break;
            case Settings: panels = {{LX, PT, LW, SB - PT}, {RX, PT, RW, SB - PT}}; break;
            case Logs: panels = {{LX, PT, 1848 - LX, SB - PT}}; break;
        }
        Canvas bg(W, H);
        bg.gradient(0, 0, W, H, kBgTop, kBgBottom);
        bg.fill(LX, 168, 1848 - LX, 1, kPanelLine);
        for (auto [x, y, w, h] : panels) {
            bg.round_rect((float)x, (float)y, (float)w, (float)h, 20, kPanel);
            bg.round_rect_line((float)x, (float)y, (float)w, (float)h, 20, 1.5f, kPanelLine);
        }
        background_.assign(bg.pixels(), bg.pixels() + (size_t)W * H);
        background_page_ = page_;
    }
    memcpy(c.pixels(), background_.data(), background_.size() * 4);
}

void Dashboard::draw_header(Canvas &c, const View &v) {
    if (logo_) {
        c.image(LX, 40, logo_w_, logo_h_, logo_);
        c.round_rect_line((float)LX - 1, 39, (float)logo_w_ + 2, (float)logo_h_ + 2, 4, 1.5f, kPanelLine);
    }
    const int tx = LX + (logo_ ? logo_w_ + 28 : 0);
    c.text(title_, tx, 100, "TokenStation5", kText);
    c.text(small_, tx, 136, "Local AI on your PS5", kText2);

    const auto &st = v.status;
    std::string state = "NOT RUNNING";
    Color col = kText2;
    switch (st.phase) {
        case app::Status::Loading: state = "LOADING", col = kAmber; break;
        case app::Status::Checking: state = "GPU CHECK", col = kAccent; break;
        case app::Status::Benchmarking: state = "SPEED TEST", col = kAccent; break;
        case app::Status::Serving: state = "READY", col = kGreen; break;
        case app::Status::Idle: break;
    }
    const int left = pill(c, smallb_, 1848, 48, state, col);
    c.circle((float)left - 18, 65, 7, col);
    draw_tabs(c);
    const bool up = !st.serving.empty() && st.phase == app::Status::Serving;
    if (v.on_network) {
        c.text_right(smallb_, 1848, 132, v.url, up ? kAccent : kText2);
        c.text_right(small_, 1848 - Canvas::measure(smallb_, v.url) - 12, 132, "For apps", kMuted);
    } else {
        c.text_right(small_, 1848, 132, "Not on a network yet", kText2);
    }
}

// The pages, as tabs right of the title, with the button that switches them.
void Dashboard::draw_tabs(Canvas &c) {
    int x = 700;
    const int y = 140;
    x += key_cap(c, smallb_, x, y - 2, "L1") + 24;
    for (Page p : {Home, Models, Chat, Settings, Logs}) {
        const char *name = page_name(p);
        const int w = Canvas::measure(bodyb_, name);
        c.text(bodyb_, x, y, name, p == page_ ? kText : kText2);
        if (p == page_) c.round_rect((float)x, 163, (float)w, 5, 2.5f, kAccent);
        x += w + 40;
    }
}

void Dashboard::draw_list(Canvas &c, const View &v, const std::vector<Item> &list, int sel) {
    c.text(h2_, LX + 32, 242, "Models", kText);
    const int n = (int)v.models.size();
    c.text_right(small_, LX + LW - 32, 242,
                 n == 0 ? "none on your PS5 yet" : std::to_string(n) + (n == 1 ? " on your PS5" : " on your PS5"),
                 kText2);

    // Scrolled so the selected row shows.
    std::vector<int> ys(list.size());
    int y = 0;
    for (size_t i = 0; i < list.size(); i++) {
        ys[i] = y;
        y += list[i].h + 8;
    }
    const int view_h = LIST_Y1 - LIST_Y0;
    if (sel >= 0) {
        if (ys[sel] - scroll_ < 0) scroll_ = ys[sel];
        if (ys[sel] + list[sel].h - scroll_ > view_h) scroll_ = ys[sel] + list[sel].h - view_h;
    }
    scroll_ = std::max(0, std::min(scroll_, std::max(0, y - 8 - view_h)));

    c.clip(LIST_X - 4, LIST_Y0, LIST_W + 8, view_h);
    for (size_t i = 0; i < list.size(); i++) {
        const Item &it = list[i];
        const int ry = LIST_Y0 + ys[i] - scroll_;
        if (ry > LIST_Y1 || ry + it.h < LIST_Y0) continue;
        const bool is_sel = (int)i == sel;
        const int right = LIST_X + LIST_W - 24;
        if (it.kind == Item::Header) {
            c.text(smallb_, LIST_X + 8, ry + 40, "DOWNLOAD", kMuted);
            continue;
        }
        if (it.kind == Item::Empty) {
            c.round_rect_line((float)LIST_X, (float)ry, (float)LIST_W, (float)it.h, 14, 1.5f, kPanelLine);
            c.text(bodyb_, LIST_X + 24, ry + 44, "No models on your PS5 yet", kText);
            text_glyph(c, small_, LIST_X + 24, ry + 84, "Choose one below and press", kGlyphCross, "to download it.",
                       kText2);
            continue;
        }
        c.round_rect((float)LIST_X, (float)ry, (float)LIST_W, (float)it.h, 14, is_sel ? kRowSel : kRow);
        if (is_sel) c.round_rect_line((float)LIST_X, (float)ry, (float)LIST_W, (float)it.h, 14, 2.5f, kAccent);
        if (it.kind == Item::Catalog) {
            const app::CatalogEntry &e = app::kCatalog[it.index];
            const app::Download d = download_of(v, e.key);
            using D = app::Download;
            std::string tag = "DOWNLOAD", line;
            Color tc = kAccent, lc = kText2;
            char pc[16];
            snprintf(pc, sizeof pc, "%d%%", percent(d));
            switch (d.phase) {
                case D::None:
                    line = std::string(e.good_for) + "  \xC2\xB7  " + app::format_bytes(e.bytes) + " download";
                    lc = kMuted;
                    break;
                case D::Queued:
                    tag = "WAITING", tc = kText2;
                    line = "Waiting for the download before it";
                    break;
                case D::Running:
                    if (d.held) tag = "WAITING", tc = kText2;
                    else tag = pc;
                    line = d.held ? std::string("Waiting while a model loads") : progress_text(d, false);
                    break;
                case D::Retrying:
                    tag = "WAITING", tc = kAmber;
                    line = "Trying again in " + std::to_string(std::max(1, (int)std::ceil(d.retry_at - v.now))) + " s";
                    lc = kAmber;
                    break;
                case D::Stopped:
                    tag = "STOPPED", tc = kAmber;
                    line = std::string("Stopped at ") + pc + "  \xC2\xB7  " + app::format_bytes(d.done) + " of " +
                           app::format_bytes(d.total);
                    break;
                case D::Failed:
                    tag = "STOPPED", tc = kRed;
                    line = d.done ? std::string("Stopped at ") + pc + ": see why on the right"
                                  : std::string("Didn't download: see why on the right");
                    lc = kRed;
                    break;
            }
            const int pl = pill(c, smallb_, right, ry + 14, tag, tc);
            c.text(bodyb_, LIST_X + 24, ry + 40, e.name, d.phase == D::None ? kText2 : kText, pl - 16 - (LIST_X + 24));
            c.text(small_, LIST_X + 24, ry + 72, line, lc, LIST_W - 48);
            if (it.h > 88) {
                // The progress bar along the foot of the row.
                const float bx = (float)LIST_X + 24, bw = (float)LIST_W - 48, by = (float)ry + it.h - 18;
                c.round_rect(bx, by, bw, 6, 3, kTrack);
                const float f = d.total ? std::min(1.0f, (float)d.done / (float)d.total) : 0;
                const Color bc = d.phase == D::Failed ? kRed : d.phase == D::Running && !d.held ? kAccent : kText2;
                if (f > 0) c.round_rect(bx, by, std::max(6.0f, bw * f), 6, 3, bc);
            }
            continue;
        }
        const app::ModelInfo &m = v.models[it.index];
        const app::Results r = v.results.count(m.file) ? v.results.at(m.file) : app::Results{};
        int pl = right + 16;
        const bool job_here = v.status.job_model == m.file;
        if (job_here && v.status.phase == app::Status::Loading) pl = pill(c, smallb_, right, ry + 14, "LOADING", kAmber);
        else if (job_here && v.status.phase == app::Status::Checking)
            pl = pill(c, smallb_, right, ry + 14, "GPU CHECK", kAccent);
        else if (job_here && v.status.phase == app::Status::Benchmarking)
            pl = pill(c, smallb_, right, ry + 14, "SPEED TEST", kAccent);
        else if (m.path == v.status.serving_path) pl = pill(c, smallb_, right, ry + 14, "IN USE", kGreen);
        else if (m.copying) pl = pill(c, smallb_, right, ry + 14, "COPYING", kAmber);
        else if (!m.runnable) pl = pill(c, smallb_, right, ry + 14, "CAN'T RUN", kRed);
        c.text(bodyb_, LIST_X + 24, ry + 40, shown_name(v.models, m), m.runnable ? kText : kText2,
               pl - 16 - (LIST_X + 24));

        // Results, right-aligned under the label: a failed check, else the
        // speed (with a tick once the check passed): as used over the last 7
        // days, else the speed test's.
        int rx = right;
        const app::UsageSums *week = week_of(v, m.name);
        const bool used = week && week->gen >= 32 && week->gen_ms > 0;
        if (r.has_check && !r.check.pass) {
            rx = c.text_right(smallb_, rx, ry + 72, "\xE2\x9C\x97 Failed GPU check", kRed) - 20;
        } else if (used || r.has_bench) {
            rx = c.text_right(small_, rx, ry + 72, tps(used ? week->gen * 1000.0 / week->gen_ms : r.bench.tg_tps), kText2) - 8;
            if (r.has_check) rx = c.text_right(smallb_, rx, ry + 72, "\xE2\x9C\x93", kGreen) - 20;
            else rx -= 12;
        } else if (r.has_check) {
            rx = c.text_right(smallb_, rx, ry + 72, "\xE2\x9C\x93 Passed GPU check", kGreen) - 20;
        }
        std::string line;
        if (m.copying) {
            line = copied(m);
        } else {
            line = good_for(m);
            if (m.params > 0) line += "  \xC2\xB7  " + app::format_params(m.params);
            line += "  \xC2\xB7  " + app::format_bytes(m.size);
        }
        c.text(small_, LIST_X + 24, ry + 72, line, m.copying ? kAmber : kText2, rx - (LIST_X + 24));
    }
    c.unclip();
}

void Dashboard::draw_details(Canvas &c, const View &v, const Item &it) {
    const int col2 = DX + DW / 2;
    auto cell = [&](int x, int top, const std::string &k, const std::string &val, int w = DW / 2 - 24) {
        label(c, smallb_, x, top + 24, k);
        c.text(bodyb_, x, top + 60, val, kText, w);
    };
    struct Btn {
        PadGlyph g;
        std::string text;
        bool on;
    };
    auto buttons = [&](const std::vector<Btn> &bs) {
        const int bw = (DW - 8 * 3) / 4, by = 772, bh = 56;
        for (size_t i = 0; i < bs.size(); i++) {
            const int bx = DX + (int)i * (bw + 8);
            c.round_rect((float)bx, (float)by, (float)bw, (float)bh, 14, bs[i].on ? rgb(0x18244C) : rgb(0x111933));
            c.round_rect_line((float)bx, (float)by, (float)bw, (float)bh, 14, 1.5f,
                              bs[i].on ? rgb(0x2A3A70) : kPanelLine);
            glyph(c, (float)bx + 30, (float)by + bh * 0.5f, bs[i].g, bs[i].on);
            c.text(smallb_, bx + 56, by + 36, bs[i].text, bs[i].on ? kText : kMuted, bw - 64);
        }
    };
    const bool busy = v.status.phase == app::Status::Loading || v.status.phase == app::Status::Checking ||
                      v.status.phase == app::Status::Benchmarking || v.status.queued > 0;

    // MEMORY NEEDED, its label's baseline at y: a meter and a line.
    auto memory = [&](int y, uint64_t need, const app::ModelInfo::Ctx &ctx, bool about) {
        const bool fits = !v.gpu_memory || (ctx.n > 0 && need <= v.gpu_memory);
        label(c, smallb_, DX, y, "MEMORY NEEDED");
        c.round_rect((float)DX, (float)y + 16, (float)DW, 14, 7, kTrack);
        if (v.gpu_memory) {
            const float f = std::min(1.0f, (float)need / (float)v.gpu_memory);
            c.round_rect((float)DX, (float)y + 16, std::max(14.0f, DW * f), 14, 7, fits ? kAccent : kRed);
        }
        std::string mem = app::format_bytes(need);
        if (v.gpu_memory) mem += " of the GPU's " + app::format_bytes(v.gpu_memory);
        // An estimate ("About"), so shorter: it has to fit the same line.
        mem = fits ? (about ? "About " : "") + mem + "  \xC2\xB7  room for " + plain::count(ctx.n) +
                         (about ? " tokens" : " tokens of conversation")
                   : "Too big for this PS5's GPU: it needs " + mem;
        c.text(small_, DX, y + 62, mem, fits ? kText2 : kRed, DW);
    };

    if (it.kind == Item::Catalog) {
        using D = app::Download;
        const app::CatalogEntry &e = app::kCatalog[it.index];
        const app::Download d = download_of(v, e.key);
        c.text(h2_, DX, 244, e.name, kText, DW);
        const char *state = d.phase == D::None      ? "Not on your PS5 yet"
                            : d.phase == D::Queued  ? "Waiting to download"
                            : d.phase == D::Running ? (d.held ? "Waiting while a model loads" : "Downloading")
                            : d.phase == D::Retrying ? "Waiting to try again"
                                                     : "Stopped";
        c.text(small_, DX, 278, state, kText2, DW);
        cell(DX, 306, "GOOD FOR", e.good_for);
        cell(col2, 306, "DOWNLOAD SIZE", app::format_bytes(d.total ? d.total : e.bytes));
        // The estimate the model will show once it's here, from the header
        // fields the catalog keeps.
        const app::ModelInfo est = app::estimate(e);
        const app::ModelInfo::Ctx ctx = est.serve_ctx(v.n_ctx, v.gpu_memory);
        memory(414, est.memory_for(ctx), ctx, true);

        label(c, smallb_, DX, 526, "DOWNLOAD");
        if (d.phase == D::None) {
            text_glyph(c, body_, DX, 570, "Press", kGlyphCross, "to download it to your PS5.", kText);
            c.text(small_, DX, 610, "It's ready to use when the download finishes.", kText2, DW);
            if (v.have_free)
                c.text(small_, DX, 644, "Free space on your PS5: " + app::format_bytes(v.free_bytes), kMuted, DW);
        } else {
            c.round_rect((float)DX, 542, (float)DW, 14, 7, kTrack);
            const float f = d.total ? std::min(1.0f, (float)d.done / (float)d.total) : 0;
            const Color mc = d.phase == D::Failed ? kRed : d.phase == D::Running && !d.held ? kAccent : kText2;
            if (f > 0) c.round_rect((float)DX, 542, std::max(14.0f, DW * f), 14, 7, mc);
            c.text(small_, DX, 590, std::to_string(percent(d)) + "%  \xC2\xB7  " + progress_text(d, true), kText2, DW);
            int y = 632;
            auto para = [&](const std::string &text, Color col) {
                for (auto &line : wrap(small_, text, DW)) {
                    if (y > 740) break;
                    c.text(small_, DX, y, line, col);
                    y += 32;
                }
            };
            const std::string other = downloading(v);
            switch (d.phase) {
                case D::Running:
                    para(d.held ? "It waits while a model loads, then goes on by itself."
                                : "It's ready to use when the download finishes.",
                         kText2);
                    break;
                case D::Queued:
                    para(other.empty() ? std::string("It starts in a moment.")
                                       : "It starts once " + other + " has downloaded.",
                         kText2);
                    break;
                case D::Retrying:
                    para(d.problem + " Trying again in " +
                             std::to_string(std::max(1, (int)std::ceil(d.retry_at - v.now))) + " s.",
                         kAmber);
                    break;
                case D::Stopped:
                    text_glyph(c, small_, DX, y, "Stopped. Press", kGlyphCross, "to go on.", kText2);
                    text_glyph(c, small_, DX, y + 36, "Press", kGlyphSquare, "to delete the part downloaded.", kMuted);
                    break;
                case D::Failed:
                    para(d.problem, kRed);
                    if (d.done > 0 && y <= 740)
                        text_glyph(c, small_, DX, y + 4, "Press", kGlyphSquare, "to delete the part downloaded.", kMuted);
                    break;
                case D::None: break;
            }
        }
        buttons({{kGlyphCross, d.active() ? "Stop" : d.done > 0 ? "Go on" : "Download", true},
                 {kGlyphSquare, "Delete part", !d.active() && d.done > 0},
                 {kGlyphCircle, "Home", true}});
        return;
    }
    if (it.kind != Item::Model) return;
    const app::ModelInfo &m = v.models[it.index];
    const app::Results r = v.results.count(m.file) ? v.results.at(m.file) : app::Results{};
    c.text(h2_, DX, 244, shown_name(v.models, m), kText, DW);
    c.text(small_, DX, 278, plain::capital("added " + ago(m.mtime, v.wall)), kText2, DW);
    cell(DX, 306, "GOOD FOR", good_for(m));
    cell(col2, 306, "SIZE", m.params > 0 ? plain::parameters(m.params) : "\xE2\x80\x94");
    cell(DX, 390, "PRECISION", m.quant.empty() ? "\xE2\x80\x94" : plain::quant_detail(m.quant));
    cell(col2, 390, "FILE SIZE", app::format_bytes(m.size));

    if (m.copying || !m.runnable) {
        // Why it can't run, in plain words; the engine's own words are in the
        // log (the title logs each model it can't open).
        label(c, smallb_, DX, 500, m.copying ? "COPYING" : "CAN'T RUN IT");
        const std::string why = m.copying      ? copied(m) + ". It's ready when the copy finishes."
                                : m.incomplete ? "This file is incomplete. Copy or download it again."
                                               : plain::sentence(plain::reason(m.error));
        int y = 540;
        for (auto &line : wrap(body_, why, DW)) {
            if (y > 700) break;
            c.text(body_, DX, y, line, m.copying ? kAmber : kRed);
            y += 36;
        }
        if (!m.copying) text_glyph(c, small_, DX, y + 12, "Press", kGlyphSquare, "for the details, in Logs.", kText2);
    } else {
        const bool is_serving = m.path == v.status.serving_path;
        // Serving: its context. Else what serving it would get: the args'
        // -c, or what fits (no -c).
        const app::ModelInfo::Ctx ctx = is_serving && v.status.serving_ctx
                                            ? app::ModelInfo::Ctx{v.status.serving_ctx, v.status.serving_kv8}
                                            : m.serve_ctx(v.n_ctx, v.gpu_memory);
        memory(500, m.memory_for(ctx), ctx, false);

        label(c, smallb_, DX, 612, "GPU CHECK");
        if (r.has_check) {
            const int w = c.text(bodyb_, DX, 648, r.check.pass ? "\xE2\x9C\x93 Passed: " : "\xE2\x9C\x97 Failed: ",
                                 r.check.pass ? kGreen : kRed);
            text_with_time(c, body_, DX + w, 648,
                           r.check.pass ? "same answers as the CPU" : "its answers don't match the CPU's",
                           ago(r.check.when, v.wall), kText2, DW - w);
        } else {
            text_glyph(c, body_, DX, 648, "Not run yet. Press", kGlyphSquare, "to check its answers on the PS5.", kMuted);
        }
        label(c, smallb_, DX, 696, "SPEED");
        if (r.has_bench) {
            char speeds[96];
            snprintf(speeds, sizeof speeds, "Writes %.0f, reads %.0f tokens a second", r.bench.tg_tps, r.bench.pp_tps);
            text_with_time(c, body_, DX, 732, speeds, ago(r.bench.when, v.wall), kText, DW);
        } else {
            text_glyph(c, body_, DX, 732, "Not measured yet. Press", kGlyphTriangle, "to time it.", kMuted);
        }
    }
    const bool can = m.runnable && !busy;
    const bool is_serving = m.path == v.status.serving_path;
    buttons({{kGlyphCross, is_serving ? "Stop" : "Start", can},
             cant_run(m) ? Btn{kGlyphSquare, "Details", true} : Btn{kGlyphSquare, "GPU check", can},
             {kGlyphTriangle, "Speed test", can},
             {kGlyphCircle, "Home", true}});
}

void Dashboard::draw_activity(Canvas &c, const View &v, const Item *sel) {
    const auto &st = v.status;
    const int x = LX + 32;
    std::string line;
    Color col = kText;
    bool spin = true;
    const std::string took = elapsed(v.now - st.job_started);
    switch (st.phase) {
        case app::Status::Loading: line = "Loading " + name_of(v, st.job_model) + "  \xC2\xB7  " + took; break;
        case app::Status::Checking:
            line = "GPU check on " + name_of(v, st.job_model) + ": " + (st.step.empty() ? "starting" : st.step) +
                   "  \xC2\xB7  " + took;
            break;
        case app::Status::Benchmarking:
            line = "Speed test on " + name_of(v, st.job_model) + ": " + (st.step.empty() ? "starting" : st.step) +
                   "  \xC2\xB7  " + took;
            break;
        case app::Status::Serving:
            spin = false;
            line = name_of(v, st.serving) + (v.on_network ? " is ready for apps on your network" : " is ready");
            col = kGreen;
            break;
        case app::Status::Idle:
            spin = false;
            col = kText2;
            break;
    }
    const int tx = x + 36;
    if (spin) spinner(c, (float)x + 12, AY + 42, v.now, kAccent);
    else c.circle((float)x + 12, AY + 42, 7, col);
    if (st.phase == app::Status::Idle)
        text_glyph(c, bodyb_, tx, AY + 52, "Nothing running. Choose a model and press", kGlyphCross,
                   v.models.empty() ? "to download it." : "to start it.", col);
    else
        c.text(bodyb_, tx, AY + 52, line, col, 1080 - tx);

    // The newest events under it.
    int y = AY + 90;
    for (int i = (int)v.events.size() - 1, shown = 0; i >= 0 && shown < 2; i--, shown++) {
        const app::Event &e = v.events[i];
        const int64_t secs = (int64_t)(v.now - e.when);
        text_with_time(c, small_, tx, y, e.text, ago(0, secs), e.bad ? kRed : kText2, 1080 - tx);
        y += 30;
    }

    const int ax = 1180, aw = 1848 - 32 - ax;
    // Downloading: the room there is for it.
    if (sel && sel->kind == Item::Catalog) {
        if (!v.have_free) return;
        const app::CatalogEntry &e = app::kCatalog[sel->index];
        const app::Download d = download_of(v, e.key);
        const uint64_t total = d.total ? d.total : e.bytes, rest = total > d.done ? total - d.done : 0;
        label(c, smallb_, ax, AY + 40, "STORAGE");
        c.text(bodyb_, ax, AY + 78, app::format_bytes(v.free_bytes) + " free", kText, aw);
        if (rest > 0)
            c.text(small_, ax, AY + 112,
                   v.free_bytes > rest ? "About " + app::format_bytes(v.free_bytes - rest) + " once it's downloaded"
                                       : std::string("Not enough for this download"),
                   v.free_bytes > rest ? kText2 : kRed, aw);
        return;
    }
    // The chosen model's use over the last 7 days (Home has the rest).
    if (!sel || sel->kind != Item::Model) return;
    const app::ModelInfo &m = v.models[sel->index];
    label(c, smallb_, ax, AY + 40, "IN USE, LAST 7 DAYS");
    const app::UsageSums *week = week_of(v, m.name);
    if (!v.usage || !v.usage->ready) {
        c.text(small_, ax, AY + 78, "Reading the usage history\xE2\x80\xA6", kMuted, aw);
    } else if (!week || !week->requests) {
        c.text(bodyb_, ax, AY + 78, "Not used", kText2, aw);
        c.text(small_, ax, AY + 112, "No requests to it in the last 7 days", kMuted, aw);
    } else {
        c.text(bodyb_, ax, AY + 78,
               plain::count(week->requests) + (week->requests == 1 ? " request" : " requests") + "  \xC2\xB7  " +
                   plain::count(week->prompt + week->gen) + " tokens",
               kText, aw);
        const std::string s = week->gen >= 32 && week->gen_ms > 0
                                  ? "Wrote " + tps(week->gen * 1000.0 / week->gen_ms) + " on average"
                                  : std::string("No replies written");
        c.text(small_, ax, AY + 112, s, kText2, aw);
    }
}

void Dashboard::draw_close_hint(Canvas &c, int x, int y) {
    const int ow = Canvas::measure(smallb_, "OPTIONS") + 28;
    c.round_rect((float)x, (float)y - 25, (float)ow, 34, 17, rgb(0x1E2850));
    c.text(smallb_, x + 14, y - 1, "OPTIONS", kText2);
    c.text(small_, x + ow + 12, y, "Close", kText2);
}

void Dashboard::draw_home_footer(Canvas &c, const View &v) {
    int x = LX;
    const int y = 1046;
    x += key_cap(c, smallb_, x, y, "L1") + 12;
    x += c.text(small_, x, y, page_name(next(Home)), kText2) + 40;
    x += c.text(small_, x, y, "\xE2\x86\x90\xE2\x86\x92  24 hours, 7 days or 30 days", kText2) + 40;
    glyph(c, (float)x + 16, (float)y - 8, kGlyphCross);
    x += 40 + c.text(small_, x + 40, y, v.status.phase == app::Status::Serving ? "Chat" : "Models", kText2) + 36;
    if (home_problems(v)) {
        glyph(c, (float)x + 16, (float)y - 8, kGlyphSquare);
        x += 40 + c.text(small_, x + 40, y, "Why some failed", kText2) + 36;
    }
    draw_close_hint(c, x, y);
}

void Dashboard::draw_logs_footer(Canvas &c) {
    int x = LX;
    const int y = 1046;
    x += key_cap(c, smallb_, x, y, "L1") + 12;
    x += c.text(small_, x, y, page_name(next(Logs)), kText2) + 40;
    x += c.text(small_, x, y, "\xE2\x86\x91\xE2\x86\x93  Scroll", kText2) + 40;
    x += c.text(small_, x, y, "\xE2\x86\x90\xE2\x86\x92  A page", kText2) + 40;
    struct {
        PadGlyph g;
        const char *text;
    } hints[] = {{kGlyphSquare, logs_.problems() ? "Everything" : "Problems only"},
                 {kGlyphTriangle, logs_.last_run() ? "This run" : "The run before"},
                 {kGlyphCircle, "Home"}};
    for (auto &h : hints) {
        glyph(c, (float)x + 16, (float)y - 8, h.g);
        x += 40 + c.text(small_, x + 40, y, h.text, kText2) + 36;
    }
    draw_close_hint(c, x, y);
}

void Dashboard::draw_footer(Canvas &c, const View &v, const Item *sel) {
    int x = LX;
    const int y = 1046;
    x += key_cap(c, smallb_, x, y, "L1") + 12;
    x += c.text(small_, x, y, page_name(next(Models)), kText2) + 40;
    x += c.text(small_, x, y, "\xE2\x86\x91\xE2\x86\x93  Choose", kText2) + 40;
    struct Hint {
        PadGlyph g;
        std::string text;
        bool on;
        bool right = false;  // right on the pad, not a button
    };
    std::vector<Hint> hints;
    if (sel && sel->kind == Item::Catalog) {
        const app::Download d = download_of(v, app::kCatalog[sel->index].key);
        hints = {{kGlyphCross, d.active() ? "Stop the download" : d.done > 0 ? "Go on" : "Download", true},
                 {kGlyphSquare, "Delete part", !d.active() && d.done > 0}};
    } else {
        const app::ModelInfo *m = sel && sel->kind == Item::Model ? &v.models[sel->index] : nullptr;
        hints = {{kGlyphCross, "Start or stop", true},
                 {kGlyphSquare, m && cant_run(*m) ? "Details" : "GPU check", true},
                 {kGlyphTriangle, "Speed test", true},
                 {kGlyphCross, "\xE2\x86\x92  Remove", m && removable(v, *m), true}};
    }
    hints.push_back({kGlyphCircle, "Home", true});
    for (auto &h : hints) {
        if (h.right) {
            x += c.text(small_, x, y, h.text, h.on ? kText2 : kMuted) + 40;
            continue;
        }
        glyph(c, (float)x + 16, (float)y - 8, h.g, h.on);
        x += 40 + c.text(small_, x + 40, y, h.text, h.on ? kText2 : kMuted) + 36;
    }
    draw_close_hint(c, x, y);
}

void Dashboard::draw_chat_footer(Canvas &c, const View &v) {
    int x = LX;
    const int y = 1046;
    x += key_cap(c, smallb_, x, y, "L1") + 12;
    x += c.text(small_, x, y, page_name(next(Chat)), kText2) + 40;
    const bool talk = !v.voice.model.empty();
    x += key_cap(c, smallb_, x, y, "R1", talk) + 12;
    x += c.text(small_, x, y, "Hold to talk", talk ? kText2 : kMuted) + 40;
    x += c.text(small_, x, y, "\xE2\x86\x91\xE2\x86\x93  Scroll", kText2) + 40;
    const bool choose = nothing_running(v);
    struct {
        PadGlyph g;
        const char *text;
        bool on;
    } hints[] = {{kGlyphCross, choose ? "Choose a model" : "Type", choose || can_type(v)},
                 {kGlyphTriangle, v.speaking && !v.chat.busy ? "Stop speaking" : "Stop the reply",
                  v.chat.busy || v.speaking},
                 {kGlyphSquare, "New chat", !v.chat.turns.empty()},
                 {kGlyphCircle, "Home", true}};
    for (auto &h : hints) {
        glyph(c, (float)x + 16, (float)y - 8, h.g, h.on);
        x += 40 + c.text(small_, x + 40, y, h.text, h.on ? kText2 : kMuted) + 36;
    }
    draw_close_hint(c, x, y);
}

void Dashboard::draw_settings_footer(Canvas &c) {
    int x = LX;
    const int y = 1046;
    x += key_cap(c, smallb_, x, y, "L1") + 12;
    x += c.text(small_, x, y, page_name(next(Settings)), kText2) + 40;
    x += c.text(small_, x, y, "\xE2\x86\x91\xE2\x86\x93  Choose", kText2) + 40;
    x += c.text(small_, x, y, "\xE2\x86\x90\xE2\x86\x92  Change", kText2) + 40;
    struct {
        PadGlyph g;
        const char *text;
    } hints[] = {{kGlyphSquare, "Default"}, {kGlyphCircle, "Home"}};
    for (auto &h : hints) {
        glyph(c, (float)x + 16, (float)y - 8, h.g);
        x += 40 + c.text(small_, x + 40, y, h.text, kText2) + 36;
    }
    draw_close_hint(c, x, y);
}

void Dashboard::draw_confirm(Canvas &c) {
    const bool stop = confirm_ == ConfirmStop, rm = confirm_ == ConfirmRemove;
    c.fill(0, 0, W, H, alpha(rgb(0x000000), 170));
    const int w = 760, x = (W - w) / 2;
    const std::string head = stop ? "Stop " + confirm_name_ + "?"
                             : rm ? "Remove " + confirm_name_ + " from your PS5?"
                                  : std::string("Close TokenStation5?");
    const std::string body = stop ? "Apps can't use TokenStation5 until you start a model again."
                             : rm ? "It frees " + app::format_bytes(confirm_bytes_) + ". You can download it again."
                                  : "Apps on your network can't use TokenStation5 until you open it again.";
    const auto head_lines = wrap(h2_, head, w - 96), body_lines = wrap(body_, body, w - 96);
    const int h = 300 + 44 * ((int)head_lines.size() - 1) + 34 * std::max(0, (int)body_lines.size() - 2);
    const int y = (H - h) / 2;
    c.round_rect((float)x, (float)y, (float)w, (float)h, 24, rgb(0x111A38));
    c.round_rect_line((float)x, (float)y, (float)w, (float)h, 24, 2, rgb(0x2A3A70));
    int ty = y + 80;
    for (auto &line : head_lines) {
        c.text(h2_, x + 48, ty, line, kText, w - 96);
        ty += 44;
    }
    for (auto &line : body_lines) {
        c.text(body_, x + 48, ty, line, kText2);
        ty += 34;
    }
    const int bw = 220, by = y + h - 104;
    for (int i = 0; i < 2; i++) {
        const int bx = x + 48 + i * (bw + 16);
        c.round_rect((float)bx, (float)by, (float)bw, 60, 14, rgb(0x18244C));
        glyph(c, (float)bx + 32, (float)by + 30, i == 0 ? kGlyphCross : kGlyphCircle);
        c.text(bodyb_, bx + 60, by + 40,
               i == 0 ? (stop ? "Stop" : rm ? "Remove" : "Close") : stop || rm ? "Keep it" : "Cancel", kText);
    }
}

HomeInfo Dashboard::home_info(const View &v) const {
    HomeInfo h;
    h.usage = v.usage;
    h.live = v.live;
    h.reading = v.live_reading;
    h.status = v.status;
    if (!v.status.serving.empty()) h.running = name_of(v, v.status.serving);
    if (!v.status.job_model.empty()) h.job = name_of(v, v.status.job_model);
    h.api = v.api;
    h.own_ip = v.own_ip;
    h.url = v.url;
    h.on_network = v.on_network;
    h.now = v.now;
    for (const auto &[key, d] : v.downloads)
        if (d.phase == app::Download::Running)
            if (const app::CatalogEntry *e = app::Downloads::find(key))
                h.download = std::string("Downloading ") + e->name + " \xC2\xB7 " + std::to_string(percent(d)) + "%";
    return h;
}

void Dashboard::draw(Canvas &c, const View &v) {
    draw_background(c);
    draw_header(c, v);
    if (page_ == Home) {
        home_.draw(c, home_info(v));
        draw_home_footer(c, v);
        if (confirm_ != NoConfirm) draw_confirm(c);
        return;
    }
    if (page_ == Logs) {
        logs_.draw(c, LX, PT, 1848 - LX, SB - PT, v.version, v.now);
        draw_logs_footer(c);
        if (confirm_ != NoConfirm) draw_confirm(c);
        return;
    }
    if (page_ == Settings) {
        const SettingsInfo info = settings_info(v);
        settings_.draw(c, info, LX, PT, LW, SB - PT);
        settings_.draw_details(c, info, RX, PT, RW, SB - PT);
        draw_settings_footer(c);
        if (confirm_ != NoConfirm) draw_confirm(c);
        return;
    }
    if (page_ == Chat) {
        chat_.draw(c, v.chat, CX, PT, CW, CB - PT, v.now, [&](const std::string &f) { return name_of(v, f); },
                   v.status.phase == app::Status::Serving ? v.status.serving : std::string(), !v.voice.model.empty());
        chat_.draw_input(c, v.chat, v.voice, v.now, CX, IY, CW, IH);
        draw_chat_footer(c, v);
        if (confirm_ != NoConfirm) draw_confirm(c);
        return;
    }
    const auto list = items(v);
    int sel = selected(list);
    if (sel >= 0 && selected_key_.empty()) {
        for (size_t i = 0; i < list.size(); i++)
            if (list[i].kind == Item::Model && v.models[list[i].index].path == v.status.serving_path) sel = (int)i;
    }
    draw_list(c, v, list, sel);
    if (sel >= 0) draw_details(c, v, list[sel]);
    draw_activity(c, v, sel >= 0 ? &list[sel] : nullptr);
    draw_footer(c, v, sel >= 0 ? &list[sel] : nullptr);
    if (confirm_ != NoConfirm) draw_confirm(c);
}

}  // namespace ui
