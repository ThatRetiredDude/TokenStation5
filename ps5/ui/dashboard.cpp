#include "dashboard.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "pad.h"

namespace ui {

namespace {

constexpr int W = 1920, H = 1080;
// Panels: models on the left, details on the right, activity below.
constexpr int PT = 192, PB = 844;
constexpr int LX = 72, LW = 900, RX = 1000, RW = 848;
constexpr int AY = 868, AH = 136;
constexpr int LIST_X = LX + 16, LIST_W = LW - 32, LIST_Y0 = 268, LIST_Y1 = PB - 16;
constexpr int DX = RX + 32, DW = RW - 64;

constexpr Color kBgTop = rgb(0x0B1226), kBgBottom = rgb(0x05070F);
constexpr Color kPanel = rgb(0x0E1530), kPanelLine = rgb(0x1B2650);
constexpr Color kRow = rgb(0x121B3A), kRowSel = rgb(0x17244E);
constexpr Color kText = rgb(0xEAF0FF), kText2 = rgb(0x9AA6C8), kMuted = rgb(0x627095);
constexpr Color kAccent = rgb(0x3CC8FF), kGreen = rgb(0x3DDC97), kRed = rgb(0xFF5C70), kAmber = rgb(0xFFB547);
constexpr Color kCrossColor = rgb(0x8FB0FF), kCircleColor = rgb(0xFF6F7F), kSquareColor = rgb(0xFF8BD0),
                kTriangleColor = rgb(0x4BE0A8);

// Models worth starting with that ts5 runs, for a console that has none of
// them yet. `id` finds one already there by its file name; `key` is its line
// in models.tsv for scripts/ts5-ps5.sh fetch.
struct Suggestion {
    const char *key, *id, *name, *quant, *size, *role;
};
const Suggestion kSuggestions[] = {
    {"qwen25-0.5b", "qwen2.5-0.5b", "Qwen2.5 0.5B Instruct", "Q8_0", "0.7 GB", "A quick test: tiny and very fast"},
    {"llama32-3b", "llama-3.2-3b", "Llama 3.2 3B Instruct", "Q4_K_M", "2.0 GB", "Small and fast everyday chat"},
    {"qwen25-coder-7b", "qwen2.5-coder-7b", "Qwen2.5 Coder 7B Instruct", "Q4_K_M", "4.7 GB", "Coding"},
    {"qwen3-8b", "qwen3-8b", "Qwen3 8B", "Q4_K_M", "5.0 GB", "General chat, with thinking"},
    {"llama31-8b", "llama-3.1-8b", "Llama 3.1 8B Instruct", "Q4_K_M", "4.9 GB", "General chat"},
};

std::string lower(std::string s) {
    for (auto &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

std::string arch_name(const std::string &a) {
    if (a == "qwen2") return "Qwen2";
    if (a == "qwen3") return "Qwen3";
    if (a == "llama") return "Llama";
    if (a == "mistral") return "Mistral";
    return a.empty() ? "Unknown" : a;
}

std::string ago(int64_t then, int64_t now) {
    const int64_t s = std::max<int64_t>(0, now - then);
    char b[48];
    if (s < 60) return "just now";
    if (s < 3600) snprintf(b, sizeof b, "%lld min ago", (long long)(s / 60));
    else if (s < 86400) snprintf(b, sizeof b, "%lld h ago", (long long)(s / 3600));
    else snprintf(b, sizeof b, "%lld days ago", (long long)(s / 86400));
    return b;
}

std::string elapsed(double s) {
    char b[32];
    if (s < 60) snprintf(b, sizeof b, "%.0f s", s);
    else snprintf(b, sizeof b, "%d min %02d s", (int)s / 60, (int)s % 60);
    return b;
}

std::string tps(double v) {
    char b[32];
    snprintf(b, sizeof b, v >= 100 ? "%.0f tok/s" : "%.1f tok/s", v);
    return b;
}

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

enum PadGlyph { kGlyphCross, kGlyphCircle, kGlyphSquare, kGlyphTriangle };

void glyph(Canvas &c, float cx, float cy, PadGlyph g, bool enabled = true) {
    c.circle(cx, cy, 16, rgb(0x1E2850));
    Color col = g == kGlyphCross    ? kCrossColor
                : g == kGlyphCircle ? kCircleColor
                : g == kGlyphSquare ? kSquareColor
                                    : kTriangleColor;
    if (!enabled) col = kMuted;
    switch (g) {
        case kGlyphCross:
            c.segment(cx - 6, cy - 6, cx + 6, cy + 6, 3, col);
            c.segment(cx - 6, cy + 6, cx + 6, cy - 6, 3, col);
            break;
        case kGlyphCircle: c.ring(cx, cy, 7, 3, col); break;
        case kGlyphSquare: c.round_rect_line(cx - 7, cy - 7, 14, 14, 2, 3, col); break;
        case kGlyphTriangle:
            c.segment(cx, cy - 7.5f, cx - 8, cy + 6, 3, col);
            c.segment(cx - 8, cy + 6, cx + 8, cy + 6, 3, col);
            c.segment(cx + 8, cy + 6, cx, cy - 7.5f, 3, col);
            break;
    }
}

// A rounded label right-aligned at `right`; returns its left edge.
int pill(Canvas &c, const Font &f, int right, int top, const std::string &text, Color col) {
    const int w = Canvas::measure(f, text) + 32, h = 34;
    c.round_rect((float)(right - w), (float)top, (float)w, (float)h, h * 0.5f, alpha(col, 46));
    c.text(f, right - w + 16, top + 24, text, col);
    return right - w;
}

void spinner(Canvas &c, float cx, float cy, double t, Color col) {
    const int phase = (int)(t * 12) % 8;
    for (int i = 0; i < 8; i++) {
        const float a = (float)(i * M_PI / 4);
        const int age = (phase - i + 8) % 8;
        c.circle(cx + 11 * std::cos(a), cy + 11 * std::sin(a), 3.2f, alpha(col, 255 - age * 28));
    }
}

void label(Canvas &c, const Font &f, int x, int y, const std::string &s) { c.text(f, x, y, s, kMuted); }

// A model's name (general.name) for its file name.
std::string name_of(const View &v, const std::string &file) {
    for (const auto &m : v.models)
        if (m.file == file) return m.name;
    return file;
}

// Text with a controller button drawn in it: before, the button, after.
void text_glyph(Canvas &c, const Font &f, int x, int y, const std::string &before, PadGlyph g, const std::string &after,
                Color col) {
    x += c.text(f, x, y, before, col) + 6;
    glyph(c, (float)x + 16, (float)y - f.px * 0.33f, g);
    c.text(f, x + 38, y, after, col);
}

}  // namespace

struct Dashboard::Item {
    enum Kind { Model, Header, Suggestion, Empty } kind;
    int index;  // into View::models or kSuggestions
    int h;
    std::string key;
};

Dashboard::Dashboard(const Fonts &fonts, const uint8_t *logo, size_t logo_size)
    : title_(fonts.get("title")), h2_(fonts.get("h2")), body_(fonts.get("body")),
      bodyb_(fonts.get("bodyb")), small_(fonts.get("small")), smallb_(fonts.get("smallb")), mono_(fonts.get("mono")) {
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
    std::vector<Item> sugg;
    for (size_t i = 0; i < sizeof kSuggestions / sizeof kSuggestions[0]; i++) {
        const bool have = std::any_of(v.models.begin(), v.models.end(), [&](const app::ModelInfo &m) {
            return lower(m.file).find(kSuggestions[i].id) != std::string::npos;
        });
        if (!have) sugg.push_back({Item::Suggestion, (int)i, 88, std::string("suggest:") + kSuggestions[i].key});
    }
    if (!sugg.empty()) {
        out.push_back({Item::Header, 0, 56, ""});
        out.insert(out.end(), sugg.begin(), sugg.end());
    }
    return out;
}

int Dashboard::selected(const std::vector<Item> &items) const {
    int first = -1;
    for (size_t i = 0; i < items.size(); i++) {
        if (items[i].kind != Item::Model && items[i].kind != Item::Suggestion) continue;
        if (first < 0) first = (int)i;
        if (items[i].key == selected_key_) return (int)i;
    }
    return first;
}

bool Dashboard::animating(const View &v) const {
    return v.status.phase == app::Status::Loading || v.status.phase == app::Status::Checking ||
           v.status.phase == app::Status::Benchmarking || v.api.generating;
}

Action Dashboard::press(uint32_t b, const View &v) {
    Action a;
    if (!b) return a;
    if (confirm_close_) {
        if (b & kCross) a.kind = Action::Close;
        if (b & (kCross | kCircle | kOptions)) confirm_close_ = false;
        return a;
    }
    if (b & kOptions) {
        confirm_close_ = true;
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
            if (list[i].kind == Item::Model || list[i].kind == Item::Suggestion) return i;
        return sel;
    };
    if (b & kUp) sel = step(-1);
    if (b & kDown) sel = step(+1);
    selected_key_ = list[sel].key;
    const Item &it = list[sel];
    const bool busy = v.status.phase == app::Status::Loading || v.status.phase == app::Status::Checking ||
                      v.status.phase == app::Status::Benchmarking || v.status.queued > 0;
    if (busy) return a;
    if (it.kind == Item::Model && v.models[it.index].runnable) {
        a.path = v.models[it.index].path;
        if ((b & kCross) && a.path != v.status.serving_path) a.kind = Action::Serve;
        else if (b & kSquare) a.kind = Action::Check;
        else if (b & kTriangle) a.kind = Action::Bench;
    }
    if ((b & kCircle) && !v.status.serving.empty()) a.kind = Action::Stop;
    return a;
}

void Dashboard::draw_background(Canvas &c) {
    if (background_.empty()) {
        Canvas bg(W, H);
        bg.gradient(0, 0, W, H, kBgTop, kBgBottom);
        bg.fill(LX, 168, 1848 - LX, 1, kPanelLine);
        for (auto [x, y, w, h] : {std::array<int, 4>{LX, PT, LW, PB - PT}, std::array<int, 4>{RX, PT, RW, PB - PT},
                                   std::array<int, 4>{LX, AY, 1848 - LX, AH}}) {
            bg.round_rect((float)x, (float)y, (float)w, (float)h, 20, kPanel);
            bg.round_rect_line((float)x, (float)y, (float)w, (float)h, 20, 1.5f, kPanelLine);
        }
        background_.assign(bg.pixels(), bg.pixels() + (size_t)W * H);
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
    c.text(small_, tx, 136, "Local AI on your PS5  \xC2\xB7  build " + v.version, kText2);

    const auto &st = v.status;
    std::string state = "NOT SERVING";
    Color col = kText2;
    switch (st.phase) {
        case app::Status::Loading: state = "LOADING", col = kAmber; break;
        case app::Status::Checking: state = "GPU CHECK", col = kAccent; break;
        case app::Status::Benchmarking: state = "SPEED TEST", col = kAccent; break;
        case app::Status::Serving: state = "SERVING", col = kGreen; break;
        case app::Status::Idle: break;
    }
    const int left = pill(c, smallb_, 1848, 48, state, col);
    c.circle((float)left - 18, 65, 7, col);
    const bool up = !st.serving.empty() && st.phase == app::Status::Serving;
    c.text_right(smallb_, 1848, 132, v.url, up ? kAccent : kText2);
    c.text_right(small_, 1848 - Canvas::measure(smallb_, v.url) - 12, 132, "API", kMuted);
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
            c.text(smallb_, LIST_X + 8, ry + 40, "SUGGESTED  \xC2\xB7  NOT ON YOUR PS5 YET", kMuted);
            continue;
        }
        if (it.kind == Item::Empty) {
            c.round_rect_line((float)LIST_X, (float)ry, (float)LIST_W, (float)it.h, 14, 1.5f, kPanelLine);
            c.text(bodyb_, LIST_X + 24, ry + 44, "No models yet", kText);
            c.text(small_, LIST_X + 24, ry + 80, "Copy a .gguf over FTP to", kText2);
            c.text(small_, LIST_X + 24, ry + 110, v.folder, kAccent);
            continue;
        }
        c.round_rect((float)LIST_X, (float)ry, (float)LIST_W, (float)it.h, 14, is_sel ? kRowSel : kRow);
        if (is_sel) c.round_rect_line((float)LIST_X, (float)ry, (float)LIST_W, (float)it.h, 14, 2.5f, kAccent);
        if (it.kind == Item::Suggestion) {
            const Suggestion &s = kSuggestions[it.index];
            const int pl = pill(c, smallb_, right, ry + 14, "SUGGESTED", kMuted);
            c.text(bodyb_, LIST_X + 24, ry + 40, s.name, kText2, pl - 16 - (LIST_X + 24));
            c.text(small_, LIST_X + 24, ry + 72, std::string(s.quant) + "  \xC2\xB7  " + s.size + "  \xC2\xB7  " + s.role,
                   kMuted, LIST_W - 48);
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
        else if (m.path == v.status.serving_path) pl = pill(c, smallb_, right, ry + 14, "SERVING", kGreen);
        else if (!m.runnable) pl = pill(c, smallb_, right, ry + 14, "CAN'T RUN", kRed);
        c.text(bodyb_, LIST_X + 24, ry + 40, m.name, m.runnable ? kText : kText2, pl - 16 - (LIST_X + 24));

        // Results, right-aligned under the label.
        int rx = right;
        if (r.has_bench) rx = c.text_right(small_, rx, ry + 72, tps(r.bench.tg_tps), kText2) - 20;
        if (r.has_check)
            rx = c.text_right(smallb_, rx, ry + 72, r.check.pass ? "\xE2\x9C\x93 GPU check" : "\xE2\x9C\x97 GPU check",
                              r.check.pass ? kGreen : kRed) - 20;
        std::string line = arch_name(m.arch);
        if (m.params > 0) line += "  \xC2\xB7  " + app::format_params(m.params);
        if (!m.quant.empty()) line += "  \xC2\xB7  " + m.quant;
        line += "  \xC2\xB7  " + app::format_bytes(m.size);
        c.text(small_, LIST_X + 24, ry + 72, line, kText2, rx - (LIST_X + 24));
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
    const bool serving_any = !v.status.serving.empty();

    if (it.kind == Item::Suggestion) {
        const Suggestion &s = kSuggestions[it.index];
        c.text(h2_, DX, 244, s.name, kText, DW);
        c.text(small_, DX, 280, "Suggested  \xC2\xB7  not on your PS5 yet", kText2);
        cell(DX, 306, "DOWNLOAD SIZE", s.size);
        cell(col2, 306, "QUANTIZATION", s.quant);
        cell(DX, 390, "GOOD FOR", s.role, DW);
        label(c, smallb_, DX, 500, "TO ADD IT");
        c.text(body_, DX, 540, "On your computer, in the TokenStation5 folder, run", kText2, DW);
        c.round_rect((float)DX, 558, (float)DW, 50, 10, rgb(0x0A1024));
        c.text(mono_, DX + 18, 590, std::string("scripts/ts5-ps5.sh fetch ") + s.key, kAccent, DW - 36);
        c.text(body_, DX, 650, "or copy its .gguf over FTP to", kText2, DW);
        c.text(mono_, DX, 684, v.folder, kAccent, DW);
        c.text(small_, DX, 730, "It shows up here by itself once it's copied.", kMuted, DW);
        buttons({{kGlyphCross, "Serve", false},
                 {kGlyphSquare, "GPU check", false},
                 {kGlyphTriangle, "Speed test", false},
                 {kGlyphCircle, "Stop", serving_any && !busy}});
        return;
    }
    if (it.kind != Item::Model) return;
    const app::ModelInfo &m = v.models[it.index];
    const app::Results r = v.results.count(m.file) ? v.results.at(m.file) : app::Results{};
    c.text(h2_, DX, 244, m.name, kText, DW);
    c.text(mono_, DX, 278, m.file, kText2, DW);
    cell(DX, 306, "ARCHITECTURE", arch_name(m.arch) + (m.layers ? "  \xC2\xB7  " + std::to_string(m.layers) + " layers" : ""));
    cell(col2, 306, "PARAMETERS", m.params > 0 ? app::format_params(m.params) : "\xE2\x80\x94");
    cell(DX, 390, "QUANTIZATION", m.quant.empty() ? "\xE2\x80\x94" : m.quant);
    cell(col2, 390, "FILE SIZE", app::format_bytes(m.size));

    if (!m.runnable) {
        label(c, smallb_, DX, 500, "CAN'T RUN IT");
        int y = 540;
        for (auto &line : wrap(body_, "TokenStation5 can't run this model: " + m.error + ".", DW)) {
            if (y > 740) break;
            c.text(body_, DX, y, line, kRed);
            y += 36;
        }
    } else {
        const bool is_serving = m.path == v.status.serving_path;
        // Serving: its context. Else what serving it would get: the args'
        // -c, or what fits (no -c).
        const int ctx = is_serving && v.status.serving_ctx ? v.status.serving_ctx
                        : v.n_ctx > 0                      ? v.n_ctx
                                                           : m.auto_ctx(v.gpu_memory);
        const uint64_t need = m.memory_for(ctx);
        const bool fits = !v.gpu_memory || (ctx > 0 && need <= v.gpu_memory);
        char cap[64];
        snprintf(cap, sizeof cap, "GPU MEMORY AT %dK CONTEXT", ctx / 1024);
        label(c, smallb_, DX, 500, cap);
        c.round_rect((float)DX, 516, (float)DW, 14, 7, rgb(0x1B2650));
        if (v.gpu_memory) {
            const float f = std::min(1.0f, (float)need / (float)v.gpu_memory);
            c.round_rect((float)DX, 516, std::max(14.0f, DW * f), 14, 7, fits ? kAccent : kRed);
        }
        std::string mem = "Needs " + app::format_bytes(need);
        if (v.gpu_memory) mem += " of the GPU's " + app::format_bytes(v.gpu_memory);
        if (!fits) mem += ": too big to load";
        c.text(small_, DX, 562, mem, fits ? kText2 : kRed, DW);

        label(c, smallb_, DX, 612, "GPU CHECK");
        if (r.has_check) {
            char b[160];
            snprintf(b, sizeof b, "  \xC2\xB7  top token %d/%d  \xC2\xB7  max diff %.2f%%  \xC2\xB7  ", r.check.same_top,
                     r.check.n, r.check.rel * 100);
            const int w = c.text(bodyb_, DX, 648, r.check.pass ? "\xE2\x9C\x93 Passed" : "\xE2\x9C\x97 Failed",
                                 r.check.pass ? kGreen : kRed);
            c.text(body_, DX + w, 648, b + ago(r.check.when, v.wall), kText2, DW - w);
        } else {
            text_glyph(c, body_, DX, 648, "Not run yet. Press", kGlyphSquare, "to compare the GPU with the CPU.", kMuted);
        }
        label(c, smallb_, DX, 696, "SPEED");
        if (r.has_bench) {
            c.text(body_, DX, 732,
                   "Prompt " + tps(r.bench.pp_tps) + "  \xC2\xB7  generation " + tps(r.bench.tg_tps) + "  \xC2\xB7  " +
                       ago(r.bench.when, v.wall),
                   kText, DW);
        } else {
            c.text(body_, DX, 732, "Not measured yet.", kMuted, DW);
        }
    }
    const bool can = m.runnable && !busy;
    const bool is_serving = m.path == v.status.serving_path;
    buttons({{kGlyphCross, is_serving ? "Serving" : "Serve", can && !is_serving},
             {kGlyphSquare, "GPU check", can},
             {kGlyphTriangle, "Speed test", can},
             {kGlyphCircle, "Stop", serving_any && !busy}});
}

void Dashboard::draw_activity(Canvas &c, const View &v) {
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
            line = "Serving " + name_of(v, st.serving) + " to your network";
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
        text_glyph(c, bodyb_, tx, AY + 52, "Not serving. Choose a model and press", kGlyphCross, "to serve it.", col);
    else
        c.text(bodyb_, tx, AY + 52, line, col, 1080 - tx);

    // The newest events under it.
    int y = AY + 90;
    for (int i = (int)v.events.size() - 1, shown = 0; i >= 0 && shown < 2; i--, shown++) {
        const app::Event &e = v.events[i];
        const int64_t secs = (int64_t)(v.now - e.when);
        c.text(small_, tx, y, e.text + "  \xC2\xB7  " + ago(0, secs), e.bad ? kRed : kText2, 1080 - tx);
        y += 30;
    }

    const int ax = 1180;
    label(c, smallb_, ax, AY + 40, "API REQUESTS");
    c.text(bodyb_, ax, AY + 78, std::to_string(v.api.requests) + (v.api.requests == 1 ? " request" : " requests"), kText);
    if (v.api.generating) {
        const float pulse = 0.5f + 0.5f * std::sin((float)v.now * 6);
        c.circle(ax + 8, AY + 104, 6, alpha(kAccent, 120 + (unsigned)(pulse * 135)));
        c.text(small_, ax + 24, AY + 112, "Answering a request\xE2\x80\xA6", kAccent);
    } else if (v.api.requests > 0) {
        const auto &g = v.api.last;
        std::string s = "Last reply: ";
        if (g.prompt_evaluated > 0 && g.prompt_ms > 0) s += "prompt " + tps(g.prompt_evaluated * 1000.0 / g.prompt_ms) + "  \xC2\xB7  ";
        if (g.gen_tokens > 0 && g.gen_ms > 0) s += "generation " + tps(g.gen_tokens * 1000.0 / g.gen_ms);
        c.text(small_, ax, AY + 112, s, kText2, 1848 - 32 - ax);
    } else {
        c.text(small_, ax, AY + 112, "None yet", kMuted);
    }
}

void Dashboard::draw_footer(Canvas &c) {
    int x = LX;
    const int y = 1046;
    x += c.text(small_, x, y, "\xE2\x86\x91\xE2\x86\x93  Choose", kText2) + 40;
    struct {
        PadGlyph g;
        const char *text;
    } hints[] = {{kGlyphCross, "Serve"}, {kGlyphSquare, "GPU check"}, {kGlyphTriangle, "Speed test"}, {kGlyphCircle, "Stop serving"}};
    for (auto &h : hints) {
        glyph(c, (float)x + 16, (float)y - 8, h.g);
        x += 40 + c.text(small_, x + 40, y, h.text, kText2) + 36;
    }
    const int ow = Canvas::measure(smallb_, "OPTIONS") + 28;
    c.round_rect((float)x, (float)y - 25, (float)ow, 34, 17, rgb(0x1E2850));
    c.text(smallb_, x + 14, y - 1, "OPTIONS", kText2);
    c.text(small_, x + ow + 12, y, "Close", kText2);
}

void Dashboard::draw_confirm(Canvas &c) {
    c.fill(0, 0, W, H, alpha(rgb(0x000000), 170));
    const int w = 760, h = 300, x = (W - w) / 2, y = (H - h) / 2;
    c.round_rect((float)x, (float)y, (float)w, (float)h, 24, rgb(0x111A38));
    c.round_rect_line((float)x, (float)y, (float)w, (float)h, 24, 2, rgb(0x2A3A70));
    c.text(h2_, x + 48, y + 80, "Close TokenStation5?", kText);
    c.text(body_, x + 48, y + 130, "The API stops until you open the app again.", kText2, w - 96);
    const int bw = 220, by = y + 196;
    for (int i = 0; i < 2; i++) {
        const int bx = x + 48 + i * (bw + 16);
        c.round_rect((float)bx, (float)by, (float)bw, 60, 14, rgb(0x18244C));
        glyph(c, (float)bx + 32, (float)by + 30, i == 0 ? kGlyphCross : kGlyphCircle);
        c.text(bodyb_, bx + 60, by + 40, i == 0 ? "Close" : "Cancel", kText);
    }
}

void Dashboard::draw(Canvas &c, const View &v) {
    draw_background(c);
    draw_header(c, v);
    const auto list = items(v);
    int sel = selected(list);
    if (sel >= 0 && selected_key_.empty()) {
        for (size_t i = 0; i < list.size(); i++)
            if (list[i].kind == Item::Model && v.models[list[i].index].path == v.status.serving_path) sel = (int)i;
    }
    draw_list(c, v, list, sel);
    if (sel >= 0) draw_details(c, v, list[sel]);
    draw_activity(c, v);
    draw_footer(c);
    if (confirm_close_) draw_confirm(c);
}

}  // namespace ui
