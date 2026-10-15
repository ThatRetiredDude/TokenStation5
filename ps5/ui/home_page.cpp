#include "home_page.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "pad.h"
#include "plain.h"
#include "style.h"

namespace ui {

namespace {

using app::UsageSums;
using app::UsageView;

// Tiles across the top; under them the tokens chart and requests by app,
// then writing speed and the live line.
constexpr int TY = 192, TH = 132, TW = 426, TGAP = 24;
constexpr int KX = 72, KY = 348, KW = 1176, KH = 392;    // tokens
constexpr int AX = 1272, AY = 348, AW = 576, AH = 392;   // apps
constexpr int SX = 72, SY = 764, SW = 1176, SH = 240;    // speed
constexpr int LX = 1272, LY = 764, LW = 576, LH = 240;   // live
// The two charts' plot: the same x axis.
constexpr int PX0 = 160, PX1 = 1216;
constexpr int KTOP = 440, KBASE = 640;
constexpr int STOP = 836, SBASE = 972;
// A bucket's speed shows once it wrote this many tokens.
constexpr int64_t kSpeedMin = 32;

const char *range_words(UsageView::Range r) {
    return r == UsageView::Day ? "24 hours" : r == UsageView::Week ? "7 days" : "30 days";
}

std::string upper(std::string s) {
    for (auto &c : s) c = (char)toupper((unsigned char)c);
    return s;
}

// 950, 9.5k, 52k, 1.2M
std::string compact(int64_t n) {
    char b[32];
    if (n < 1000) snprintf(b, sizeof b, "%lld", (long long)n);
    else if (n < 10000) snprintf(b, sizeof b, "%.1fk", n / 1e3);
    else if (n < 1000000) snprintf(b, sizeof b, "%lldk", (long long)((n + 500) / 1000));
    else if (n < 10000000) snprintf(b, sizeof b, "%.1fM", n / 1e6);
    else snprintf(b, sizeof b, "%lldM", (long long)((n + 500000) / 1000000));
    std::string s = b;
    const size_t z = s.find(".0");
    if (z != std::string::npos && z + 2 < s.size() && !isdigit((unsigned char)s[z + 2])) s.erase(z, 2);
    return s;
}

// A model's name without the words every chat model has.
std::string short_name(std::string n) {
    for (const char *drop : {" Instruct", "-Instruct", " Chat", "-Chat"}) {
        const size_t at = n.find(drop);
        if (at != std::string::npos && at + strlen(drop) == n.size()) n.erase(at);
    }
    for (const char *drop : {"Meta-", "Meta "})
        if (n.rfind(drop, 0) == 0) n.erase(0, strlen(drop));
    return n;
}

// A round number at or above v: 1, 2 or 5 times a power of ten.
double nice_up(double v) {
    if (v <= 0) return 1;
    const double p = std::pow(10.0, std::floor(std::log10(v)));
    for (double m : {1.0, 2.0, 5.0, 10.0})
        if (m * p >= v) return m * p;
    return 10 * p;
}

double speed(const UsageSums &s) { return s.gen_ms > 0 ? s.gen * 1000.0 / s.gen_ms : 0; }

// An app's requests as the page names them.
std::string who(const std::string &client, const std::string &own_ip) {
    if (client == "127.0.0.1" || (!own_ip.empty() && client == own_ip)) return "Chat on this PS5";
    return client.empty() ? "an app" : client;
}

// A model's mark on the speed chart, besides its colour: a dot, a square, a
// ring, an open square, a dot in a ring.
void marker(Canvas &c, float x, float y, int slot, Color col) {
    switch (slot) {
        case 1: c.round_rect(x - 5, y - 5, 10, 10, 1.5f, col); break;
        case 2: c.ring(x, y, 5, 2, col); break;
        case 3: c.round_rect_line(x - 5, y - 5, 10, 10, 1.5f, 2, col); break;
        case 4:
            c.ring(x, y, 6, 1.5f, col);
            c.circle(x, y, 2.5f, col);
            break;
        default: c.circle(x, y, 5, col); break;
    }
}

}  // namespace

HomePage::HomePage(const Fonts &fonts)
    : h2_(fonts.get("h2")), body_(fonts.get("body")), bodyb_(fonts.get("bodyb")), small_(fonts.get("small")),
      smallb_(fonts.get("smallb")), mono_(fonts.get("mono")) {}

void HomePage::panels(std::vector<std::array<int, 4>> &out) {
    for (int i = 0; i < 4; i++) out.push_back({72 + i * (TW + TGAP), TY, TW, TH});
    out.push_back({KX, KY, KW, KH});
    out.push_back({AX, AY, AW, AH});
    out.push_back({SX, SY, SW, SH});
    out.push_back({LX, LY, LW, LH});
}

void HomePage::press(uint32_t b) {
    if (b & kLeft) range_ = range_ == UsageView::Month ? UsageView::Week : UsageView::Day;
    if (b & kRight) range_ = range_ == UsageView::Day ? UsageView::Week : UsageView::Month;
}

void HomePage::draw(Canvas &c, const HomeInfo &info) {
    draw_tiles(c, info);
    draw_tokens(c, info);
    draw_apps(c, info);
    draw_speed(c, info);
    draw_live(c, info);
}

void HomePage::draw_tiles(Canvas &c, const HomeInfo &info) {
    const UsageView *u = info.usage && info.usage->ready ? info.usage.get() : nullptr;
    const std::string range = range_words(range_), in_range = "in the last " + range;
    const std::string none = u && !u->ever ? "None yet" : "None " + in_range;
    auto tile = [&](int i, const std::string &label, const std::string &value, Color value_col) {
        const int x = 72 + i * (TW + TGAP) + 28;
        c.text(smallb_, x, TY + 38, label, kMuted, TW - 56);
        c.text(h2_, x, TY + 84, value, value_col, TW - 56);
        return x;
    };
    auto sub = [&](int x, const std::string &s, Color col) { c.text(small_, x, TY + 116, s, col, TW - 56); };

    // What runs now.
    {
        const auto &st = info.status;
        std::string value = info.running, line;
        Color col = kText2;
        const double took = info.now - st.job_started;
        char secs[32];
        snprintf(secs, sizeof secs, "%.0f s", took < 0 ? 0 : took);
        switch (st.phase) {
            case app::Status::Loading: value = info.job, line = std::string("Loading \xC2\xB7 ") + secs, col = kAmber; break;
            case app::Status::Checking:
                value = info.job, line = "GPU check \xC2\xB7 " + (st.step.empty() ? "starting" : st.step), col = kAccent;
                break;
            case app::Status::Benchmarking:
                value = info.job, line = "Speed test \xC2\xB7 " + (st.step.empty() ? "starting" : st.step), col = kAccent;
                break;
            case app::Status::Serving:
                if (info.api.generating && info.api.phase == 1) {
                    line = "Reading a " + plain::count(info.api.prompt) + "-token prompt", col = kAccent;
                } else if (info.api.generating) {
                    line = "Writing a reply";
                    if (!info.live.empty() && info.live.back() > 0)
                        line = "Writing \xC2\xB7 " + plain::tokens_a_second(info.live.back());
                    col = kAccent;
                } else {
                    line = info.on_network ? "Ready for apps on your network" : "Ready on this PS5", col = kGreen;
                }
                break;
            case app::Status::Idle: break;
        }
        const int x =
            tile(0, "RUNNING NOW", value.empty() ? "Nothing running" : short_name(value), value.empty() ? kText2 : kText);
        if (st.phase == app::Status::Idle) text_glyph(c, small_, x, TY + 116, "Press", kGlyphCross, "to choose a model", kText2);
        else sub(x, line, col);
    }
    // Tokens over the range.
    {
        const int x = tile(1, "TOKENS \xC2\xB7 " + upper(range), u ? plain::count(u->total.prompt + u->total.gen) : "\xE2\x80\x94",
                           kText);
        if (!u) sub(x, "Reading the usage history\xE2\x80\xA6", kMuted);
        else if (!u->total.requests) sub(x, none, kMuted);
        else sub(x, compact(u->total.gen) + " written \xC2\xB7 " + compact(u->total.prompt) + " read", kText2);
    }
    // Requests, and why any weren't answered.
    {
        const int x = tile(2, "REQUESTS \xC2\xB7 " + upper(range), u ? plain::count(u->total.requests) : "\xE2\x80\x94", kText);
        if (!u) {
            sub(x, "", kMuted);
        } else if (!u->total.requests) {
            sub(x, none, kMuted);
        } else if (!u->total.failed) {
            sub(x, "All answered", kText2);
        } else {
            const UsageSums &t = u->total;
            // How many, beside the count; under it, the most common reason.
            c.text(small_, x + Canvas::measure(h2_, plain::count(t.requests)) + 12, TY + 84,
                   plain::count(t.failed) + " unanswered", kRed);
            const std::pair<int64_t, const char *> why[] = {{t.no_model, "Mostly: no model was running"},
                                                             {t.busy, "Mostly: a GPU check was running"},
                                                             {t.error, "Mostly: the GPU hit a problem"},
                                                             {t.bad_request, "Mostly: apps sent bad requests"}};
            const auto *most = std::max_element(std::begin(why), std::end(why),
                                                [](auto &a, auto &b) { return a.first < b.first; });
            sub(x, most->second, kRed);
        }
    }
    // Writing speed over the range.
    {
        const bool any = u && u->total.gen >= kSpeedMin && u->total.gen_ms > 0;
        char v[32];
        snprintf(v, sizeof v, "%.0f", any ? speed(u->total) : 0.0);
        const int x = tile(3, "WRITING SPEED \xC2\xB7 " + upper(range), any ? v : "\xE2\x80\x94", kText);
        if (any) {
            c.text(small_, x + Canvas::measure(h2_, v) + 10, TY + 84, "tokens a second", kText2);
            if (u->total.prompt_ms > 0 && u->total.prompt_new > 0) {
                char r[64];
                snprintf(r, sizeof r, "Reads prompts at %.0f a second", u->total.prompt_new * 1000.0 / u->total.prompt_ms);
                sub(x, r, kText2);
            }
        } else if (u) {
            sub(x, u->ever ? "No replies " + in_range : "No replies yet", kMuted);
        }
    }
}

void HomePage::draw_tokens(Canvas &c, const HomeInfo &info) {
    const UsageView *u = info.usage && info.usage->ready ? info.usage.get() : nullptr;
    const char *per = range_ == UsageView::Day ? "Tokens per hour" : range_ == UsageView::Week ? "Tokens per 6 hours"
                                                                                              : "Tokens per day";
    c.text(h2_, KX + 32, KY + 52, per, kText);
    // The ranges, the chosen one underlined.
    int rx = KX + KW - 32;
    for (UsageView::Range r : {UsageView::Month, UsageView::Week, UsageView::Day}) {
        const bool on = r == range_;
        const Font &f = on ? smallb_ : small_;
        const int left = c.text_right(f, rx, KY + 48, range_words(r), on ? kText : kMuted);
        if (on) c.round_rect((float)left, KY + 58, (float)(rx - left), 3, 1.5f, kAccent);
        rx = left - 28;
    }
    const int mid_x = KX + KW / 2, mid_y = (KTOP + KBASE) / 2;
    auto centered = [&](const Font &f, int y, const std::string &s, Color col) {
        c.text(f, mid_x - Canvas::measure(f, s) / 2, y, s, col, KW - 64);
    };
    if (!u) {
        centered(body_, mid_y, "Reading the usage history\xE2\x80\xA6", kMuted);
        return;
    }
    if (!u->ever) {
        centered(bodyb_, mid_y - 16, "No requests yet.", kText);
        centered(small_, mid_y + 24,
                 info.on_network ? "Apps on your network can use " + info.url + ". The Chat page counts too."
                                 : "Apps can use TokenStation5 once the PS5 is on a network. The Chat page counts too.",
                 kText2);
        return;
    }

    // Each bucket's tokens, and the scale.
    const int n = u->buckets;
    std::vector<int64_t> totals(n, 0);
    for (const auto &m : u->models)
        for (int b = 0; b < n; b++) totals[b] += m.by_bucket[b].prompt + m.by_bucket[b].gen;
    const int64_t most = *std::max_element(totals.begin(), totals.end());
    const double step = nice_up(std::max<int64_t>(most, 3) / 3.0), top = step * std::ceil(std::max<int64_t>(most, 1) / step);
    const double ph = KBASE - KTOP;
    for (int g = 1; g * step <= top + 0.5; g++) {
        const int y = KBASE - (int)std::lround(g * step / top * ph);
        c.fill(PX0, y, PX1 - PX0, 1, kPanelLine);
        c.text_right(mono_, PX0 - 10, y + 7, compact((int64_t)(g * step)), kMuted);
    }
    c.fill(PX0, KBASE, PX1 - PX0, 1, kPanelLine);
    c.text_right(mono_, PX0 - 10, KBASE + 7, "0", kMuted);

    const float slot = (float)(PX1 - PX0) / n, bw = std::max(4.0f, std::floor(slot * 0.8f));
    int tallest = -1;
    for (int b = 0; b < n; b++) {
        if (!totals[b]) continue;
        if (tallest < 0 || totals[b] > totals[tallest]) tallest = b;
        const float x = PX0 + b * slot + (slot - bw) / 2;
        float y = KBASE;
        const float h_all = (float)(totals[b] / top * ph);
        // From the bottom up in slot order, a 2 px gap between models; the
        // top rounded.
        std::vector<std::pair<float, Color>> parts;
        for (const auto &m : u->models) {
            const int64_t t = m.by_bucket[b].prompt + m.by_bucket[b].gen;
            if (t) parts.push_back({std::max(3.0f, (float)(t / top * ph)), kSeries[m.slot]});
        }
        float sum = 0;
        for (auto &p : parts) sum += p.first;
        const float scale = sum > h_all && h_all > 3 * parts.size() ? h_all / sum : 1;
        for (size_t i = 0; i < parts.size(); i++) {
            const float h = parts[i].first * scale;
            const bool last = i + 1 == parts.size();
            if (last) {
                c.round_rect(x, y - h, bw, h, std::min(4.0f, h / 2), parts[i].second);
                if (h > 8) c.fill((int)x, (int)(y - h / 2), (int)bw, (int)std::ceil(h / 2), parts[i].second);
            } else {
                c.fill((int)std::lround(x), (int)std::lround(y - h), (int)bw, (int)std::lround(h), parts[i].second);
                c.fill((int)std::lround(x), (int)std::lround(y - h), (int)bw, 2, kPanel);
            }
            y -= h;
        }
    }
    if (tallest >= 0) {
        const float x = PX0 + tallest * slot + slot / 2;
        const int y = KBASE - (int)std::lround(totals[tallest] / top * ph) - 10;
        const std::string s = compact(totals[tallest]);
        c.text(small_, std::max(PX0 + 12, (int)x - Canvas::measure(small_, s) / 2), y, s, kText2);
    }

    // Time along the bottom: how long ago each label's bucket began.
    struct Tick {
        int bucket;
        const char *text;
    };
    std::vector<Tick> ticks;
    if (range_ == UsageView::Day) ticks = {{0, "24 h ago"}, {6, "18 h ago"}, {12, "12 h ago"}, {18, "6 h ago"}};
    else if (range_ == UsageView::Week) ticks = {{0, "7 days ago"}, {8, "5 days ago"}, {16, "3 days ago"}};
    else ticks = {{0, "30 days ago"}, {10, "20 days ago"}, {20, "10 days ago"}};
    for (const Tick &t : ticks) c.text(small_, PX0 + (int)(t.bucket * slot), KBASE + 30, t.text, kMuted);
    c.text_right(small_, PX1, KBASE + 30, "now", kMuted);

    // The models, each with its colour and tokens over the range.
    int x = PX0;
    const int y = KY + KH - 24;
    if (u->models.empty()) {
        centered(body_, mid_y, "No requests in the last " + std::string(range_words(range_)), kMuted);
        return;
    }
    for (const auto &m : u->models) {
        const std::string name = short_name(m.name), total = compact(m.total.prompt + m.total.gen);
        const int need = 20 + Canvas::measure(small_, name) + 8 + Canvas::measure(smallb_, total);
        if (x + need > KX + KW - 32) break;
        c.round_rect((float)x, (float)y - 14, 12, 12, 3, kSeries[m.slot]);
        x += 20;
        x += c.text(small_, x, y, name, kText2) + 8;
        x += c.text(smallb_, x, y, total, kText2) + 32;
    }
}

void HomePage::draw_apps(Canvas &c, const HomeInfo &info) {
    const UsageView *u = info.usage && info.usage->ready ? info.usage.get() : nullptr;
    c.text(h2_, AX + 32, AY + 52, "Requests by app", kText);
    if (!u || u->apps.empty()) {
        const std::string s = !u ? "" : u->ever ? "None in the last " + std::string(range_words(range_)) : "None yet";
        c.text(body_, AX + (AW - Canvas::measure(body_, s)) / 2, AY + AH / 2 + 10, s, kMuted);
        return;
    }
    int64_t most = 1;
    for (const auto &a : u->apps) most = std::max(most, a.second);
    int y = AY + 108;
    const int w = AW - 64;
    for (const auto &a : u->apps) {
        const int cw = c.text_right(bodyb_, AX + AW - 32, y, plain::count(a.second), kText);
        c.text(body_, AX + 32, y, a.first, kText2, cw - 16 - (AX + 32));
        c.round_rect(AX + 32, (float)y + 12, (float)w, 12, 6, kTrack);
        c.round_rect(AX + 32, (float)y + 12, std::max(12.0f, (float)w * a.second / most), 12, 6, kAccent);
        y += 56;
    }
}

void HomePage::draw_speed(Canvas &c, const HomeInfo &info) {
    const UsageView *u = info.usage && info.usage->ready ? info.usage.get() : nullptr;
    const char *const sub = "tokens a second, by model";
    const int tw = c.text(h2_, SX + 32, SY + 52, "Writing speed", kText);
    if (!u) {
        c.text(small_, SX + 32 + tw + 16, SY + 52, sub, kText2);
        return;
    }
    // The models that wrote most over the range, as many as the key at the
    // top right has room for. The words after the heading give way to the
    // key when it needs the room (all four models of the set do).
    std::vector<const UsageView::Series *> top;
    for (const auto &m : u->models)
        if (m.total.gen >= kSpeedMin && m.total.gen_ms > 0) top.push_back(&m);
    std::stable_sort(top.begin(), top.end(), [](auto *a, auto *b) { return a->total.gen > b->total.gen; });
    if (top.size() > 5) top.resize(5);
    int key_w = 0;
    for (auto *m : top) key_w += Canvas::measure(small_, short_name(m->name)) + 44;
    const int key_right = SX + SW - 32, after_heading = SX + 32 + tw + 40;
    int room = key_right - after_heading;
    if (key_w <= room - 16 - Canvas::measure(small_, sub)) {
        c.text(small_, SX + 32 + tw + 16, SY + 52, sub, kText2);
        room -= 16 + Canvas::measure(small_, sub);
    }
    int used = 0;
    size_t fit = 0;
    while (fit < top.size()) {
        const int w = Canvas::measure(small_, short_name(top[fit]->name)) + 44;
        if (used + w > room) break;
        used += w;
        fit++;
    }
    top.resize(std::max<size_t>(fit, std::min<size_t>(top.size(), 1)));
    if (top.empty()) {
        const std::string s = u->ever ? "No replies in the last " + std::string(range_words(range_)) : "No replies yet";
        c.text(body_, SX + (SW - Canvas::measure(body_, s)) / 2, (STOP + SBASE) / 2 + 10, s, kMuted);
        return;
    }
    // Which line is which, top right.
    int lx = SX + SW - 32;
    for (auto it = top.rbegin(); it != top.rend(); ++it) {
        const std::string name = short_name((*it)->name);
        lx = c.text_right(small_, lx, SY + 50, name, kText2) - 20;
        marker(c, (float)lx + 6, SY + 43, (*it)->slot, kSeries[(*it)->slot]);
        lx -= 24;
    }
    const int n = u->buckets;
    double most = 0;
    for (auto *m : top)
        for (const auto &s : m->by_bucket)
            if (s.gen >= kSpeedMin) most = std::max(most, speed(s));
    const double top_v = nice_up(std::max(most, 10.0) / 2) * 2;
    const double ph = SBASE - STOP;
    for (int g = 1; g <= 2; g++) {
        const int y = SBASE - (int)std::lround(g / 2.0 * ph);
        c.fill(PX0, y, PX1 - PX0, 1, kPanelLine);
        c.text_right(mono_, PX0 - 10, y + 7, compact((int64_t)(top_v * g / 2)), kMuted);
    }
    c.fill(PX0, SBASE, PX1 - PX0, 1, kPanelLine);
    c.text_right(mono_, PX0 - 10, SBASE + 7, "0", kMuted);
    const float slot = (float)(PX1 - PX0) / n;
    for (auto *m : top) {
        const Color col = kSeries[m->slot];
        float px = 0, py = 0;
        bool prev = false;
        for (int b = 0; b < n; b++) {
            const UsageSums &s = m->by_bucket[b];
            if (s.gen < kSpeedMin || s.gen_ms <= 0) {
                prev = false;
                continue;
            }
            const float x = PX0 + (b + 0.5f) * slot, y = SBASE - (float)(std::min(speed(s), top_v) / top_v * ph);
            if (prev) c.segment(px, py, x, y, 2, col);
            px = x, py = y, prev = true;
        }
        for (int b = 0; b < n; b++) {
            const UsageSums &s = m->by_bucket[b];
            if (s.gen < kSpeedMin || s.gen_ms <= 0) continue;
            marker(c, PX0 + (b + 0.5f) * slot, SBASE - (float)(std::min(speed(s), top_v) / top_v * ph), m->slot, col);
        }
    }
}

void HomePage::draw_live(Canvas &c, const HomeInfo &info) {
    c.text(h2_, LX + 32, LY + 52, "Live", kText);
    const float now_v = info.live.empty() ? 0 : info.live.back();
    char v[32];
    snprintf(v, sizeof v, "%.0f", now_v);
    const int left = c.text_right(small_, LX + LW - 32, LY + 52, "tokens a second", kText2);
    c.text_right(h2_, left - 12, LY + 52, v, info.api.generating ? kText : kText2);

    const int x0 = LX + 32, x1 = LX + LW - 32, top = LY + 72, base = LY + 156;
    c.fill(x0, base, x1 - x0, 1, kPanelLine);
    constexpr int kPoints = 120;
    const int n = std::min<int>((int)info.live.size(), kPoints);
    const float dx = (float)(x1 - x0) / (kPoints - 1);
    float most = 60;
    for (int i = 0; i < n; i++) most = std::max(most, info.live[info.live.size() - n + i]);
    // The newest point at the right edge.
    auto px = [&](int i) { return x1 - (n - 1 - i) * dx; };
    for (int i = 0; i < n; i++) {
        const size_t k = info.live.size() - n + i;
        if (k < info.reading.size() && info.reading[k])
            c.fill((int)px(i) - 1, base - 4, std::max(2, (int)dx), 4, kText2);
    }
    for (int i = 1; i < n; i++) {
        const float a = info.live[info.live.size() - n + i - 1], b = info.live[info.live.size() - n + i];
        if (a <= 0 && b <= 0) continue;
        c.segment(px(i - 1), base - a / most * (base - top), px(i), base - b / most * (base - top), 2, kAccent);
    }
    c.text(small_, x0, base + 26, "2 minutes ago", kMuted);
    c.text_right(small_, x1, base + 26, "now", kMuted);

    std::string line;
    Color col = kText2;
    const auto &api = info.api;
    if (api.generating && api.phase == 1) {
        line = "Reading a " + plain::count(api.prompt) + "-token prompt for " + who(api.client, info.own_ip);
        col = kAccent;
    } else if (api.generating) {
        line = "Writing a reply for " + who(api.client, info.own_ip);
        col = kAccent;
    } else if (!info.download.empty()) {
        line = info.download;
    } else if (info.status.phase == app::Status::Serving) {
        line = "Waiting for requests";
    } else {
        line = "Nothing running";
        col = kMuted;
    }
    if (api.waiting > 0) line += " \xC2\xB7 " + std::to_string(api.waiting) + " more waiting";
    c.text(small_, x0, LY + LH - 16, line, col, x1 - x0);
}

}  // namespace ui
