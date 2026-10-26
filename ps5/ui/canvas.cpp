#include "canvas.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ui {

namespace {

template <class T>
T rd(const uint8_t *&p) {
    T v;
    memcpy(&v, p, sizeof v);
    p += sizeof v;
    return v;
}

// The next code point of UTF-8 text; malformed bytes come out one by one.
uint32_t next_cp(const std::string &s, size_t &i) {
    const unsigned char c = (unsigned char)s[i++];
    int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    uint32_t cp = extra == 3 ? c & 7u : extra == 2 ? c & 15u : extra == 1 ? c & 31u : c;
    for (; extra > 0 && i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80; extra--)
        cp = (cp << 6) | ((unsigned char)s[i++] & 0x3Fu);
    return cp;
}

}  // namespace

const Glyph *Font::find(uint32_t cp) const {
    auto it = std::lower_bound(glyphs.begin(), glyphs.end(), cp, [](const Glyph &g, uint32_t v) { return g.cp < v; });
    if (it != glyphs.end() && it->cp == cp) return &*it;
    return cp == '?' ? nullptr : find('?');
}

bool Fonts::load(const uint8_t *data, size_t size) {
    if (size < 8 || memcmp(data, "TS5F", 4) != 0) return false;
    const uint8_t *p = data + 4, *end = data + size;
    const uint32_t count = rd<uint32_t>(p);
    for (uint32_t f = 0; f < count; f++) {
        if (end - p < 32) return false;
        Font font;
        char name[17] = {};
        memcpy(name, p, 16);
        p += 16;
        font.name = name;
        font.px = rd<uint16_t>(p);
        font.ascent = rd<int16_t>(p);
        font.descent = rd<int16_t>(p);
        font.line_height = rd<uint16_t>(p);
        const uint32_t n = rd<uint32_t>(p), bytes = rd<uint32_t>(p);
        if ((size_t)(end - p) < (size_t)n * 20 + bytes) return false;
        const uint8_t *bits = p + (size_t)n * 20;
        for (uint32_t i = 0; i < n; i++) {
            Glyph g;
            g.cp = rd<uint32_t>(p);
            g.left = rd<int16_t>(p);
            g.top = rd<int16_t>(p);
            g.w = rd<uint16_t>(p);
            g.h = rd<uint16_t>(p);
            g.adv64 = rd<uint16_t>(p);
            rd<uint16_t>(p);
            g.bits = bits + rd<uint32_t>(p);
            font.glyphs.push_back(g);
        }
        p = bits + bytes;
        fonts_.push_back(std::move(font));
    }
    return !fonts_.empty();
}

const Font &Fonts::get(const char *name) const {
    for (auto &f : fonts_)
        if (f.name == name) return f;
    return fonts_.front();
}

void Canvas::clip(int x, int y, int w, int h) {
    cx0_ = std::max(0, x);
    cy0_ = std::max(0, y);
    cx1_ = std::min(w_, x + w);
    cy1_ = std::min(h_, y + h);
}

void Canvas::unclip() { clip(0, 0, w_, h_); }

void Canvas::clear(Color c) { std::fill(px_.begin(), px_.end(), c | 0xFF000000u); }

inline void Canvas::blend(int x, int y, Color c, unsigned cov) {
    if (x < cx0_ || y < cy0_ || x >= cx1_ || y >= cy1_) return;
    const unsigned a = ((c >> 24) * cov + 127) / 255;
    if (a == 0) return;
    uint32_t &d = px_[(size_t)y * w_ + x];
    if (a >= 255) {
        d = c | 0xFF000000u;
        return;
    }
    const unsigned ia = 255 - a;
    const unsigned r = (((c >> 16) & 255) * a + ((d >> 16) & 255) * ia + 127) / 255;
    const unsigned g = (((c >> 8) & 255) * a + ((d >> 8) & 255) * ia + 127) / 255;
    const unsigned b = ((c & 255) * a + (d & 255) * ia + 127) / 255;
    d = 0xFF000000u | (r << 16) | (g << 8) | b;
}

void Canvas::fill(int x, int y, int w, int h, Color c) {
    const int x0 = std::max(x, cx0_), y0 = std::max(y, cy0_);
    const int x1 = std::min(x + w, cx1_), y1 = std::min(y + h, cy1_);
    for (int yy = y0; yy < y1; yy++)
        for (int xx = x0; xx < x1; xx++) blend(xx, yy, c, 255);
}

void Canvas::gradient(int x, int y, int w, int h, Color top, Color bottom) {
    for (int i = 0; i < h; i++) {
        const float t = h > 1 ? (float)i / (h - 1) : 0.0f;
        auto mix = [&](int sh) {
            const float a = (float)((top >> sh) & 255), b = (float)((bottom >> sh) & 255);
            return (uint32_t)std::lround(a + (b - a) * t) << sh;
        };
        fill(x, y + i, w, 1, mix(24) | mix(16) | mix(8) | mix(0));
    }
}

// Coverage from a signed distance (negative inside), one pixel of
// antialiasing at the edge.
template <class Dist>
void Canvas::shape(float x0, float y0, float x1, float y1, Color c, Dist dist) {
    const int ix0 = std::max(cx0_, (int)std::floor(x0) - 1), iy0 = std::max(cy0_, (int)std::floor(y0) - 1);
    const int ix1 = std::min(cx1_, (int)std::ceil(x1) + 1), iy1 = std::min(cy1_, (int)std::ceil(y1) + 1);
    for (int y = iy0; y < iy1; y++)
        for (int x = ix0; x < ix1; x++) {
            const float d = dist(x + 0.5f, y + 0.5f);
            const float cov = std::min(1.0f, std::max(0.0f, 0.5f - d));
            if (cov > 0) blend(x, y, c, (unsigned)(cov * 255.0f + 0.5f));
        }
}

namespace {

float round_rect_dist(float px, float py, float x, float y, float w, float h, float r) {
    r = std::min(r, std::min(w, h) * 0.5f);
    const float qx = std::fabs(px - (x + w * 0.5f)) - (w * 0.5f - r);
    const float qy = std::fabs(py - (y + h * 0.5f)) - (h * 0.5f - r);
    const float ox = std::max(qx, 0.0f), oy = std::max(qy, 0.0f);
    return std::sqrt(ox * ox + oy * oy) + std::min(std::max(qx, qy), 0.0f) - r;
}

}  // namespace

void Canvas::round_rect(float x, float y, float w, float h, float r, Color c) {
    shape(x, y, x + w, y + h, c, [&](float px, float py) { return round_rect_dist(px, py, x, y, w, h, r); });
}

void Canvas::round_rect_line(float x, float y, float w, float h, float r, float t, Color c) {
    shape(x, y, x + w, y + h, c, [&](float px, float py) {
        // The band of width t just inside the edge.
        return std::fabs(round_rect_dist(px, py, x, y, w, h, r) + t * 0.5f) - t * 0.5f;
    });
}

void Canvas::circle(float cx, float cy, float r, Color c) {
    shape(cx - r, cy - r, cx + r, cy + r, c, [&](float px, float py) { return std::hypot(px - cx, py - cy) - r; });
}

void Canvas::ring(float cx, float cy, float r, float t, Color c) {
    shape(cx - r - t, cy - r - t, cx + r + t, cy + r + t, c,
          [&](float px, float py) { return std::fabs(std::hypot(px - cx, py - cy) - r) - t * 0.5f; });
}

void Canvas::segment(float x0, float y0, float x1, float y1, float t, Color c) {
    const float dx = x1 - x0, dy = y1 - y0, len2 = std::max(1e-6f, dx * dx + dy * dy);
    shape(std::min(x0, x1) - t, std::min(y0, y1) - t, std::max(x0, x1) + t, std::max(y0, y1) + t, c,
          [&](float px, float py) {
              const float k = std::min(1.0f, std::max(0.0f, ((px - x0) * dx + (py - y0) * dy) / len2));
              return std::hypot(px - (x0 + k * dx), py - (y0 + k * dy)) - t * 0.5f;
          });
}

void Canvas::image(int x, int y, int w, int h, const uint32_t *pixels) {
    for (int yy = std::max(y, cy0_); yy < std::min(y + h, cy1_); yy++)
        for (int xx = std::max(x, cx0_); xx < std::min(x + w, cx1_); xx++)
            px_[(size_t)yy * w_ + xx] = pixels[(size_t)(yy - y) * w + (xx - x)] | 0xFF000000u;
}

int Canvas::measure(const Font &f, const std::string &s) {
    int adv = 0;
    for (size_t i = 0; i < s.size();)
        if (const Glyph *g = f.find(next_cp(s, i))) adv += g->adv64;
    return (adv + 32) / 64;
}

std::string Canvas::fit(const Font &f, const std::string &s, int max_w) {
    if (measure(f, s) <= max_w) return s;
    std::string cut = s;
    while (!cut.empty()) {
        // Drop the last code point (and any trailing space).
        size_t i = cut.size() - 1;
        while (i > 0 && ((unsigned char)cut[i] & 0xC0) == 0x80) i--;
        cut.resize(i);
        while (!cut.empty() && cut.back() == ' ') cut.pop_back();
        if (measure(f, cut + "\xE2\x80\xA6") <= max_w) break;
    }
    return cut + "\xE2\x80\xA6";
}

std::vector<std::string> *Canvas::record = nullptr;
bool Canvas::conversation = false;

int Canvas::text(const Font &f, int x, int y, const std::string &s, Color c, int max_w) {
    const std::string t = max_w > 0 ? fit(f, s, max_w) : s;
    if (record && !t.empty()) record->push_back(conversation ? "conversation\t" + t : t);
    int pen = x * 64;
    for (size_t i = 0; i < t.size();) {
        const Glyph *g = f.find(next_cp(t, i));
        if (!g) continue;
        const int gx = (pen + 32) / 64 + g->left, gy = y + g->top;
        for (int yy = 0; yy < g->h; yy++)
            for (int xx = 0; xx < g->w; xx++) {
                const unsigned cov = g->bits[(size_t)yy * g->w + xx];
                if (cov) blend(gx + xx, gy + yy, c, cov);
            }
        pen += g->adv64;
    }
    return (pen + 32) / 64 - x;
}

int Canvas::text_right(const Font &f, int x, int y, const std::string &s, Color c) {
    const int left = x - measure(f, s);
    text(f, left, y, s, c);
    return left;
}

bool Canvas::save_bmp(const std::string &path) const {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t data = (uint32_t)w_ * h_ * 4;
    uint8_t h[54] = {'B', 'M'};
    auto put32 = [&](int at, uint32_t v) { memcpy(h + at, &v, 4); };
    put32(2, 54 + data);
    put32(10, 54);
    put32(14, 40);
    put32(18, (uint32_t)w_);
    put32(22, (uint32_t)-h_);  // top-down rows
    h[26] = 1;
    h[28] = 32;
    put32(34, data);
    bool ok = fwrite(h, 1, sizeof h, f) == sizeof h && fwrite(px_.data(), 1, data, f) == data;
    return fclose(f) == 0 && ok;
}

}  // namespace ui
