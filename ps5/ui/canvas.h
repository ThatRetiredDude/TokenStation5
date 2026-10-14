// The dashboard's drawing: a CPU framebuffer with antialiased shapes and
// text from the fonts make_font.py bakes. The title copies the finished frame
// to the TV (display.h); a frame is mostly flat panels and text, so a few
// milliseconds of CPU draw it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ui {

// 0xAARRGGBB, the swapchain's B8G8R8A8 byte order.
using Color = uint32_t;
constexpr Color rgb(uint32_t v) { return 0xFF000000u | v; }
constexpr Color alpha(Color c, unsigned a) { return (c & 0x00FFFFFFu) | (a << 24); }

struct Glyph {
    uint32_t cp;
    int16_t left, top;  // bitmap offset from the pen at the baseline (top: up is negative)
    uint16_t w, h, adv64;
    const uint8_t *bits;
};

struct Font {
    std::string name;
    int px = 0, ascent = 0, descent = 0, line_height = 0;
    std::vector<Glyph> glyphs;  // by code point
    const Glyph *find(uint32_t cp) const;
};

// The fonts in a font.bin, by name ("title", "h2", "body", "bodyb", "small",
// "smallb", "mono"); the data must outlive them.
class Fonts {
public:
    bool load(const uint8_t *data, size_t size);
    const Font &get(const char *name) const;

private:
    std::vector<Font> fonts_;
};

class Canvas {
public:
    Canvas(int w, int h) : w_(w), h_(h), px_((size_t)w * h) { unclip(); }
    int width() const { return w_; }
    int height() const { return h_; }
    const uint32_t *pixels() const { return px_.data(); }
    uint32_t *pixels() { return px_.data(); }

    // Drawing stays inside this rectangle until unclip().
    void clip(int x, int y, int w, int h);
    void unclip();

    void clear(Color c);
    void fill(int x, int y, int w, int h, Color c);
    // A vertical gradient from top to bottom colour.
    void gradient(int x, int y, int w, int h, Color top, Color bottom);
    void round_rect(float x, float y, float w, float h, float r, Color c);
    void round_rect_line(float x, float y, float w, float h, float r, float t, Color c);
    void circle(float cx, float cy, float r, Color c);
    void ring(float cx, float cy, float r, float t, Color c);
    void segment(float x0, float y0, float x1, float y1, float t, Color c);
    // Opaque w x h pixels, row by row.
    void image(int x, int y, int w, int h, const uint32_t *pixels);

    // UTF-8 text with its baseline at y; returns its width. With max_w > 0,
    // text wider than that ends in "…" to fit.
    int text(const Font &f, int x, int y, const std::string &s, Color c, int max_w = 0);
    // Right-aligned at x; returns the left edge.
    int text_right(const Font &f, int x, int y, const std::string &s, Color c);
    static int measure(const Font &f, const std::string &s);
    // The longest start of s, plus "…", that fits in max_w.
    static std::string fit(const Font &f, const std::string &s, int max_w);

    bool save_bmp(const std::string &path) const;

private:
    void blend(int x, int y, Color c, unsigned cov);
    template <class Dist>
    void shape(float x0, float y0, float x1, float y1, Color c, Dist dist);

    int w_, h_;
    int cx0_ = 0, cy0_ = 0, cx1_ = 0, cy1_ = 0;
    std::vector<uint32_t> px_;
};

}  // namespace ui
