// The look the TV's pages share: colours, the controller's button glyphs,
// labels and the spinner.
#pragma once
#include <cmath>
#include <string>

#include "canvas.h"

namespace ui {

constexpr Color kBgTop = rgb(0x0B1226), kBgBottom = rgb(0x05070F);
constexpr Color kPanel = rgb(0x0E1530), kPanelLine = rgb(0x1B2650);
constexpr Color kRow = rgb(0x121B3A), kRowSel = rgb(0x17244E);
constexpr Color kText = rgb(0xEAF0FF), kText2 = rgb(0x9AA6C8), kMuted = rgb(0x627095);
constexpr Color kAccent = rgb(0x3CC8FF), kGreen = rgb(0x3DDC97), kRed = rgb(0xFF5C70), kAmber = rgb(0xFFB547);
// A model's colour on the Home page's charts, by UsageStore::slot_of: Qwen
// 2.5 Coder 7B, Qwen 3 8B, Llama 3.1 8B, Llama 3.2 3B, Gemma 4, then every
// other model. Checked as a set for colour-blind viewers on kPanel; status
// green, amber and red are never a model's.
constexpr Color kSeries[6] = {rgb(0x3987E5), rgb(0x199E70), rgb(0xD95926), rgb(0x7E4AC2), rgb(0xB42F73), kMuted};
constexpr Color kTrack = rgb(0x1B2650);
constexpr Color kCrossColor = rgb(0x8FB0FF), kCircleColor = rgb(0xFF6F7F), kSquareColor = rgb(0xFF8BD0),
                kTriangleColor = rgb(0x4BE0A8);

enum PadGlyph { kGlyphCross, kGlyphCircle, kGlyphSquare, kGlyphTriangle };

inline void glyph(Canvas &c, float cx, float cy, PadGlyph g, bool enabled = true) {
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

// A shoulder button's name (L1, R1) in a rounded key at x; returns its width.
inline int key_cap(Canvas &c, const Font &f, int x, int y, const std::string &name, bool enabled = true) {
    const int w = Canvas::measure(f, name) + 24;
    c.round_rect((float)x, (float)y - 25, (float)w, 34, 10, rgb(0x1E2850));
    c.text(f, x + 12, y - 1, name, enabled ? kText2 : kMuted);
    return w;
}

// A rounded label right-aligned at `right`; returns its left edge.
inline int pill(Canvas &c, const Font &f, int right, int top, const std::string &text, Color col) {
    const int w = Canvas::measure(f, text) + 32, h = 34;
    c.round_rect((float)(right - w), (float)top, (float)w, (float)h, h * 0.5f, alpha(col, 46));
    c.text(f, right - w + 16, top + 24, text, col);
    return right - w;
}

inline void spinner(Canvas &c, float cx, float cy, double t, Color col) {
    const int phase = (int)(t * 12) % 8;
    for (int i = 0; i < 8; i++) {
        const float a = (float)(i * M_PI / 4);
        const int age = (phase - i + 8) % 8;
        c.circle(cx + 11 * std::cos(a), cy + 11 * std::sin(a), 3.2f, alpha(col, 255 - age * 28));
    }
}

// Text with a controller button drawn in it: before, the button, after.
inline void text_glyph(Canvas &c, const Font &f, int x, int y, const std::string &before, PadGlyph g,
                       const std::string &after, Color col) {
    x += c.text(f, x, y, before, col) + 6;
    glyph(c, (float)x + 16, (float)y - f.px * 0.33f, g);
    c.text(f, x + 38, y, after, col);
}

}  // namespace ui
