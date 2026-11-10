// ts5-tui's terminal layer: raw mode, keys, colors and a screen of cells
// that is written out as only the rows that changed. Plain ANSI escapes; no
// curses, no library.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tui {

// 0xRRGGBB, or kDefault for the terminal's own color.
constexpr uint32_t kDefault = 0xFF000000u;

enum Attr : uint8_t { BOLD = 1, DIM = 2, ITALIC = 4, UNDER = 8, REVERSE = 16 };

struct Style {
    uint32_t fg = kDefault, bg = kDefault;
    uint8_t attr = 0;
    Style() = default;
    Style(uint32_t f, uint32_t b = kDefault, uint8_t a = 0) : fg(f), bg(b), attr(a) {}
    Style with_bg(uint32_t b) const { Style s = *this; s.bg = b; return s; }
    Style with(uint8_t a) const { Style s = *this; s.attr |= a; return s; }
    bool operator==(const Style &o) const { return fg == o.fg && bg == o.bg && attr == o.attr; }
    bool operator!=(const Style &o) const { return !(*this == o); }
};

// Colors between a and b, t from 0 to 1.
uint32_t mix(uint32_t a, uint32_t b, double t);
// Along a list of stops, t from 0 to 1.
uint32_t gradient(const std::vector<uint32_t> &stops, double t);

// The width a UTF-8 string takes on screen (wide characters count 2).
int text_width(const std::string &s);
// Splits UTF-8 into characters (each with its width).
struct Glyph {
    std::string s;
    int w;
};
std::vector<Glyph> glyphs(const std::string &s);
// The string cut to at most w columns, with "…" when cut and ell is set.
std::string fit(const std::string &s, int w, bool ell = true);

struct Cell {
    std::string g = " ";  // "" for the right half of a wide character
    Style st;
};

class Screen {
public:
    int w = 80, h = 24;
    void resize(int nw, int nh);
    void clear(Style st);
    // Writes text at (x, y), clipped to the screen and to max_w columns
    // (-1: the screen's edge). Returns the column after the text.
    int put(int x, int y, const std::string &text, Style st, int max_w = -1);
    void fill(int x, int y, int w, int h, Style st, const std::string &g = " ");
    Cell *at(int x, int y) { return x < 0 || y < 0 || x >= w || y >= h ? nullptr : &cells_[(size_t)y * w + x]; }
    // Everything that changed since the last flush, as escapes.
    std::string flush(bool truecolor, bool full);

private:
    std::vector<Cell> cells_;
    std::vector<std::string> prev_rows_;
};

enum Key : int {
    K_NONE = 0,
    K_CHAR,  // Event::ch holds the character (UTF-8)
    K_ENTER,
    K_NEWLINE,  // Ctrl+J or Alt+Enter: a line break in the message
    K_TAB,
    K_BACKTAB,
    K_BACKSPACE,
    K_DELETE,
    K_ESC,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_HOME,
    K_END,
    K_PGUP,
    K_PGDN,
    K_F1,
    K_F2,
    K_F3,
    K_F4,
    K_F5,
    K_F6,
    K_CTRL,   // Event::ch holds the letter ('a'..'z')
    K_ALT,    // Event::ch holds the character
    K_PASTE,  // Event::ch holds the pasted text
    K_CTRL_UP,
    K_CTRL_DOWN,
    K_CTRL_LEFT,
    K_CTRL_RIGHT,
};

struct Event {
    Key key = K_NONE;
    std::string ch;
};

// Raw mode, the alternate screen, bracketed paste. leave() puts it all back
// (also called at exit and on a fatal signal).
bool term_enter();
void term_leave();
bool term_size(int &w, int &h);
bool term_truecolor();
// Waits up to timeout_ms for input and decodes what came.
std::vector<Event> read_events(int timeout_ms);
void term_write(const std::string &s);

}  // namespace tui
