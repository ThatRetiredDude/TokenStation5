#!/usr/bin/env python3
# Bakes the dashboard's fonts into one file of antialiased glyph bitmaps
# (font.bin, embedded in the title by font_data.S), so the title needs no
# font engine:
#
#   python3 ps5/ui/make_font.py ps5/ui/font.bin   (needs Pillow, Inter and DejaVu Sans Mono)
#
# Format, little-endian: "TS5F", u32 font count, then for each font:
#   char name[16], u16 pixel size, i16 ascent, i16 descent, u16 line height,
#   u32 glyph count, u32 bitmap bytes,
#   glyphs: u32 code point, i16 left, i16 top (from the baseline, up is
#           negative), u16 width, u16 height, u16 advance (1/64 px), u16 0,
#           u32 bitmap offset,
#   bitmaps: 8-bit coverage, row by row.
import struct, sys
from PIL import Image, ImageDraw, ImageFont

INTER = "/usr/share/fonts/opentype/inter/"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONTS = [
    ("title", INTER + "InterDisplay-SemiBold.otf", 52),
    ("h2", INTER + "Inter-SemiBold.otf", 34),
    ("body", INTER + "Inter-Regular.otf", 28),
    ("bodyb", INTER + "Inter-SemiBold.otf", 28),
    ("small", INTER + "Inter-Regular.otf", 22),
    ("smallb", INTER + "Inter-SemiBold.otf", 22),
    ("mono", MONO, 20),
    ("code", MONO, 24),
]
CODEPOINTS = list(range(32, 127)) + [0xB7, 0xD7, 0x2013, 0x2014, 0x2022, 0x2026, 0x2190, 0x2191, 0x2192,
                                     0x2193, 0x2713, 0x2717, 0xB0]
# For the chat, where any text can come: Latin-1, typographic quotes and
# dashes, a few signs, and (in the mono fonts) box drawing for code. The
# title font shows only the app's name.
EXTRA = sorted(set(range(0xA0, 0x100)) - set(CODEPOINTS)) + [
    0x2018, 0x2019, 0x201A, 0x201C, 0x201D, 0x201E, 0x2020, 0x2021, 0x2030, 0x2032, 0x2033, 0x2039, 0x203A,
    0x20AC, 0x2122, 0x2212, 0x2248, 0x2260, 0x2264, 0x2265, 0x221E, 0x2194, 0x21D2, 0x2500, 0x2502, 0x250C,
    0x2510, 0x2514, 0x2518, 0x251C, 0x2524, 0x252C, 0x2534, 0x253C, 0x2550, 0x2551, 0x25CF, 0x25CB, 0x25A0]


def glyph(font, ch):
    x0, y0, x1, y1 = font.getbbox(ch, anchor="ls")
    w, h = max(0, x1 - x0), max(0, y1 - y0)
    img = Image.new("L", (max(1, w), max(1, h)), 0)
    if w and h:
        ImageDraw.Draw(img).text((-x0, -y0), ch, font=font, fill=255, anchor="ls")
    return x0, y0, w, h, round(font.getlength(ch) * 64), img.tobytes() if w and h else b""


def main(out):
    blob = bytearray(b"TS5F" + struct.pack("<I", len(FONTS)))
    for name, path, px in FONTS:
        font = ImageFont.truetype(path, px)
        ascent, descent = font.getmetrics()
        missing = glyph(font, "")  # a private-use code point: the font's .notdef
        entries, bitmaps = [], bytearray()
        # Sorted: the title looks code points up by binary search.
        for cp in sorted(set(CODEPOINTS + (EXTRA if name != "title" else []))):
            g = glyph(font, chr(cp))
            if cp > 126 and g[2:] == missing[2:]:
                continue
            x, y, w, h, adv, bits = g
            entries.append(struct.pack("<IhhHHHHI", cp, x, y, w, h, adv, 0, len(bitmaps)))
            bitmaps += bits
        blob += struct.pack("<16sHhhHII", name.encode(), px, ascent, -descent, ascent + descent, len(entries),
                            len(bitmaps))
        blob += b"".join(entries) + bitmaps
        print(f"{name}: {path.rsplit('/', 1)[1]} {px}px, {len(entries)} glyphs, {len(bitmaps)} bytes")
    with open(out, "wb") as f:
        f.write(blob)


main(sys.argv[1])
