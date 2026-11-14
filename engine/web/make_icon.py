#!/usr/bin/env python3
"""The home-screen icon, written into the pages themselves (chat.html and
terminal.html) as a data: address: on iOS 27 the Add to Home Screen sheet
fetches a network icon whole and still draws a letter tile (seen on iOS 27.2,
2026-11-14: five fetches of the logo, 200 and every byte, then a "T"), and the
fix others report for that sheet is the icon in the page. 180x180, Apple's
size, in 256 colours (about 20 KB): the logo looks the same at that size.
icon.png stays the pages' tab icon and what /apple-touch-icon.png answers.

    python3 engine/web/make_icon.py engine/web   (needs Pillow)

The logo is a brand asset, not licensed (LICENSING.md).
"""
import base64
import io
import os
import re
import sys

from PIL import Image


def icon_png(path):
    im = Image.open(path).convert("RGBA")
    flat = Image.new("RGBA", im.size, (0, 0, 0, 255))  # iOS puts no see-through icon on the home screen
    flat.alpha_composite(im)
    small = flat.convert("RGB").resize((180, 180), Image.LANCZOS)
    small = small.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    small.save(out, format="PNG", optimize=True)
    return out.getvalue()


def main():
    web = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    png = icon_png(os.path.join(web, "icon.png"))
    link = ('<link rel="apple-touch-icon" sizes="180x180" href="data:image/png;base64,'
            + base64.b64encode(png).decode() + '">')
    for page in ("chat.html", "terminal.html"):
        p = os.path.join(web, page)
        text = open(p, encoding="utf-8").read()
        new, n = re.subn(r'<link rel="apple-touch-icon"[^>]*>', lambda _: link, text)
        if n != 1:
            sys.exit(f"{page}: {n} apple-touch-icon links, expected 1")
        open(p, "w", encoding="utf-8").write(new)
        print(f"{page}: icon {len(png)} bytes, {len(link)} in the page")


if __name__ == "__main__":
    main()
