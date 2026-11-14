#!/usr/bin/env python3
"""The launch images an iPhone shows while a home-screen icon of the chat page
(or the browser terminal) opens: the logo on black, the name, "Connecting…".
iOS shows one only when its size is the phone's screen exactly, so there is
one per iPhone screen (portrait), named splash-WxH.jpg, and the pages list
them with a media query each (SPLASH below; chat.html and terminal.html), by
device-width and pixel ratio only: iOS 27 takes the window's height for
device-height (WebKit bug 326500), so no launch image matched there.

    python3 engine/web/make_splash.py ps5/sce_sys/logo.jpg FONT.otf engine/web   (needs Pillow)

FONT: Inter SemiBold (OFL) was used. The logo is a brand asset, not licensed
(LICENSING.md).
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

# CSS width, height, pixel ratio: every iPhone screen since the X, and the SE.
SPLASH = [
    (440, 956, 3),  # 16 Pro Max, 17 Pro Max
    (402, 874, 3),  # 16 Pro, 17, 17 Pro
    (420, 912, 3),  # Air
    (430, 932, 3),  # 14 Pro Max, 15 Plus, 15 Pro Max, 16 Plus
    (393, 852, 3),  # 14 Pro, 15, 15 Pro, 16
    (428, 926, 3),  # 12 Pro Max, 13 Pro Max, 14 Plus
    (390, 844, 3),  # 12, 12 Pro, 13, 13 Pro, 14, 16e
    (375, 812, 3),  # X, XS, 11 Pro, 12 mini, 13 mini
    (414, 896, 3),  # XS Max, 11 Pro Max
    (414, 896, 2),  # XR, 11
    (375, 667, 2),  # SE (2nd and 3rd), 8
]


def main():
    logo_path, font_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    # Its near-black background (about 1,5,9 in the JPEG) to pure black, so
    # no box shows around it.
    logo = Image.open(logo_path).convert("RGB").point(lambda v: max(0, v - 14) * 255 // 241)
    for w, h, r in SPLASH:
        W, H = w * r, h * r
        im = Image.new("RGB", (W, H), (0, 0, 0))
        lw = int(w * 0.62 * r)
        lh = int(logo.height * lw / logo.width)
        top = int(H * 0.40 - lh / 2)
        im.paste(logo.resize((lw, lh), Image.LANCZOS), ((W - lw) // 2, top))
        d = ImageDraw.Draw(im)
        name = ImageFont.truetype(font_path, 26 * r)
        small = ImageFont.truetype(font_path, 15 * r)
        y = top + lh + 18 * r
        for text, font, color in (("TokenStation5", name, (126, 214, 255)), ("Connecting…", small, (139, 148, 158))):
            box = d.textbbox((0, 0), text, font=font)
            d.text(((W - (box[2] - box[0])) // 2 - box[0], y), text, font=font, fill=color)
            y += (box[3] - box[1]) + 16 * r
        path = os.path.join(out, "splash-%dx%d.jpg" % (W, H))
        im.save(path, quality=80, optimize=True)
        print(path, os.path.getsize(path))


if __name__ == "__main__":
    main()
