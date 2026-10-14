#!/usr/bin/env python3
# Draws the title icon (512x512 PNG): a chip with "TS5" on a deep-blue glow.
#   python3 ps5/sce_sys/make_icon.py ps5/sce_sys/icon0.png   (needs Pillow and the Inter font)
import math, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 512
out = sys.argv[1]
img = Image.new("RGB", (S, S))
px = img.load()
for y in range(S):
    for x in range(S):
        d = math.hypot(x - S * 0.5, y - S * 0.42) / (S * 0.75)
        t = max(0.0, min(1.0, d))
        c0, c1 = (28, 52, 140), (4, 6, 18)
        px[x, y] = tuple(int(c0[i] * (1 - t) + c1[i] * t) for i in range(3))

glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
g = ImageDraw.Draw(glow)
x0, y0, x1, y1 = 116, 96, 396, 376
g.rounded_rectangle((x0 - 10, y0 - 10, x1 + 10, y1 + 10), 46, outline=(60, 200, 255, 200), width=14)
glow = glow.filter(ImageFilter.GaussianBlur(18))
img.paste(glow, (0, 0), glow)

d = ImageDraw.Draw(img)
pin = (120, 210, 255)
for i in range(6):
    o = y0 + 38 + i * 41
    d.rounded_rectangle((x0 - 40, o - 6, x0 - 6, o + 6), 4, fill=pin)
    d.rounded_rectangle((x1 + 6, o - 6, x1 + 40, o + 6), 4, fill=pin)
    o = x0 + 38 + i * 41
    d.rounded_rectangle((o - 6, y0 - 40, o + 6, y0 - 6), 4, fill=pin)
    d.rounded_rectangle((o - 6, y1 + 6, o + 6, y1 + 40), 4, fill=pin)
d.rounded_rectangle((x0, y0, x1, y1), 36, fill=(10, 16, 40), outline=(90, 210, 255), width=6)

big = ImageFont.truetype("/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf", 128)
small = ImageFont.truetype("/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf", 40)
d.text((S / 2, 222), "TS5", font=big, fill=(255, 255, 255), anchor="mm")
d.text((S / 2, 318), "TOKENS", font=small, fill=(120, 210, 255), anchor="mm")
d.text((S / 2, 470), "TokenStation5", font=ImageFont.truetype("/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf", 34),
       fill=(220, 230, 255), anchor="mm")
img.save(out, optimize=True)
