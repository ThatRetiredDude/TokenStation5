#!/usr/bin/env python3
# The TokenStation5 logo for the dashboard's header: the title icon
# (ps5/sce_sys/icon0.png) scaled down, as raw pixels the title can draw
# without a PNG decoder (logo.bin, embedded by font_data.S):
#
#   python3 ps5/ui/make_logo.py ps5/sce_sys/icon0.png ps5/ui/logo.bin   (needs Pillow)
#
# Format, little-endian: "TS5L", u16 width, u16 height, then 0xAARRGGBB
# pixels row by row.
import struct, sys
from PIL import Image

SIZE = 104
img = Image.open(sys.argv[1]).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
out = bytearray(b"TS5L" + struct.pack("<HH", SIZE, SIZE))
rgb = img.tobytes()
for i in range(0, len(rgb), 3):
    out += bytes((rgb[i + 2], rgb[i + 1], rgb[i], 255))
with open(sys.argv[2], "wb") as f:
    f.write(out)
