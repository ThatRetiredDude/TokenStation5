#!/usr/bin/env python3
# Packages the TokenStation5 logo as the title icon: the logo centred on black
# to make it square, then scaled to the 512x512 PNG the console shows.
#   python3 ps5/sce_sys/make_icon.py ps5/sce_sys/logo.jpg ps5/sce_sys/icon0.png   (needs Pillow)
import sys
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
logo = Image.open(src).convert("RGB")
side = max(logo.size)
square = Image.new("RGB", (side, side), (0, 0, 0))
square.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2))
square.resize((512, 512), Image.LANCZOS).save(out, optimize=True)
