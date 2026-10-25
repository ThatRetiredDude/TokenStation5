# QR Code generator, vendored

[QR Code generator library](https://github.com/nayuki/QR-Code-generator) (C++),
commit `3c6d0b3cefb4e049dc337e82237c9644399716a8`. MIT, "Copyright (c) Project Nayuki"; the license is at the
top of each file. NOTICE.md at the top of this repo lists it.

The title draws the WireGuard join page's address as a QR code on the TV
(`ps5/ui/settings_page.cpp`), so a phone's camera can open the page.
Unchanged; `scripts/vendor-qrcodegen.sh` copies it.

Kept: `cpp/qrcodegen.hpp` and `cpp/qrcodegen.cpp`. Left out: the other
languages, the demo.
