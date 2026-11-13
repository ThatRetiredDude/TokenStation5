// The pages ts5 serve shows a browser (web_serve.h) and the files they
// load, built into ts5 as they are: from engine/web and third_party/xterm
// (assembled with both as include folders, -Wa,-I).
#define FILE(sym, path) \
    .global sym; .global sym##_end; .balign 16; sym: .incbin path; sym##_end:
    .section .rodata
FILE(ts5_web_chat_html, "chat.html")
FILE(ts5_web_terminal_html, "terminal.html")
FILE(ts5_web_xterm_js, "xterm.js")
FILE(ts5_web_xterm_css, "xterm.css")
FILE(ts5_web_addon_fit_js, "addon-fit.js")
FILE(ts5_web_xterm_license, "LICENSE")
FILE(ts5_web_addon_fit_license, "LICENSE-addon-fit")
// The logo for a phone's home screen and the tab: ps5/sce_sys/icon0.png at
// 192x192 (Pillow, LANCZOS). A brand asset, not licensed (LICENSING.md).
FILE(ts5_web_icon_png, "icon.png")
FILE(ts5_web_keep_js, "keep.js")
// The launch images a phone shows while a home-screen icon opens (make_splash.py).
FILE(ts5_web_splash_1320x2868, "splash-1320x2868.jpg")
FILE(ts5_web_splash_1206x2622, "splash-1206x2622.jpg")
FILE(ts5_web_splash_1260x2736, "splash-1260x2736.jpg")
FILE(ts5_web_splash_1290x2796, "splash-1290x2796.jpg")
FILE(ts5_web_splash_1179x2556, "splash-1179x2556.jpg")
FILE(ts5_web_splash_1284x2778, "splash-1284x2778.jpg")
FILE(ts5_web_splash_1170x2532, "splash-1170x2532.jpg")
FILE(ts5_web_splash_1125x2436, "splash-1125x2436.jpg")
FILE(ts5_web_splash_1242x2688, "splash-1242x2688.jpg")
FILE(ts5_web_splash_828x1792, "splash-828x1792.jpg")
FILE(ts5_web_splash_750x1334, "splash-750x1334.jpg")
    .section .note.GNU-stack, "", %progbits
