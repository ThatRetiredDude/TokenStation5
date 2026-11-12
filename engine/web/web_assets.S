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
    .section .note.GNU-stack, "", %progbits
