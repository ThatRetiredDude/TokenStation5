// The Mozilla CA bundle (certifi's cacert.pem, fetched by engine/deps.sh), built
// into ts5 as one NUL-terminated string: the PS5 title has no CA files of its
// own to check HTTPS servers with. TS5_CA_PEM is its path, set by the build.
#ifdef TS5_HTTPS
__asm__(".section .rodata\n"
        ".balign 16\n"
        ".globl ts5_ca_pem\n"
        ".hidden ts5_ca_pem\n"
        "ts5_ca_pem:\n"
        ".incbin \"" TS5_CA_PEM "\"\n"
        ".byte 0\n"
        ".previous\n");
#endif
