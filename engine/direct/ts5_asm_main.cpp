// ts5-asm: assembles a kernel file with the direct engine's assembler.
//   ts5-asm FILE.s            prints the kernel's words and settings
//   ts5-asm --list FILE.s     one line per instruction: "LINE: b0 b1 ..." (bytes, hex)
#include "gfx10_asm.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    bool list = false;
    const char* path = nullptr;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--list") list = true;
        else path = argv[i];
    }
    if (!path) {
        std::fprintf(stderr, "usage: ts5-asm [--list] FILE.s\n");
        return 2;
    }
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "ts5-asm: can't read %s\n", path);
        return 2;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    ts5::gfx10::Kernel k;
    std::string err;
    if (!ts5::gfx10::assemble(ss.str(), k, &err)) {
        std::fprintf(stderr, "%s: %s\n", path, err.c_str());
        return 1;
    }
    if (list) {
        for (auto& a : k.at) {
            std::printf("%d:", a.line);
            for (size_t w = a.word; w < a.word + a.words; w++)
                for (int b = 0; b < 4; b++) std::printf(" %02x", (k.code[w] >> (8 * b)) & 0xff);
            std::printf("\n");
        }
        return 0;
    }
    std::printf("kernel %s: %zu words, wave%d, %d vgprs, %d sgprs, %u bytes lds, rsrc1 0x%08x rsrc2 0x%08x\n",
                k.name.empty() ? "(unnamed)" : k.name.c_str(), k.code.size(), k.wave32 ? 32 : 64, k.vgprs, k.sgprs,
                k.lds_bytes, ts5::gfx10::pgm_rsrc1(k), ts5::gfx10::pgm_rsrc2(k));
    for (size_t i = 0; i < k.code.size(); i++) std::printf("%08x%s", k.code[i], (i % 8 == 7 || i + 1 == k.code.size()) ? "\n" : " ");
    return 0;
}
