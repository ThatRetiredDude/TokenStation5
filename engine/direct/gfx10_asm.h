// The direct engine's assembler: GFX10.1 (RDNA 1, the PS5 GPU, gfx1013)
// machine code from text, with no compiler, driver or LLVM in between.
//
// The text is the usual AMDGPU assembly syntax (what llvm-mc reads), so a
// kernel can be checked byte for byte against an independent assembler
// (tests/asm_check.py does, in CI only). On top of that it reads a few
// directives of its own:
//
//   .kernel NAME      names the kernel
//   .wave32 / .wave64 the wave size the kernel is written for (default 32)
//   .lds BYTES        shared memory the kernel needs per workgroup
//   .vgprs N          at least N vector registers (default: highest used + 1)
//   .sgprs N          at least N scalar registers
//   .user_sgprs N     scalar registers loaded from the dispatch's arguments
//   .tgid_x / .tgid_y / .tgid_z   the workgroup id goes to the next free SGPR
//   .tid_y / .tid_z   thread ids in y (v1) and z (v2) as well as x (v0)
//
// Labels end in ':' and branch with s_branch / s_cbranch_*.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ts5::gfx10 {

struct Kernel {
    std::string name;
    std::vector<uint32_t> code;      // machine words, s_code_end padding not included
    bool wave32 = true;
    int vgprs = 0;                   // registers each lane uses
    int sgprs = 0;
    uint32_t lds_bytes = 0;
    int user_sgprs = 0;
    bool tgid[3] = {false, false, false};
    int tid_dims = 1;                // 1: v0 only, 2: v0 v1, 3: v0 v1 v2
    // For tests: the source line of each instruction and where its words start.
    struct At { int line; size_t word; size_t words; };
    std::vector<At> at;
};

// Assembles `text`. On failure returns false and sets *err to "line N: why".
bool assemble(const std::string& text, Kernel& out, std::string* err);

// The 32-bit words of COMPUTE_PGM_RSRC1 and RSRC2 for this kernel, as the
// command stream sets them (registers, wave size, shared memory, ids).
uint32_t pgm_rsrc1(const Kernel& k);
uint32_t pgm_rsrc2(const Kernel& k);

}  // namespace ts5::gfx10
