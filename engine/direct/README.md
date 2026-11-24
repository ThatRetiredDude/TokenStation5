# The direct engine

ts5 driving the PS5's GPU (GFX10.1, gfx1013) itself: our own machine code,
our own command packets, and the kernel's amdgpu driver only for memory and
starting the chip. No Vulkan, Mesa, RADV, ACO, LLVM or SPIR-V at run time.
It is the engine for TokenStation5 OS, and the Vulkan engine stays as the
reference every kernel here has to match.

| Part | Status |
| --- | --- |
| `gfx10_asm.*`: assembler, text to machine words | 557 instruction forms, byte-identical to llvm-mc (`tests/asm_check.py`) |
| `kernels/*.s`: kernels, written in GPU assembly | `fill` (smoke test), `read_bw` (read bandwidth) |
| Command stream (PM4) and user-mode queue on /dev/kfd | next |
| Mat-vecs, attention, the rest of a token | after the read test beats Vulkan's 429 GB/s |

Why each choice:

- **Wave size per kernel.** RDNA 1 runs wave32 and wave64; each kernel says
  which (`.wave32` / `.wave64`) and we keep the one that measures faster.
- **Waits by hand.** Every `s_waitcnt` is written where the data is needed,
  so loads stay in flight as long as possible.
- **Only the barriers a step needs.** The command stream waits for the
  previous kernel only when this one reads what it wrote.
- **User-mode queue.** Work is started by writing the queue's doorbell, with
  no system call per kernel.

Build and check (llvm-mc only as a second opinion, never shipped):

    cmake -S engine/direct -B build/direct && cmake --build build/direct
    python3 engine/direct/tests/asm_check.py build/direct/ts5-asm
