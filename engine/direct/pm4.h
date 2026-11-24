// PM4 command packets for the PS5 GPU's compute queues (GFX10.1). The
// direct engine writes these straight into a queue's ring: set the kernel's
// registers, dispatch, wait only where a kernel needs the one before, and
// write a fence when the work is done.
#pragma once
#include <cstdint>
#include <vector>

#include "gfx10_asm.h"

namespace ts5::pm4 {

// Packet opcodes.
enum : uint32_t {
    NOP = 0x10,
    DISPATCH_DIRECT = 0x15,
    WRITE_DATA = 0x37,
    WAIT_REG_MEM = 0x3c,
    INDIRECT_BUFFER = 0x3f,
    EVENT_WRITE = 0x46,
    RELEASE_MEM = 0x49,
    ACQUIRE_MEM = 0x58,
    SET_SH_REG = 0x76,
};

// Shader registers (dword addresses).
enum : uint32_t {
    SH_REG_BASE = 0x2c00,
    COMPUTE_DISPATCH_INITIATOR = 0x2e00,
    COMPUTE_START_X = 0x2e04,          // then Y, Z
    COMPUTE_NUM_THREAD_X = 0x2e07,     // then Y, Z
    COMPUTE_PGM_LO = 0x2e0c,           // then HI
    COMPUTE_PGM_RSRC1 = 0x2e12,        // then RSRC2
    COMPUTE_RESOURCE_LIMITS = 0x2e15,
    COMPUTE_STATIC_THREAD_MGMT_SE0 = 0x2e16,   // then SE1
    COMPUTE_TMPRING_SIZE = 0x2e18,
    COMPUTE_STATIC_THREAD_MGMT_SE2 = 0x2e19,   // then SE3
    COMPUTE_RESTART_X = 0x2e1b,        // then Y, Z
    COMPUTE_PGM_RSRC3 = 0x2e28,
    COMPUTE_USER_DATA_0 = 0x2e40,      // 16 of them
};

// Events.
enum : uint32_t {
    CS_PARTIAL_FLUSH = 7,              // event index 4
    CACHE_FLUSH_AND_INV_TS_EVENT = 20, // event index 5 (end of pipe) for RELEASE_MEM
};

inline uint32_t header(uint32_t op, uint32_t body_dwords) {
    // type 3, count = body - 1, compute shader type (bit 1)
    return 3u << 30 | ((body_dwords - 1) & 0x3fff) << 16 | (op & 0xff) << 8 | 1u << 1;
}

struct Stream {
    std::vector<uint32_t> w;

    void packet(uint32_t op, std::initializer_list<uint32_t> body) {
        w.push_back(header(op, (uint32_t)body.size()));
        w.insert(w.end(), body.begin(), body.end());
    }

    void set_sh(uint32_t reg, const uint32_t* vals, uint32_t n) {
        w.push_back(header(SET_SH_REG, n + 1));
        w.push_back(reg - SH_REG_BASE);
        w.insert(w.end(), vals, vals + n);
    }
    void set_sh(uint32_t reg, std::initializer_list<uint32_t> vals) { set_sh(reg, vals.begin(), (uint32_t)vals.size()); }

    // Invalidates the GPU caches (instruction, scalar, vector, L1, L2) so
    // the next kernel sees what the CPU or an earlier kernel wrote, and
    // writes L2 back.
    void acquire_all() {
        uint32_t gcr = 1u << 0 /* GLI_INV */ | 1u << 4 /* GLM_WB */ | 1u << 5 /* GLM_INV */ | 1u << 6 /* GLK_WB */ |
                       1u << 7 /* GLK_INV */ | 1u << 8 /* GLV_INV */ | 1u << 9 /* GL1_INV */ | 1u << 14 /* GL2_INV */ |
                       1u << 15 /* GL2_WB */;
        packet(ACQUIRE_MEM, {0, 0xffffffff, 0xffffffff, 0, 0, 0x0000000a, gcr});
    }

    // Waits for every kernel before it to finish (no cache flush): the one
    // barrier a kernel needs when it reads what the previous one wrote.
    void wait_kernels() { packet(EVENT_WRITE, {CS_PARTIAL_FLUSH | 4u << 8}); }

    // Sets the registers for `k` and dispatches groups[0..2] workgroups of
    // threads[0..2] threads, with `args` in the kernel's user SGPRs.
    void dispatch(const gfx10::Kernel& k, uint64_t code_addr, const uint32_t* args, uint32_t nargs, const uint32_t groups[3],
                  const uint32_t threads[3]) {
        set_sh(COMPUTE_PGM_LO, {(uint32_t)(code_addr >> 8), (uint32_t)(code_addr >> 40)});
        set_sh(COMPUTE_PGM_RSRC1, {gfx10::pgm_rsrc1(k), gfx10::pgm_rsrc2(k)});
        set_sh(COMPUTE_PGM_RSRC3, {0});
        set_sh(COMPUTE_TMPRING_SIZE, {0});
        set_sh(COMPUTE_RESOURCE_LIMITS, {0});
        set_sh(COMPUTE_STATIC_THREAD_MGMT_SE0, {0xffffffff, 0xffffffff});
        set_sh(COMPUTE_STATIC_THREAD_MGMT_SE2, {0xffffffff, 0xffffffff});
        set_sh(COMPUTE_RESTART_X, {0, 0, 0});
        if (nargs) set_sh(COMPUTE_USER_DATA_0, args, nargs);
        set_sh(COMPUTE_START_X, {0, 0, 0, threads[0], threads[1], threads[2]});
        uint32_t init = 1u << 0 /* COMPUTE_SHADER_EN */ | 1u << 2 /* FORCE_START_AT_000 */ | 1u << 4 /* ORDER_MODE */;
        if (k.wave32) init |= 1u << 15;   // CS_W32_EN
        packet(DISPATCH_DIRECT, {groups[0], groups[1], groups[2], init});
    }

    // After everything before it is done and written back: writes `value`
    // (64-bit) to `addr`, or the GPU's 64-bit clock when timestamp is set.
    void release(uint64_t addr, uint64_t value, bool timestamp = false) {
        uint32_t event = CACHE_FLUSH_AND_INV_TS_EVENT | 5u << 8 | 1u << 12 /* GLM_WB */ | 1u << 13 /* GLM_INV */ |
                         1u << 14 /* GLV_INV */ | 1u << 15 /* GL1_INV */ | 1u << 20 /* GL2_INV */ | 1u << 21 /* GL2_WB */ |
                         1u << 22 /* SEQ */;
        uint32_t data_sel = timestamp ? 3 : 2;   // 3: GPU clock, 2: 64-bit value
        uint32_t sel = data_sel << 29 | 0u << 24 /* no interrupt */ | 0u << 16 /* to memory */;
        packet(RELEASE_MEM, {event, sel, (uint32_t)addr, (uint32_t)(addr >> 32), (uint32_t)value, (uint32_t)(value >> 32), 0});
    }
};

}  // namespace ts5::pm4
