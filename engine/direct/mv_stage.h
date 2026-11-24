// What a mat-vec does to x while it stages it into LDS, so the small kernels
// between a token's mat-vecs (and the waits around them) go away:
//   Stage::Norm: x = rmsnorm(x + t) * w, and workgroup 0 writes x + t to
//     xout (the residual stream, ping-ponged between two buffers so no
//     workgroup reads what another writes);
//   Stage::Silu: x = silu(x) * u (x: the gate's output, u: up's).
// The pointers come from an 8-dword block at s7 (low) / s9 (high): w or u,
// then t, then xout (0: not written). The mat-vec's own staging has issued
// its x loads into v[40 ..] (thread j: 16 bytes at 16 j + 4096 i) with v7 =
// 16 j; this goes where it would wait for them, and leaves v7..v13 alone.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>

#include "ops.h"

namespace ts5::direct {

enum class Stage { None, Norm, Silu };

// scratch: 32 bytes of LDS the kernel doesn't use. tgid: its workgroup id SGPR.
inline std::string gen_stage(Stage st, int ncols, float eps, int scratch, const std::string& tgid) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int nload = ncols / 1024, per = nload * 4;
    const int A = 40 + per, B = A + per;   // w or u, then t
    auto load_from = [&](const std::string& base, int dst) {
        o("s_mov_b32 s62, s" + base);
        o("s_mov_b32 s63, s" + std::to_string(std::stoi(base) + 1));
        for (int i = 0; i < nload; i++) {
            o("global_load_dwordx4 v[" + S(dst + 4 * i) + ":" + S(dst + 3 + 4 * i) + "], v7, s[62:63]");
            o("v_nop");
            o("s_add_u32 s62, s62, 4096");
            o("s_addc_u32 s63, s63, 0");
        }
    };
    o("s_mov_b32 s60, s7");
    o("s_mov_b32 s61, s9");
    o("s_load_dwordx8 s[52:59], s[60:61], 0x0");
    o("s_waitcnt lgkmcnt(0)");
    load_from("52", A);
    if (st == Stage::Norm) load_from("54", B);
    o("s_waitcnt vmcnt(0)");
    if (st == Stage::Silu) {
        for (int i = 0; i < per; i++) {
            const std::string g = "v" + S(40 + i);
            o("v_mul_f32 v20, " + f32_bits(-1.4426950408889634f) + ", " + g);
            o("v_exp_f32 v20, v20");
            o("v_add_f32 v20, 1.0, v20");
            o("v_rcp_f32 v20, v20");
            o("v_mul_f32 " + g + ", " + g + ", v20");
            o("v_mul_f32 " + g + ", v" + S(A + i) + ", " + g);
        }
        return k;
    }
    for (int i = 0; i < per; i++) o("v_add_f32 v" + S(40 + i) + ", v" + S(40 + i) + ", v" + S(B + i));
    // workgroup 0 keeps the sum, when asked
    o("s_or_b32 s64, s56, s57");
    o("s_cmp_eq_u32 s64, 0");
    o("s_cbranch_scc1 stage_nostore");
    o("s_cmp_lg_u32 " + tgid + ", 0");
    o("s_cbranch_scc1 stage_nostore");
    o("s_mov_b32 s62, s56");
    o("s_mov_b32 s63, s57");
    for (int i = 0; i < nload; i++) {
        o("global_store_dwordx4 v7, v[" + S(40 + 4 * i) + ":" + S(43 + 4 * i) + "], s[62:63]");
        o("v_nop");
        o("s_add_u32 s62, s62, 4096");
        o("s_addc_u32 s63, s63, 0");
    }
    k += "stage_nostore:\n";
    o("v_mul_f32 v20, v40, v40");
    for (int i = 1; i < per; i++) o("v_fmac_f32 v20, v" + S(40 + i) + ", v" + S(40 + i));
    o("v_nop");
    o("v_nop");
    for (int x : {1, 2, 4, 8}) o("v_add_f32_dpp v20, v20, v20 row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
    o("s_mov_b32 s64, 0x76543210");
    o("s_mov_b32 s65, 0xfedcba98");
    o("v_permlanex16_b32 v21, v20, s64, s65");
    o("v_add_f32 v20, v20, v21");
    o("v_lshrrev_b32 v21, 5, v0");
    o("v_lshlrev_b32 v21, 2, v21");
    o("ds_write_b32 v21, v20 offset:" + S(scratch));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_mov_b32 v21, 0");
    o("ds_read_b128 v[24:27], v21 offset:" + S(scratch));
    o("ds_read_b128 v[28:31], v21 offset:" + S(scratch + 16));
    o("s_waitcnt lgkmcnt(0)");
    o("v_add_f32 v20, v24, v25");
    for (int i = 26; i < 32; i++) o("v_add_f32 v20, v20, v" + S(i));
    o("v_mul_f32 v20, " + f32_bits(1.0f / ncols) + ", v20");
    o("v_add_f32 v20, " + f32_bits(eps) + ", v20");
    o("v_rsq_f32 v20, v20");
    for (int i = 0; i < per; i++) {
        o("v_mul_f32 v" + S(40 + i) + ", v20, v" + S(40 + i));
        o("v_mul_f32 v" + S(40 + i) + ", v" + S(A + i) + ", v" + S(40 + i));
    }
    return k;
}

}  // namespace ts5::direct
