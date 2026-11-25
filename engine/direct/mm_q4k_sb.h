// mm_q4k_sb: the prompt kernel of mm_q4k.h with each Q4_K block split
// across the workgroup's 4 waves by sub-block pairs instead of each wave
// taking a quarter of the row's blocks.
//
// Why (rounds 46-48): the per-step x loads were 57% of a pass, and the same
// loads served from one cached line cost nothing, so the cost was scalar
// cache misses. With a quarter of the blocks per wave, the waves of a WGP
// read 4 different stretches of x; here every wave of every workgroup walks
// the same blocks at the same time, wave w reading only the 64 columns of
// sub-blocks 2w and 2w + 1, so they share the scalar cache's lines. Each
// lane loads 48 of a block's 144 bytes (d, dmin, the scales, and the 32
// bytes of quants that hold its two sub-blocks: low nibbles 2w, high 2w + 1),
// which also cuts the VGPRs to about a half.
//
// Arguments as mm_q4k: s[0:1] W, s[2:3] xT (K x T), s[4:5] Y (T x R), s6 R,
// s[8:9] xs (K/32 x T); s10 workgroup id. Grid: R / 32 workgroups of 128.
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_mm_q4k_sb(int K, int T) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int NB = K / 256, RB = NB * 144;
    const int A = 8, B = 20;                       // a block's 12 dwords: header 4, quants 8; the next block's
    const int SC = 32, MN = 34;                    // d sc, dmin m of the two sub-blocks
    const int ACC = 40, TMP = 40 + T, U = 40 + 2 * T;
    const int SX = 20, SXS = 84;
    k += ".kernel mm_q4k_sb\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(4 * T * 128) + "\n";
    o("v_lshrrev_b32 v2, 5, v0");
    o("v_readfirstlane_b32 s19, v2");              // w
    o("v_and_b32 v0, 31, v0");
    o("s_lshl_b32 s11, s10, 5");
    o("v_add_nc_u32 v3, s11, v0");                 // the row
    o("v_mul_u32_u24 v1, " + S(RB) + ", v3");      // its block's header
    o("s_lshl_b32 s11, s19, 5");
    o("s_add_u32 s11, s11, 16");
    o("v_add_nc_u32 v2, s11, v1");                 // its quants: 16 + 32 w
    o("v_lshlrev_b32 v3, 2, v3");
    for (int t = 0; t < T; t++) o("v_mov_b32 v" + S(ACC + t) + ", 0");
    auto load = [&](int base) {
        o("global_load_dwordx4 v[" + S(base) + ":" + S(base + 3) + "], v1, s[0:1]");
        o("global_load_dwordx4 v[" + S(base + 4) + ":" + S(base + 7) + "], v2, s[0:1]");
        o("global_load_dwordx4 v[" + S(base + 8) + ":" + S(base + 11) + "], v2, s[0:1] offset:16");
    };
    load(A);
    o("s_mul_i32 s11, s19, " + S(64 * T * 4));
    o("s_add_u32 s12, s2, s11");                   // x of the wave's 64 columns of the current block
    o("s_addc_u32 s13, s3, 0");
    o("s_mul_i32 s11, s19, " + S(2 * T * 4));
    o("s_add_u32 s14, s8, s11");                   // their sub-block sums
    o("s_addc_u32 s15, s9, 0");
    o("s_mov_b32 s16, " + S(NB));
    auto sload = [&](int dst, int n, const std::string& base, int off) {
        for (int i = 0; i < n; i += (n - i >= 16 ? 16 : 8)) {
            const int w = n - i >= 16 ? 16 : 8;
            o("s_load_dwordx" + S(w) + " s[" + S(dst + i) + ":" + S(dst + i + w - 1) + "], " + base + ", " + S(off + 4 * i));
        }
    };
    sload(SX, 2 * T, "s[12:13]", 0);
    k += "block:\n";
    o("v_add_nc_u32 v1, 144, v1");
    o("v_add_nc_u32 v2, 144, v2");
    load(B);
    o("s_waitcnt vmcnt(3)");
    // d, dmin; the scale and min of sub-blocks 2w, 2w + 1 (w uniform: a branch per w)
    o("v_cvt_f32_f16 v" + S(U) + ", v" + S(A));
    o("v_lshrrev_b32 v" + S(U + 1) + ", 16, v" + S(A));
    o("v_cvt_f32_f16 v" + S(U + 1) + ", v" + S(U + 1));
    auto H = [&](int byte) { return "v" + S(A + 1 + byte / 4); };
    auto bit = [](int byte) { return 8 * (byte % 4); };
    for (int w = 0; w < 4; w++) {
        k += "sc" + S(w) + ":\n";
        if (w < 3) {
            o("s_cmp_lg_u32 s19, " + S(w));
            o("s_cbranch_scc1 sc" + S(w + 1));
        }
        for (int jj = 0; jj < 2; jj++) {
            const int j = 2 * w + jj;
            const std::string sc = "v" + S(SC + jj), m = "v" + S(MN + jj);
            if (j < 4) {
                o("v_bfe_u32 " + sc + ", " + H(j) + ", " + S(bit(j)) + ", 6");
                o("v_bfe_u32 " + m + ", " + H(j + 4) + ", " + S(bit(j + 4)) + ", 6");
            } else {
                o("v_bfe_u32 " + sc + ", " + H(j + 4) + ", " + S(bit(j + 4)) + ", 4");
                o("v_bfe_u32 v" + S(U + 2) + ", " + H(j - 4) + ", " + S(bit(j - 4) + 6) + ", 2");
                o("v_lshl_or_b32 " + sc + ", v" + S(U + 2) + ", 4, " + sc);
                o("v_bfe_u32 " + m + ", " + H(j + 4) + ", " + S(bit(j + 4) + 4) + ", 4");
                o("v_bfe_u32 v" + S(U + 2) + ", " + H(j) + ", " + S(bit(j) + 6) + ", 2");
                o("v_lshl_or_b32 " + m + ", v" + S(U + 2) + ", 4, " + m);
            }
        }
        o("s_branch scdone");
    }
    k += "scdone:\n";
    for (int jj = 0; jj < 2; jj++) {
        const std::string sc = "v" + S(SC + jj), m = "v" + S(MN + jj);
        o("v_cvt_f32_u32 " + sc + ", " + sc);
        o("v_cvt_f32_u32 " + m + ", " + m);
        o("v_mul_f32 " + sc + ", v" + S(U) + ", " + sc);
        o("v_mul_f32 " + m + ", v" + S(U + 1) + ", " + m);
    }
    // 32 steps of 2 columns: sub-block jj (low / high nibbles), dword dw, half h.
    auto xoff = [&](int st) { const int jj = st / 16, dw = (st / 2) % 8, h = st % 2; return (32 * jj + 4 * dw + 2 * h) * T * 4; };
    auto buf = [&](int st) { return SX + (st & 1) * 2 * T; };
    for (int st = 0; st < 32; st++) {
        const int jj = st / 16, dw = (st / 2) % 8, h = st % 2;
        o("s_waitcnt lgkmcnt(0)");
        if (st + 1 < 32) sload(buf(st + 1), 2 * T, "s[12:13]", xoff(st + 1));
        else sload(buf(0), 2 * T, "s[12:13]", 256 * T * 4);   // the next block's first step
        if (dw == 7 && h == 0) sload(SXS, T, "s[14:15]", jj * T * 4);
        const std::string q = "v" + S(A + 4 + dw);
        if (h == 0) {
            if (jj) {
                o("v_lshrrev_b32 v" + S(U) + ", 4, " + q);
                o("v_and_b32 v" + S(U) + ", 0x0f0f0f0f, v" + S(U));
            } else {
                o("v_and_b32 v" + S(U) + ", 0x0f0f0f0f, " + q);
            }
            for (int b = 0; b < 4; b++) o("v_cvt_f32_ubyte" + S(b) + " v" + S(U + 1 + b) + ", v" + S(U));
        }
        for (int c = 0; c < 2; c++)
            for (int t = 0; t < T; t++) {
                const std::string x = "s" + S(buf(st) + c * T + t), w = "v" + S(U + 1 + 2 * h + c);
                if (dw == 0 && h == 0 && c == 0) o("v_mul_f32 v" + S(TMP + t) + ", " + x + ", " + w);
                else o("v_fmac_f32 v" + S(TMP + t) + ", " + x + ", " + w);
            }
        if (dw == 7 && h == 1) {
            o("s_waitcnt lgkmcnt(0)");
            for (int t = 0; t < T; t++) {
                o("v_fmac_f32 v" + S(ACC + t) + ", v" + S(SC + jj) + ", v" + S(TMP + t));
                o("v_fma_f32 v" + S(ACC + t) + ", -v" + S(MN + jj) + ", s" + S(SXS + t) + ", v" + S(ACC + t));
            }
        }
    }
    o("s_add_u32 s12, s12, " + S(256 * T * 4));
    o("s_addc_u32 s13, s13, 0");
    o("s_add_u32 s14, s14, " + S(8 * T * 4));
    o("s_addc_u32 s15, s15, 0");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 12; i++) o("v_mov_b32 v" + S(A + i) + ", v" + S(B + i));
    o("s_sub_u32 s16, s16, 1");
    o("s_cmp_lg_u32 s16, 0");
    o("s_cbranch_scc1 block");
    o("v_lshlrev_b32 v5, 2, v0");
    o("s_mul_i32 s11, s19, " + S(T * 128));
    o("v_add_nc_u32 v5, s11, v5");
    for (int t = 0; t < T; t++) o("ds_write_b32 v5, v" + S(ACC + t) + " offset:" + S(t * 128));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("s_cmp_lg_u32 s19, 0");
    o("s_cbranch_scc1 done");
    o("v_lshlrev_b32 v5, 2, v0");
    for (int w = 1; w < 4; w++) {
        for (int t = 0; t < T; t++) o("ds_read_b32 v" + S(TMP + t) + ", v5 offset:" + S((w * T + t) * 128));
        o("s_waitcnt lgkmcnt(0)");
        for (int t = 0; t < T; t++) o("v_add_f32 v" + S(ACC + t) + ", v" + S(ACC + t) + ", v" + S(TMP + t));
    }
    for (int t = 0; t < T; t++) {
        o("s_mul_i32 s17, s6, " + S(4 * t));
        o("s_add_u32 s100, s4, s17");
        o("s_addc_u32 s101, s5, 0");
        o("global_store_dword v3, v" + S(ACC + t) + ", s[100:101]");
    }
    k += "done:\n";
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
