// mm_q6k: Y = W X for T tokens at once, W a Q6_K matrix (blocks of 256
// values in 210 bytes: 128 bytes of low nibbles, 64 of high bit pairs, 16
// int8 scales, f16 d). The same frame as mm_q4k.h: a lane per row, the
// tokens' x through scalar loads, the next block loaded ahead.
//
// A block's 16 values of one scale: tmp[t] = sum q x[t] (q the 6-bit
// value, 0..63); at its end acc[t] += d sc (tmp[t] - 32 xs[t]), xs the
// 16 values' sum of x.
//
// 210-byte blocks: every other one starts 2 bytes into a dword, and the
// GPU rounds an unaligned load's address down (round 21), so each block is
// loaded from its dword-aligned start (54 dwords) and, when odd, shifted
// down 2 bytes with v_alignbyte_b32.
//
// Arguments: s[0:1] W, s[2:3] xT (K x T), s[4:5] Y (T x R), s6 R, s7 unused,
// s[8:9] xs (K/16 x T); s10 workgroup id. Grid: R / 32 workgroups of 32.
// W needs 216 readable bytes after its last row, xT 2T dwords.
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_mm_q6k(int K, int T, int SPLIT = 4) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int NB = K / 256, RB = NB * 210;
    const int A = 8, B = 62;                   // 54 dwords each
    const int ACC = 116, TMP = 132, U = 148, M32 = 157;
    const int SX = 20, SXS = 84;
    // SPLIT waves a workgroup, each its own slice of the blocks, summed in
    // LDS at the end (round 40: with a wave per 32 rows and the whole row,
    // 4096-row shapes left most SIMDs with one or two waves).
    const int PB = NB / SPLIT;                 // blocks a wave
    k += ".kernel mm_q6k\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(SPLIT * T * 128) + "\n";
    o("v_lshrrev_b32 v2, 5, v0");
    o("v_readfirstlane_b32 s19, v2");          // the wave: its slice
    o("v_and_b32 v0, 31, v0");
    o("s_lshl_b32 s11, s10, 5");
    o("v_add_nc_u32 v3, s11, v0");
    o("v_mul_u32_u24 v1, " + S(RB) + ", v3");
    o("s_mul_i32 s11, s19, " + S(PB * 210));
    o("v_add_nc_u32 v1, s11, v1");
    o("v_lshlrev_b32 v3, 2, v3");
    o("v_mov_b32 v" + S(M32) + ", " + "0xc2000000");   // -32.0
    for (int t = 0; t < T; t++) o("v_mov_b32 v" + S(ACC + t) + ", 0");
    auto load = [&](int base) {
        for (int i = 0; i < 13; i++) o("global_load_dwordx4 v[" + S(base + 4 * i) + ":" + S(base + 3 + 4 * i) + "], v1, s[0:1] offset:" + S(16 * i));
        o("global_load_dwordx2 v[" + S(base + 52) + ":" + S(base + 53) + "], v1, s[0:1] offset:208");
    };
    load(A);
    o("s_mul_i32 s11, s19, " + S(PB * 256 * T * 4));
    o("s_add_u32 s12, s2, s11");
    o("s_addc_u32 s13, s3, 0");
    o("s_mul_i32 s11, s19, " + S(PB * 16 * T * 4));
    o("s_add_u32 s14, s8, s11");
    o("s_addc_u32 s15, s9, 0");
    o("s_mov_b32 s16, " + S(PB));
    o("s_mov_b32 s18, 0");                      // this block odd
    for (int i = 0; i < 2 * T; i += (2 * T - i >= 16 ? 16 : 8)) {
        const int w = 2 * T - i >= 16 ? 16 : 8;
        o("s_load_dwordx" + S(w) + " s[" + S(SX + i) + ":" + S(SX + i + w - 1) + "], s[12:13], " + S(4 * i));
    }
    k += "block:\n";
    // the next block's aligned start: + 208 from an even block, + 212 from an odd one
    o("s_lshl_b32 s17, s18, 2");
    o("s_add_u32 s17, s17, 208");
    o("v_add_nc_u32 v1, s17, v1");
    load(B);
    o("s_waitcnt vmcnt(14)");
    o("s_cmp_eq_u32 s18, 0");
    o("s_cbranch_scc1 aligned");
    for (int i = 0; i < 53; i++) o("v_alignbyte_b32 v" + S(A + i) + ", v" + S(A + i + 1) + ", v" + S(A + i) + ", 2");
    k += "aligned:\n";
    o("v_cvt_f32_f16 v4, v" + S(A + 52));      // d
    auto sload = [&](int dst, int n, const std::string& base, int off) {
        for (int i = 0; i < n; i += (n - i >= 16 ? 16 : 8)) {
            const int w = n - i >= 16 ? 16 : 8;
            o("s_load_dwordx" + S(w) + " s[" + S(dst + i) + ":" + S(dst + i + w - 1) + "], " + base + ", " + S(off + 4 * i));
        }
    };
    auto xoff = [&](int st) { const int g = st / 8, kk = (st / 2) % 4, h = st % 2; return (16 * g + 4 * kk + 2 * h) * T * 4; };
    auto buf = [&](int st) { return SX + (st & 1) * 2 * T; };
    for (int st = 0; st < 128; st++) {
        const int g = st / 8, kk = (st / 2) % 4, h = st % 2;
        const int n = g / 8, r = g % 8, qq = r / 2, is = r % 2;
        o("s_waitcnt lgkmcnt(0)");
        if (st + 1 < 128) sload(buf(st + 1), 2 * T, "s[12:13]", xoff(st + 1));
        else sload(buf(0), 2 * T, "s[12:13]", 256 * T * 4);
        if (kk == 3 && h == 0) sload(SXS, T, "s[14:15]", g * T * 4);
        if (h == 0) {
            const std::string ql = "v" + S(A + 16 * n + 8 * (qq & 1) + 4 * is + kk), qh = "v" + S(A + 32 + 8 * n + 4 * is + kk);
            const std::string u = "v" + S(U), u1 = "v" + S(U + 1);
            if (qq < 2) o("v_and_b32 " + u + ", 0x0f0f0f0f, " + ql);
            else {
                o("v_lshrrev_b32 " + u + ", 4, " + ql);
                o("v_and_b32 " + u + ", 0x0f0f0f0f, " + u);
            }
            if (qq) {
                o("v_lshrrev_b32 " + u1 + ", " + S(2 * qq) + ", " + qh);
                o("v_and_b32 " + u1 + ", 0x03030303, " + u1);
            } else {
                o("v_and_b32 " + u1 + ", 0x03030303, " + qh);
            }
            o("v_lshl_or_b32 " + u + ", " + u1 + ", 4, " + u);
            for (int b = 0; b < 4; b++) o("v_cvt_f32_ubyte" + S(b) + " v" + S(U + 1 + b) + ", " + u);
        }
        for (int c = 0; c < 2; c++)
            for (int t = 0; t < T; t++) {
                const std::string x = "s" + S(buf(st) + c * T + t), w = "v" + S(U + 1 + 2 * h + c);
                if (kk == 0 && h == 0 && c == 0) o("v_mul_f32 v" + S(TMP + t) + ", " + x + ", " + w);
                else o("v_fmac_f32 v" + S(TMP + t) + ", " + x + ", " + w);
            }
        if (kk == 3 && h == 1) {
            o("v_bfe_i32 v" + S(U + 5) + ", v" + S(A + 48 + g / 4) + ", " + S(8 * (g % 4)) + ", 8");
            o("v_cvt_f32_i32 v" + S(U + 5) + ", v" + S(U + 5));
            o("v_mul_f32 v" + S(U + 5) + ", v4, v" + S(U + 5));   // d sc
            o("s_waitcnt lgkmcnt(0)");
            for (int t = 0; t < T; t++) {
                o("v_fmac_f32 v" + S(TMP + t) + ", s" + S(SXS + t) + ", v" + S(M32));
                o("v_fmac_f32 v" + S(ACC + t) + ", v" + S(U + 5) + ", v" + S(TMP + t));
            }
        }
    }
    o("s_add_u32 s12, s12, " + S(256 * T * 4));
    o("s_addc_u32 s13, s13, 0");
    o("s_add_u32 s14, s14, " + S(16 * T * 4));
    o("s_addc_u32 s15, s15, 0");
    o("s_xor_b32 s18, s18, 1");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 54; i++) o("v_mov_b32 v" + S(A + i) + ", v" + S(B + i));
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
    for (int w = 1; w < SPLIT; w++) {
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
