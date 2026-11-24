// mm_q4k: Y = W X for T tokens at once (a prompt's batch), W a Q4_K matrix
// of R rows and K columns. One pass over the weights serves all T tokens,
// which is what makes a prompt fast: the token-at-a-time mat-vecs read the
// whole model once per token.
//
// Layout: one wave32 per workgroup, a lane per row (32 rows a wave). Each
// lane dequantizes its own row's blocks; the T tokens' x values for a column
// are the same for every lane, so they come in through scalar loads into
// SGPRs, which a VALU multiply-add takes as its first operand. Per value:
// about 1.25 ops to unpack the nibble, then T multiply-adds.
//
// Inside a 32-value sub-block: tmp[t] = sum q x[t]; at its end
// acc[t] += d sc tmp[t] - dmin m xs[t], xs the sub-block's sum of x (the
// host or the previous kernel provides it), so the minimum costs nothing per
// value.
//
// Arguments: s[0:1] W, s[2:3] xT (K x T, f32, a column's T values
// together), s[4:5] Y (T x R, f32), s6 R, s7 unused, s[8:9] xs (K/32 x T,
// f32); s10 workgroup id. Grid: R / 32 workgroups of 32 threads. W must have
// 144 readable bytes after its last row (the next block is loaded ahead).
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_mm_q4k(int K, int T) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int NB = K / 256, RB = NB * 144;
    const int A = 8, B = 44;                  // the block being worked on, the next one (36 dwords each)
    const int DSC = 80, DMM = 88, ACC = 96, TMP = 112, U = 128;
    const int SX = 20, SXS = 84;              // x: 4 columns of T; the sub-block sums: T
    k += ".kernel mm_q4k\n.wave32\n.user_sgprs 10\n.tgid_x\n";
    o("s_lshl_b32 s11, s10, 5");
    o("v_add_nc_u32 v3, s11, v0");             // the lane's row
    o("v_mul_u32_u24 v1, " + S(RB) + ", v3");  // its first byte
    o("v_lshlrev_b32 v3, 2, v3");
    for (int t = 0; t < T; t++) o("v_mov_b32 v" + S(ACC + t) + ", 0");
    auto load = [&](int base) {
        for (int i = 0; i < 9; i++) o("global_load_dwordx4 v[" + S(base + 4 * i) + ":" + S(base + 3 + 4 * i) + "], v1, s[0:1] offset:" + S(16 * i));
    };
    load(A);
    o("s_mov_b32 s12, s2");                    // x of the current block
    o("s_mov_b32 s13, s3");
    o("s_mov_b32 s14, s8");                    // its sub-block sums
    o("s_mov_b32 s15, s9");
    o("s_mov_b32 s16, " + S(NB));
    k += "block:\n";
    o("v_add_nc_u32 v1, 144, v1");
    load(B);                                   // the next block (past the row's end on its last: unused)
    o("s_waitcnt vmcnt(9)");
    // d, dmin; each sub-block's scale and min
    o("v_cvt_f32_f16 v" + S(U) + ", v" + S(A));
    o("v_lshrrev_b32 v" + S(U + 1) + ", 16, v" + S(A));
    o("v_cvt_f32_f16 v" + S(U + 1) + ", v" + S(U + 1));
    auto H = [&](int byte) { return "v" + S(A + 1 + byte / 4); };
    auto bit = [](int byte) { return 8 * (byte % 4); };
    for (int j = 0; j < 8; j++) {
        const std::string sc = "v" + S(DSC + j), m = "v" + S(DMM + j);
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
        o("v_cvt_f32_u32 " + sc + ", " + sc);
        o("v_cvt_f32_u32 " + m + ", " + m);
        o("v_mul_f32 " + sc + ", v" + S(U) + ", " + sc);
        o("v_mul_f32 " + m + ", v" + S(U + 1) + ", " + m);
    }
    auto sload = [&](int dst, int n, const std::string& base, int off) {
        // n dwords (8 or 16 at a time) into s[dst ..]
        for (int i = 0; i < n; i += (n - i >= 16 ? 16 : 8)) {
            const int w = n - i >= 16 ? 16 : 8;
            o("s_load_dwordx" + S(w) + " s[" + S(dst + i) + ":" + S(dst + i + w - 1) + "], " + base + ", " + S(off + 4 * i));
        }
    };
    for (int j = 0; j < 8; j++) {
        for (int dw = 0; dw < 8; dw++) {
            const std::string q = "v" + S(A + 4 + 8 * (j / 2) + dw);
            sload(SX, 4 * T, "s[12:13]", (32 * j + 4 * dw) * T * 4);
            if (j & 1) {
                o("v_lshrrev_b32 v" + S(U) + ", 4, " + q);
                o("v_and_b32 v" + S(U) + ", 0x0f0f0f0f, v" + S(U));
            } else {
                o("v_and_b32 v" + S(U) + ", 0x0f0f0f0f, " + q);
            }
            for (int b = 0; b < 4; b++) o("v_cvt_f32_ubyte" + S(b) + " v" + S(U + 1 + b) + ", v" + S(U));
            o("s_waitcnt lgkmcnt(0)");
            for (int c = 0; c < 4; c++)
                for (int t = 0; t < T; t++) {
                    const std::string x = "s" + S(SX + c * T + t), w = "v" + S(U + 1 + c);
                    if (dw == 0 && c == 0) o("v_mul_f32 v" + S(TMP + t) + ", " + x + ", " + w);
                    else o("v_fmac_f32 v" + S(TMP + t) + ", " + x + ", " + w);
                }
        }
        sload(SXS, T, "s[14:15]", j * T * 4);
        o("s_waitcnt lgkmcnt(0)");
        for (int t = 0; t < T; t++) {
            o("v_fmac_f32 v" + S(ACC + t) + ", v" + S(DSC + j) + ", v" + S(TMP + t));
            o("v_fma_f32 v" + S(ACC + t) + ", -v" + S(DMM + j) + ", s" + S(SXS + t) + ", v" + S(ACC + t));
        }
    }
    o("s_add_u32 s12, s12, " + S(256 * T * 4));
    o("s_addc_u32 s13, s13, 0");
    o("s_add_u32 s14, s14, " + S(8 * T * 4));
    o("s_addc_u32 s15, s15, 0");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 36; i++) o("v_mov_b32 v" + S(A + i) + ", v" + S(B + i));
    o("s_sub_u32 s16, s16, 1");
    o("s_cmp_lg_u32 s16, 0");
    o("s_cbranch_scc1 block");
    for (int t = 0; t < T; t++) {
        o("s_mul_i32 s17, s6, " + S(4 * t));
        o("s_add_u32 s18, s4, s17");
        o("s_addc_u32 s19, s5, 0");
        o("global_store_dword v3, v" + S(ACC + t) + ", s[18:19]");
    }
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
