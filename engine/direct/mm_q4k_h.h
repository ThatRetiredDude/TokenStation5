// mm_q4k_h: mm_q4k_sb.h with packed half-precision multiply-adds. Each
// v_pk_fma_f16 does two tokens' multiply-adds, so a step needs half the
// instructions of the f32 kernel, and x (two tokens' values packed in a
// dword) is half the scalar loads. Round 50's prompt (360 tok/s) is at
// about 60% of the f32 FMA peak, and Vulkan's 858 tok/s is above that peak:
// only packed f16 math gets there (gfx1013 has no dot2 / dot4).
//
// Precision: x and the 4-bit quants in f16 (the quants exactly), each
// 32-value sub-block's sum in f16, then into f32 with its scale (d sc) and
// the minimum term (dmin m xs, all f32), as round 49's kernel.
// x16 is x 2^-E, E per 32 columns and token so that |x 2^-E| < 16 (each
// sum stays under 32 x 15 x 16 = 7680; round 53: unscaled, Qwen3's
// activations overflowed f16's 65504 and gave NaN). E is the low byte of
// xs32 (to_xT, its value's last 8 bits: 2^-15 relative), undone exactly
// with ldexp before the f32 scale.
//
// x16: K x T/2 dwords (to_xT writes it): column c's tokens 2p, 2p + 1 as
// f16 in dword c T/2 + p (low half 2p). Arguments as mm_q4k_sb but s[2:3]
// is x16. Grid: R / 32 workgroups of 128.
#pragma once
#include <string>

namespace ts5::direct {

// exp (timing only, wrong values): 1 no x loads in the loop, 2 no unpacking
// (the quant dword used as the f16 pair), 3 neither.
//
// SK (1, 2, 4, 8): SK groups of 4 waves a workgroup (128 SK threads), group
// g taking blocks g, g + SK, ...; the 4 SK partial sums added through LDS.
// Round 56: a launch runs at full speed only with ~2 or more workgroups' worth
// of waves for every slot (one round is ~280 workgroups of 4 waves; 128
// workgroups ran 45% slower a row, a second round of 8 workgroups +17%).
inline std::string gen_mm_q4k_h(int K, int T, int exp = 0, int SK = 1) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int NB = K / 256, RB = NB * 144, P = T / 2;   // dwords of x a column
    if (NB % SK) return "";
    const int A = 8, B = 20, SC = 32, MN = 34, ACC = 36, TMP = ACC + T, U = TMP + P;
    const int SX = 20, SXS = 84;                         // x: two sets of 4 columns (4 P); the sub-block sums: T (f32)
    k += ".kernel mm_q4k_h\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(4 * SK * T * 128) + "\n";
    o("v_lshrrev_b32 v2, 5, v0");
    o("v_readfirstlane_b32 s18, v2");                    // the wave: 4 g + w
    o("s_and_b32 s19, s18, 3");                          // w: its quarter of a block
    o("s_lshr_b32 s17, s18, 2");                         // g: blocks g, g + SK, ..
    o("v_and_b32 v0, 31, v0");
    o("s_lshl_b32 s11, s10, 5");
    o("v_add_nc_u32 v3, s11, v0");                       // the row
    o("v_mul_u32_u24 v1, " + S(RB) + ", v3");
    o("s_mul_i32 s11, s17, 144");
    o("v_add_nc_u32 v1, s11, v1");                       // block g
    o("s_lshl_b32 s11, s19, 5");
    o("s_add_u32 s11, s11, 16");
    o("v_add_nc_u32 v2, s11, v1");                       // its quants: 16 + 32 w
    o("v_lshlrev_b32 v3, 2, v3");
    for (int t = 0; t < T; t++) o("v_mov_b32 v" + S(ACC + t) + ", 0");
    auto load = [&](int base) {
        o("global_load_dwordx4 v[" + S(base) + ":" + S(base + 3) + "], v1, s[0:1]");
        o("global_load_dwordx4 v[" + S(base + 4) + ":" + S(base + 7) + "], v2, s[0:1]");
        o("global_load_dwordx4 v[" + S(base + 8) + ":" + S(base + 11) + "], v2, s[0:1] offset:16");
    };
    load(A);
    o("s_mul_i32 s11, s19, " + S(64 * P * 4));
    o("s_mul_i32 s16, s17, " + S(256 * P * 4));
    o("s_add_u32 s11, s11, s16");
    o("s_add_u32 s12, s2, s11");
    o("s_addc_u32 s13, s3, 0");
    o("s_mul_i32 s11, s19, " + S(2 * T * 4));
    o("s_mul_i32 s16, s17, " + S(8 * T * 4));
    o("s_add_u32 s11, s11, s16");
    o("s_add_u32 s14, s8, s11");
    o("s_addc_u32 s15, s9, 0");
    o("s_mov_b32 s16, " + S(NB / SK));
    auto sload = [&](int dst, int n, const std::string& base, int off) {
        for (int i = 0; i < n; i += (n - i >= 16 ? 16 : 8)) {
            const int w = n - i >= 16 ? 16 : 8;
            o("s_load_dwordx" + S(w) + " s[" + S(dst + i) + ":" + S(dst + i + w - 1) + "], " + base + ", " + S(off + 4 * i));
        }
    };
    sload(SX, 4 * P, "s[12:13]", 0);
    k += "block:\n";
    o("v_add_nc_u32 v1, " + S(144 * SK) + ", v1");
    o("v_add_nc_u32 v2, " + S(144 * SK) + ", v2");
    // the next block's quants, unless this is the last (past the matrix's end)
    o("s_cmp_eq_u32 s16, 1");
    o("s_cbranch_scc1 noload");
    load(B);
    k += "noload:\n";
    o("s_waitcnt vmcnt(3)");
    auto bit = [](int byte) { return 8 * (byte % 4); };
    auto H = [&](int byte) { return "v" + S(A + 1 + byte / 4); };
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
                o("v_bfe_u32 v" + S(U) + ", " + H(j - 4) + ", " + S(bit(j - 4) + 6) + ", 2");
                o("v_lshl_or_b32 " + sc + ", v" + S(U) + ", 4, " + sc);
                o("v_bfe_u32 " + m + ", " + H(j + 4) + ", " + S(bit(j + 4) + 4) + ", 4");
                o("v_bfe_u32 v" + S(U) + ", " + H(j) + ", " + S(bit(j) + 6) + ", 2");
                o("v_lshl_or_b32 " + m + ", v" + S(U) + ", 4, " + m);
            }
        }
        o("s_branch scdone");
    }
    k += "scdone:\n";
    o("v_cvt_f32_f16 v" + S(U) + ", v" + S(A));
    o("v_lshrrev_b32 v" + S(U + 1) + ", 16, v" + S(A));
    o("v_cvt_f32_f16 v" + S(U + 1) + ", v" + S(U + 1));
    for (int jj = 0; jj < 2; jj++) {
        const std::string sc = "v" + S(SC + jj), m = "v" + S(MN + jj);
        o("v_cvt_f32_u32 " + sc + ", " + sc);
        o("v_cvt_f32_u32 " + m + ", " + m);
        o("v_mul_f32 " + sc + ", v" + S(U) + ", " + sc);
        o("v_mul_f32 " + m + ", v" + S(U + 1) + ", " + m);
    }
    // 16 steps of 4 columns (a dword of quants): sub-block jj, dword dw.
    auto xoff = [&](int st) { const int jj = st / 8, dw = st % 8; return (32 * jj + 4 * dw) * P * 4; };
    auto buf = [&](int st) { return SX + (st & 1) * 4 * P; };
    for (int st = 0; st < 16; st++) {
        const int jj = st / 8, dw = st % 8;
        o("s_waitcnt lgkmcnt(0)");
        if (exp & 1) {}
        else if (st + 1 < 16) sload(buf(st + 1), 4 * P, "s[12:13]", xoff(st + 1));
        else {   // the next block's first step, unless this is the last
            o("s_cmp_eq_u32 s16, 1");
            o("s_cbranch_scc1 nox");
            sload(buf(0), 4 * P, "s[12:13]", 256 * SK * P * 4);
            k += "nox:\n";
        }
        if (dw == 6) sload(SXS, T, "s[14:15]", jj * T * 4);   // the sub-block's sums, for its end
        const std::string q = "v" + S(A + 4 + dw), u = "v" + S(U);
        if (exp & 2) {}
        else if (jj) {
            o("v_lshrrev_b32 " + u + ", 4, " + q);
            o("v_and_b32 " + u + ", 0x0f0f0f0f, " + u);
        } else {
            o("v_and_b32 " + u + ", 0x0f0f0f0f, " + q);
        }
        if (!(exp & 2)) {
            for (int b = 0; b < 4; b++) o("v_cvt_f32_ubyte" + S(b) + " v" + S(U + 1 + b) + ", " + u);
            o("v_cvt_pkrtz_f16_f32 v" + S(U + 5) + ", v" + S(U + 1) + ", v" + S(U + 2));
            o("v_cvt_pkrtz_f16_f32 v" + S(U + 6) + ", v" + S(U + 3) + ", v" + S(U + 4));
        }
        for (int c = 0; c < 4; c++) {
            const std::string w = exp & 2 ? q : "v" + S(U + 5 + c / 2);
            const bool hi = c & 1;
            for (int p = 0; p < P; p++) {
                const std::string x = "s" + S(buf(st) + c * P + p), d = "v" + S(TMP + p);
                if (dw == 0 && c == 0) o("v_pk_mul_f16 " + d + ", " + x + ", " + w + (hi ? " op_sel:[0,1] op_sel_hi:[1,1]" : " op_sel_hi:[1,0]"));
                else o("v_pk_fma_f16 " + d + ", " + x + ", " + w + ", " + d + (hi ? " op_sel:[0,1,0] op_sel_hi:[1,1,1]" : " op_sel_hi:[1,0,1]"));
            }
        }
        if (dw == 7) {
            // the sub-block into f32: acc[t] += d sc tmp[t] - dmin m xs[t]
            for (int p = 0; p < P; p++)
                for (int h = 0; h < 2; h++) {
                    const int t = 2 * p + h;
                    const std::string a = "v" + S(ACC + t), f = "v" + S(U + 7);
                    if (h) {
                        o("v_lshrrev_b32 " + f + ", 16, v" + S(TMP + p));
                        o("v_cvt_f32_f16 " + f + ", " + f);
                    } else {
                        o("v_cvt_f32_f16 " + f + ", v" + S(TMP + p));
                    }
                    o("s_sext_i32_i8 s17, s" + S(SXS + t));   // x16's scale exponent (to_xT)
                    o("v_ldexp_f32 " + f + ", " + f + ", s17");
                    o("v_fmac_f32 " + a + ", v" + S(SC + jj) + ", " + f);
                    o("v_fma_f32 " + a + ", -v" + S(MN + jj) + ", s" + S(SXS + t) + ", " + a);
                }
        }
    }
    o("s_add_u32 s12, s12, " + S(256 * SK * P * 4));
    o("s_addc_u32 s13, s13, 0");
    o("s_add_u32 s14, s14, " + S(8 * SK * T * 4));
    o("s_addc_u32 s15, s15, 0");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 12; i++) o("v_mov_b32 v" + S(A + i) + ", v" + S(B + i));
    o("s_sub_u32 s16, s16, 1");
    o("s_cmp_lg_u32 s16, 0");
    o("s_cbranch_scc1 block");
    o("v_lshlrev_b32 v5, 2, v0");
    o("s_mul_i32 s11, s18, " + S(T * 128));
    o("v_add_nc_u32 v5, s11, v5");
    for (int t = 0; t < T; t++) o("ds_write_b32 v5, v" + S(ACC + t) + " offset:" + S(t * 128));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("s_cmp_lg_u32 s18, 0");
    o("s_cbranch_scc1 done");
    o("v_lshlrev_b32 v5, 2, v0");
    // (TMP and U, contiguous, hold the T partials read back)
    for (int w = 1; w < 4 * SK; w++) {
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
