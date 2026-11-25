// mm_q4k_w: the Q4_K prompt kernel for wide batches (T = 32 or 64 tokens a
// pass over the weights). Round 63: at T 16 the f16 kernel (mm_q4k_h.h) is
// near its instruction limit, and a 512-token prompt reads every weight 32
// times (about 150 GB at 183 GB/s); Vulkan's 858 tok/s reads them once.
// A wider T reads them T/16 times less, and the per-weight work (unpacking,
// scales) is shared by more tokens.
//
// Each lane is a row, the workgroup's 4 waves split each block (wave w:
// sub-blocks 2w, 2w + 1, as mm_q4k_h), x comes through scalar loads (the
// same for every lane). Unlike mm_q4k_h the weights are dequantized to f16
// first (w = d sc n - dmin m: v_pk_fma_f16 per pair of values), so a
// sub-block's sum needs no minimum term and no xs; a step is one column:
// T/2 v_pk_fma_f16 (two tokens each) with the column's weight.
//
// Precision: w in f16 (d sc and dmin m rounded to f16: 2^-11 relative), x16
// as mm_q4k_h (x 2^-E, one E per 32 columns for all tokens, E in xs32's low
// byte), each sub-block's 32 products summed in f16 (|w| ~ 0.1, |x16| < 16:
// far below 65504), then 2^E tmp into f32 (v_fma_mix_f32).
//
// x16: K x T/2 dwords. Arguments as mm_q4k_h: s[0:1] W, s[2:3] x16, s[4:5]
// Y (T x R), s6 R, s[8:9] xs32 (for E); s10 workgroup id. SK groups of 4
// waves a workgroup (blocks g, g + SK, ..). Grid: R / 32 workgroups of 128 SK.
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_mm_q4k_w(int K, int T, int SK = 1) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int NB = K / 256, RB = NB * 144, P = T / 2;
    if (NB % SK || (T != 32 && T != 64) || SK > 8) return "";
    // VGPRs: quants A (B: the next block's), SC16 / MN16 (f16 pairs, per
    // sub-block), W0 W1 the step's weights, U temporaries, VE 2^E, TMP (P
    // f16 pairs), ACC (T f32)
    const int A = 8, B = 20, SC16 = 32, MN16 = 34, W0 = 36, W1 = 37, U = 38, VE = 42, TMP = 43, ACC = TMP + P;
    const int SX = 20, SE = 84;   // x: two sets of one column (P dwords); E's xs32 dword
    k += ".kernel mm_q4k_w\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(4 * SK * 16 * 128) + "\n";
    o("v_lshrrev_b32 v2, 5, v0");
    o("v_readfirstlane_b32 s18, v2");                    // the wave: 4 g + w
    o("s_and_b32 s19, s18, 3");                          // w
    o("s_lshr_b32 s17, s18, 2");                         // g
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
    // x: the wave's first column (64 w of the block g), and xs32's (chunk 2 w of block g)
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
    o("s_mov_b32 s103, 0x64006400");                     // f16 1024, 1024 (n | 0x6400 = 1024 + n)
    o("s_mov_b32 s104, 0xe400e400");                     // f16 -1024, -1024
    auto sload = [&](int dst, int n, const std::string& base, int off) {
        for (int i = 0, w; i < n; i += w) {
            for (w = 16; w > n - i; w /= 2) {}
            o(w == 1 ? "s_load_dword s" + S(dst + i) + ", " + base + ", " + S(off + 4 * i)
                     : "s_load_dwordx" + S(w) + " s[" + S(dst + i) + ":" + S(dst + i + w - 1) + "], " + base + ", " + S(off + 4 * i));
        }
    };
    sload(SX, P, "s[12:13]", 0);
    k += "block:\n";
    o("v_add_nc_u32 v1, " + S(144 * SK) + ", v1");
    o("v_add_nc_u32 v2, " + S(144 * SK) + ", v2");
    o("s_cmp_eq_u32 s16, 1");                            // the next block's quants, unless this is the last
    o("s_cbranch_scc1 noload");
    load(B);
    k += "noload:\n";
    o("s_waitcnt vmcnt(3)");
    // the wave's two sub-blocks' d sc and dmin m (as mm_q4k_h), as f16 pairs (MN16 negated)
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
            const std::string sc = "v" + S(SC16 + jj), m = "v" + S(MN16 + jj);
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
        const std::string sc = "v" + S(SC16 + jj), m = "v" + S(MN16 + jj);
        o("v_cvt_f32_u32 " + sc + ", " + sc);
        o("v_cvt_f32_u32 " + m + ", " + m);
        o("v_mul_f32 " + sc + ", v" + S(U) + ", " + sc);
        o("v_mul_f32 " + m + ", -v" + S(U + 1) + ", " + m);
        o("v_cvt_pkrtz_f16_f32 " + sc + ", " + sc + ", " + sc);
        o("v_cvt_pkrtz_f16_f32 " + m + ", " + m + ", " + m);
    }
    // 64 steps of one column: sub-block jj, column c (quant dword c / 4).
    auto buf = [&](int st) { return SX + (st & 1) * P; };
    for (int st = 0; st < 64; st++) {
        const int jj = st / 32, c = st % 32, dw = c / 4;
        o("s_waitcnt lgkmcnt(0)");
        if (st + 1 < 64) sload(buf(st + 1), P, "s[12:13]", (st + 1) * P * 4);
        else {   // the next block's first column, unless this is the last
            o("s_cmp_eq_u32 s16, 1");
            o("s_cbranch_scc1 nox");
            sload(buf(0), P, "s[12:13]", 256 * SK * P * 4);
            k += "nox:\n";
        }
        if (c == 0) sload(SE, 1, "s[14:15]", jj * T * 4);   // the sub-block's E (any token's xs32)
        if (c % 4 == 0) {
            // dword dw's 4 values: n | 0x6400 - 1024 = n, then d sc n - dmin m. W0: columns 0, 2; W1: 1, 3
            const std::string q = "v" + S(A + 4 + dw), u = "v" + S(U), w0 = "v" + S(W0), w1 = "v" + S(W1);
            if (jj) {
                o("v_lshrrev_b32 " + u + ", 4, " + q);
                o("v_and_or_b32 " + w0 + ", " + u + ", 0x000f000f, s103");
            } else {
                o("v_and_or_b32 " + w0 + ", " + q + ", 0x000f000f, s103");
            }
            o("v_lshrrev_b32 " + u + ", " + S(jj ? 12 : 8) + ", " + q);
            o("v_and_or_b32 " + w1 + ", " + u + ", 0x000f000f, s103");
            o("v_pk_add_f16 " + w0 + ", " + w0 + ", s104");
            o("v_pk_add_f16 " + w1 + ", " + w1 + ", s104");
            o("v_pk_fma_f16 " + w0 + ", " + w0 + ", v" + S(SC16 + jj) + ", v" + S(MN16 + jj));
            o("v_pk_fma_f16 " + w1 + ", " + w1 + ", v" + S(SC16 + jj) + ", v" + S(MN16 + jj));
        }
        const std::string w = "v" + S((c & 1) ? W1 : W0);
        const bool hi = (c % 4) >> 1;
        for (int p = 0; p < P; p++) {
            const std::string x = "s" + S(buf(st) + p), d = "v" + S(TMP + p);
            if (c == 0) o("v_pk_mul_f16 " + d + ", " + x + ", " + w + (hi ? " op_sel:[0,1] op_sel_hi:[1,1]" : " op_sel_hi:[1,0]"));
            else o("v_pk_fma_f16 " + d + ", " + x + ", " + w + ", " + d + (hi ? " op_sel:[0,1,0] op_sel_hi:[1,1,1]" : " op_sel_hi:[1,0,1]"));
        }
        if (c == 31) {
            // the sub-block into f32: acc[t] += 2^E tmp[t]
            o("s_sext_i32_i8 s17, s" + S(SE));
            o("v_ldexp_f32 v" + S(VE) + ", 1.0, s17");
            for (int p = 0; p < P; p++)
                for (int h = 0; h < 2; h++) {
                    const std::string a = "v" + S(ACC + 2 * p + h);
                    o("v_fma_mix_f32 " + a + ", v" + S(VE) + ", v" + S(TMP + p) + ", " + a + " op_sel:[0," + S(h) + ",0] op_sel_hi:[0,1,0]");
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
    // the 4 SK waves' sums through LDS, 16 tokens at a time ([wave][16][lane]:
    // 8 KiB a group of 4 waves, so LDS doesn't cap how many workgroups a CU
    // holds); wave 0 adds them
    o("v_lshlrev_b32 v5, 2, v0");
    o("s_mul_i32 s11, s18, " + S(16 * 128));
    o("v_add_nc_u32 v5, s11, v5");
    o("v_lshlrev_b32 v6, 2, v0");
    for (int h = 0; h < T; h += 16) {
        if (h) o("s_barrier");                           // wave 0 has read the last group
        for (int t = 0; t < 16; t++) o("ds_write_b32 v5, v" + S(ACC + h + t) + " offset:" + S(t * 128));
        o("s_waitcnt lgkmcnt(0)");
        o("s_barrier");
        o("s_cmp_lg_u32 s18, 0");
        o("s_cbranch_scc1 skip" + S(h));
        for (int w = 1; w < 4 * SK; w++) {
            for (int t = 0; t < 16; t++) o("ds_read_b32 v" + S(TMP + t) + ", v6 offset:" + S((w * 16 + t) * 128));
            o("s_waitcnt lgkmcnt(0)");
            for (int t = 0; t < 16; t++) o("v_add_f32 v" + S(ACC + h + t) + ", v" + S(ACC + h + t) + ", v" + S(TMP + t));
        }
        k += "skip" + S(h) + ":\n";
    }
    o("s_cmp_lg_u32 s18, 0");
    o("s_cbranch_scc1 done");
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
