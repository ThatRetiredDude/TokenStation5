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
//
// RW (1, 2): rows a lane (rows r and r + 32), so each x value loaded serves
// RW rows (round 55: the x loads cost 25% on attn_q and ffn_down).
// Grid: R / (32 RW) workgroups of 128 SK.
inline std::string gen_mm_q4k_h(int K, int T, int exp = 0, int SK = 1, int RW = 1) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int NB = K / 256, RB = NB * 144, P = T / 2;   // dwords of x a column
    if (NB % SK) return "";
    // per row r: quants A + 12 r (B + 12 r), scales SC + 2 r, minimums MN + 2 r, sums ACC + T r, f16 sums TMP + P r
    const int A = 8, B = A + 12 * RW, SC = B + 12 * RW, MN = SC + 2 * RW, ACC = MN + 2 * RW, TMP = ACC + T * RW, U = TMP + P * RW;
    const int SX = 20, SXS = 84;                         // x: two sets of 4 columns (4 P); the sub-block sums: T (f32)
    k += ".kernel mm_q4k_h\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(4 * SK * RW * T * 128) + "\n";
    o("v_lshrrev_b32 v2, 5, v0");
    o("v_readfirstlane_b32 s18, v2");                    // the wave: 4 g + w
    o("s_and_b32 s19, s18, 3");                          // w: its quarter of a block
    o("s_lshr_b32 s17, s18, 2");                         // g: blocks g, g + SK, ..
    o("v_and_b32 v0, 31, v0");
    o("s_lshl_b32 s11, s10, " + S(RW == 2 ? 6 : 5));
    o("v_add_nc_u32 v3, s11, v0");                       // the row (and row + 32)
    o("v_mul_u32_u24 v1, " + S(RB) + ", v3");
    o("s_mul_i32 s11, s17, 144");
    o("v_add_nc_u32 v1, s11, v1");                       // block g
    o("s_lshl_b32 s11, s19, 5");
    o("s_add_u32 s11, s11, 16");
    o("v_add_nc_u32 v2, s11, v1");                       // its quants: 16 + 32 w
    if (RW == 2) {                                       // row + 32's: v6, v7
        o("v_add_nc_u32 v6, " + S(32 * RB) + ", v1");
        o("v_add_nc_u32 v7, " + S(32 * RB) + ", v2");
    }
    o("v_lshlrev_b32 v3, 2, v3");
    for (int t = 0; t < T * RW; t++) o("v_mov_b32 v" + S(ACC + t) + ", 0");
    o("s_mov_b32 s103, 0x64006400");                     // f16 1024, 1024 (the quants' exponent trick)
    o("s_mov_b32 s104, 0xe400e400");                     // f16 -1024, -1024
    auto load = [&](int base) {
        for (int r = 0; r < RW; r++) {
            const std::string h = r ? "v6" : "v1", q = r ? "v7" : "v2";
            const int b = base + 12 * r;
            o("global_load_dwordx4 v[" + S(b) + ":" + S(b + 3) + "], " + h + ", s[0:1]");
            o("global_load_dwordx4 v[" + S(b + 4) + ":" + S(b + 7) + "], " + q + ", s[0:1]");
            o("global_load_dwordx4 v[" + S(b + 8) + ":" + S(b + 11) + "], " + q + ", s[0:1] offset:16");
        }
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
    for (const char* v : {"v1", "v2", "v6", "v7"})
        if (RW == 2 || v[1] < '3') o(std::string("v_add_nc_u32 ") + v + ", " + S(144 * SK) + ", " + v);
    // the next block's quants, unless this is the last (past the matrix's end)
    o("s_cmp_eq_u32 s16, 1");
    o("s_cbranch_scc1 noload");
    load(B);
    k += "noload:\n";
    o("s_waitcnt vmcnt(" + S(3 * RW) + ")");
    auto bit = [](int byte) { return 8 * (byte % 4); };
    std::string Hr;
    auto H = [&](int byte) { return "v" + S(std::stoi(Hr) + 1 + byte / 4); };
    for (int w = 0; w < 4; w++) {
        k += "sc" + S(w) + ":\n";
        if (w < 3) {
            o("s_cmp_lg_u32 s19, " + S(w));
            o("s_cbranch_scc1 sc" + S(w + 1));
        }
        for (int r = 0; r < RW; r++)
        for (int jj = 0; jj < 2; jj++) {
            Hr = S(A + 12 * r);
            const int j = 2 * w + jj;
            const std::string sc = "v" + S(SC + 2 * r + jj), m = "v" + S(MN + 2 * r + jj);
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
    for (int r = 0; r < RW; r++) {
    o("v_cvt_f32_f16 v" + S(U) + ", v" + S(A + 12 * r));
    o("v_lshrrev_b32 v" + S(U + 1) + ", 16, v" + S(A + 12 * r));
    o("v_cvt_f32_f16 v" + S(U + 1) + ", v" + S(U + 1));
    for (int jj = 0; jj < 2; jj++) {
        const std::string sc = "v" + S(SC + 2 * r + jj), m = "v" + S(MN + 2 * r + jj);
        o("v_cvt_f32_u32 " + sc + ", " + sc);
        o("v_cvt_f32_u32 " + m + ", " + m);
        o("v_mul_f32 " + sc + ", v" + S(U) + ", " + sc);
        o("v_mul_f32 " + m + ", v" + S(U + 1) + ", " + m);
    }
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
        for (int r = 0; r < RW; r++) {
        // The quants to f16 pairs by the exponent trick (round 62): a nibble
        // n or'd into 0x6400 is the f16 1024 + n, less 1024 exactly. W0 holds
        // columns 0 and 2 (bytes 0, 2), W1 columns 1 and 3.
        const std::string q = "v" + S(A + 12 * r + 4 + dw), u = "v" + S(U), w0 = "v" + S(U + 5), w1 = "v" + S(U + 6);
        if (!(exp & 2)) {
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
        }
        for (int c = 0; c < 4; c++) {
            const std::string w = exp & 2 ? q : (c & 1) ? w1 : w0;
            const bool hi = c >> 1;
            for (int p = 0; p < P; p++) {
                const std::string x = "s" + S(buf(st) + c * P + p), d = "v" + S(TMP + P * r + p);
                if (dw == 0 && c == 0) o("v_pk_mul_f16 " + d + ", " + x + ", " + w + (hi ? " op_sel:[0,1] op_sel_hi:[1,1]" : " op_sel_hi:[1,0]"));
                else o("v_pk_fma_f16 " + d + ", " + x + ", " + w + ", " + d + (hi ? " op_sel:[0,1,0] op_sel_hi:[1,1,1]" : " op_sel_hi:[1,0,1]"));
            }
        }
        }
        if (dw == 7) {
            // the sub-block into f32: acc[t] += d sc 2^E tmp[t] - dmin m xs[t]
            // (E: x16's exponent, the same for every token, in xs's low byte)
            o("s_sext_i32_i8 s17, s" + S(SXS));
            for (int r = 0; r < RW; r++) o("v_ldexp_f32 v" + S(U + 5 + r) + ", v" + S(SC + 2 * r + jj) + ", s17");
            for (int p = 0; p < P; p++)
                for (int h = 0; h < 2; h++) {
                    const int t = 2 * p + h;
                    for (int r = 0; r < RW; r++) {
                        const std::string a = "v" + S(ACC + T * r + t), tmp = "v" + S(TMP + P * r + p);
                        o("v_fma_mix_f32 " + a + ", v" + S(U + 5 + r) + ", " + tmp + ", " + a + " op_sel:[0," + S(h) + ",0] op_sel_hi:[0,1,0]");
                        o("v_fma_f32 " + a + ", -v" + S(MN + 2 * r + jj) + ", s" + S(SXS + t) + ", " + a);
                    }
                }
        }
    }
    o("s_add_u32 s12, s12, " + S(256 * SK * P * 4));
    o("s_addc_u32 s13, s13, 0");
    o("s_add_u32 s14, s14, " + S(8 * SK * T * 4));
    o("s_addc_u32 s15, s15, 0");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 12 * RW; i++) o("v_mov_b32 v" + S(A + i) + ", v" + S(B + i));
    o("s_sub_u32 s16, s16, 1");
    o("s_cmp_lg_u32 s16, 0");
    o("s_cbranch_scc1 block");
    o("v_lshlrev_b32 v5, 2, v0");
    o("s_mul_i32 s11, s18, " + S(RW * T * 128));
    o("v_add_nc_u32 v5, s11, v5");
    for (int t = 0; t < RW * T; t++) o("ds_write_b32 v5, v" + S(ACC + t) + " offset:" + S(t * 128));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("s_cmp_lg_u32 s18, 0");
    o("s_cbranch_scc1 done");
    o("v_lshlrev_b32 v5, 2, v0");
    // (TMP and U, contiguous, hold the T partials read back)
    for (int w = 1; w < 4 * SK; w++)
        for (int r = 0; r < RW; r++) {
            for (int t = 0; t < T; t++) o("ds_read_b32 v" + S(TMP + t) + ", v5 offset:" + S(((w * RW + r) * T + t) * 128));
            o("s_waitcnt lgkmcnt(0)");
            for (int t = 0; t < T; t++) o("v_add_f32 v" + S(ACC + T * r + t) + ", v" + S(ACC + T * r + t) + ", v" + S(TMP + t));
        }
    for (int t = 0; t < T; t++) {
        o("s_mul_i32 s17, s6, " + S(4 * t));
        o("s_add_u32 s100, s4, s17");
        o("s_addc_u32 s101, s5, 0");
        for (int r = 0; r < RW; r++) o("global_store_dword v3, v" + S(ACC + T * r + t) + ", s[100:101]" + (r ? " offset:128" : ""));
    }
    k += "done:\n";
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
