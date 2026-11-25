// mm_q4k_t: the prompt mat-mat as a tiled GEMM. Each workgroup of 256 takes
// RM rows x T tokens (T 128: 128 x 128; T 64: 256 x 64), each thread 8 rows x
// 8 tokens, the weights and x of one 32-column sub-block at a time staged in
// LDS (the quants as exact f16 0..15), so every weight byte is read from
// memory once per T tokens instead of once per 16 (rounds 72-77: the 16-token
// kernels stall on their weight and x streams at ~658 tok/s; Vulkan's 858
// reads a prompt in large tiles).
//
// Math as mm_q4k_h: per sub-block, the f16 sums of q x16 (|x16| < 16, so
// under 32 x 15 x 16), then acc += d sc 2^E sum - dmin m xs in f32.
// Each k step: one ds_read_b128 of 8 rows' quants (f16) and one of 8 tokens'
// x (4 pairs), 32 v_pk_fma_f16 (the row's quant broadcast by op_sel).
// A sub-block's loads go out a whole sub-block ahead (registers), and are
// written to LDS between two barriers.
//
// Arguments as mm_q4k_h: s[0:1] W (Q4_K rows), s[2:3] x16 (K x T f16, a
// column's tokens together), s[4:5] y (T x R f32; with SK, SK of them, R T
// floats apart, to be summed), s6 R, s[8:9] xs32 (K/32 x T: the sub-block
// sums, E in the low byte). Grid: R / RM x SK workgroups of 256; workgroup
// (x, y) takes rows RM x .., blocks y K/256/SK ...
#pragma once
#include <functional>
#include <string>

namespace ts5::direct {

inline int mm_q4k_t_rows(int T) { return T == 64 ? 256 : 128; }

inline std::string gen_mm_q4k_t(int K, int T, int SK = 1) {
    if (T != 64 && T != 128) return "";
    const int NS = K / 32, NBK = K / 256;
    if (K % 256 || NBK % SK) return "";
    const int NSW = NS / SK;                               // sub-blocks a workgroup (whole blocks)
    const int TG = T / 8, RG = 256 / TG, RM = 8 * RG, WPR = RG / 8, LWPR = WPR == 2 ? 1 : 2;
    const int RB = NBK * 144;
    const int QN = RM * 32 / 256 / 4, XN = T / 16;          // staged quant / x dwords a thread
    const int LW = 0, LX = 64 * RM, LSC = LX + 64 * T, LMN = LSC + 32 * RM, LXS = LMN + 32 * RM, LDS = LXS + 4 * T;
    // VGPRs: v0 thread, v1 W read (rg 16), v2 x read (tg 16), v3 quant row offset,
    // v4 thread T/4 (x16 offset, and in LDS), v5 thread 4, v6 scratch address,
    // A v7-10 (8 rows' quants), B v11-14 (8 tokens' x), staging v15-26 (quants
    // QN, then x XN), header v27-30, xs v31, f16 sums v32-63 (row i, pair p:
    // 32 + 4 i + p), f32 sums v64-127 (token t, row i: 64 + 8 t + i).
    const int A = 7, B = 11, Q = 15, X = 15 + QN, H = 27, XSV = 31, TMP = 32, ACC = 64;
    // SGPRs: s[12:13] the block being loaded, s[14:15] its quants, s[16:17]
    // x16 and s[18:19] xs32 of the sub-block being loaded, s20 sub-blocks
    // loaded, s21 the sub-block in LDS, s23 E loaded, s24 E in LDS, s25-27 scratch.
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto vr = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    auto v = [&](int a) { return "v" + S(a); };
    int lab = 0;
    auto label = [&](const std::string& base) { return base + S(lab++); };
    k += ".kernel mm_q4k_t\n.wave32\n.user_sgprs 10\n.tgid_x\n.tgid_y\n.lds " + S(LDS) + "\n";
    o("v_lshrrev_b32 v6, 5, v0");
    o("v_readfirstlane_b32 s25, v6");                      // the wave
    o("v_and_b32 v7, 31, v0");                              // its lane
    o("v_and_b32 v8, 7, v7");
    o("s_and_b32 s26, s25, " + S(WPR - 1));
    o("s_lshl_b32 s26, s26, 3");
    o("v_add_nc_u32 v8, s26, v8");                          // rg = lane % 8 + 8 (wave % WPR)
    o("v_lshrrev_b32 v9, 3, v7");
    o("s_lshr_b32 s26, s25, " + S(LWPR));
    o("s_lshl_b32 s26, s26, 2");
    o("v_add_nc_u32 v9, s26, v9");                          // tg = lane / 8 + 4 (wave / WPR)
    o("v_lshlrev_b32 v1, 4, v8");
    o("v_lshlrev_b32 v2, 4, v9");
    // the quants each thread stages: T 128 row thread/2, bytes 16 (thread % 2) ..
    // of each 32 (the sub-block pair); T 64 row thread, all 32
    o("s_mul_i32 s26, s10, " + S(RM));
    if (T == 128) o("v_lshrrev_b32 v7, 1, v0");
    else o("v_mov_b32 v7, v0");
    o("v_add_nc_u32 v7, s26, v7");
    o("v_mul_u32_u24 v3, " + S(RB) + ", v7");
    if (T == 128) {
        o("v_and_b32 v8, 1, v0");
        o("v_lshl_add_u32 v3, v8, 4, v3");
    }
    o("v_lshlrev_b32 v4, " + S(T == 128 ? 5 : 4) + ", v0");
    o("v_lshlrev_b32 v5, 2, v0");
    o("s_mul_i32 s25, s11, " + S(NBK / SK * 144));
    o("s_add_u32 s12, s0, s25");
    o("s_addc_u32 s13, s1, 0");
    o("s_mul_i32 s25, s11, " + S(NSW * 64 * T));
    o("s_add_u32 s16, s2, s25");
    o("s_addc_u32 s17, s3, 0");
    o("s_mul_i32 s25, s11, " + S(NSW * 4 * T));
    o("s_add_u32 s18, s8, s25");
    o("s_addc_u32 s19, s9, 0");
    o("s_mov_b32 s20, 0");
    o("s_mov_b32 s21, 0");
    o("s_mov_b32 s103, 0x64006400");                        // f16 1024, 1024 (the quants' exponent trick)
    o("s_mov_b32 s104, 0xe400e400");                        // f16 -1024, -1024
    for (int i = 0; i < 64; i++) o("v_mov_b32 " + v(ACC + i) + ", 0");
    auto masked = [&](int n, const std::function<void()>& body) {   // threads < n only
        if (n >= 256) { body(); return; }
        o("v_cmp_gt_u32 vcc_lo, " + S(n) + ", v0");
        o("s_and_saveexec_b32 s27, vcc_lo");
        body();
        o("s_mov_b32 exec_lo, s27");
    };
    // Sub-block s20's loads into the staging registers, then the pointers on.
    auto issue = [&]() {
        const std::string noq = label("noq"), noh = label("noh"), nob = label("nob");
        o("s_and_b32 s25, s20, 7");                         // j, its place in the block
        o("s_bitcmp1_b32 s25, 0");
        o("s_cbranch_scc1 " + noq);                         // odd: the pair's quants are staged already
        o("s_lshl_b32 s26, s25, 4");
        o("s_add_u32 s26, s26, 16");
        o("s_add_u32 s14, s12, s26");
        o("s_addc_u32 s15, s13, 0");
        o("global_load_dwordx4 " + vr(Q, 4) + ", v3, s[14:15]");
        if (QN == 8) o("global_load_dwordx4 " + vr(Q + 4, 4) + ", v3, s[14:15] offset:16");
        k += noq + ":\n";
        o("s_cmp_lg_u32 s25, 0");
        o("s_cbranch_scc1 " + noh);                         // the block's header, for its scales
        masked(RM, [&]() {
            o("s_mul_i32 s26, s10, " + S(RM));
            o("v_add_nc_u32 v6, s26, v0");
            o("v_mul_u32_u24 v6, " + S(RB) + ", v6");
            o("global_load_dwordx4 " + vr(H, 4) + ", v6, s[12:13]");
        });
        k += noh + ":\n";
        o("global_load_dwordx4 " + vr(X, 4) + ", v4, s[16:17]");
        if (XN == 8) o("global_load_dwordx4 " + vr(X + 4, 4) + ", v4, s[16:17] offset:16");
        masked(T, [&]() { o("global_load_dword " + v(XSV) + ", v5, s[18:19]"); });
        o("s_load_dword s23, s[18:19], 0x0");
        o("s_add_u32 s16, s16, " + S(64 * T));
        o("s_addc_u32 s17, s17, 0");
        o("s_add_u32 s18, s18, " + S(4 * T));
        o("s_addc_u32 s19, s19, 0");
        o("s_cmp_lg_u32 s25, 7");
        o("s_cbranch_scc1 " + nob);
        o("s_add_u32 s12, s12, 144");
        o("s_addc_u32 s13, s13, 0");
        k += nob + ":\n";
        o("s_add_u32 s20, s20, 1");
    };
    // The staged sub-block s21 into LDS: quants (f16), x, xs, and at a block's
    // start its 8 sub-blocks' scales (d sc, dmin m; f32).
    auto write = [&]() {
        const std::string hi = label("whi"), done = label("wdone"), nosc = label("nosc");
        o("s_waitcnt vmcnt(0) lgkmcnt(0)");
        o("s_sext_i32_i8 s24, s23");
        o("s_and_b32 s25, s21, 7");
        if (T == 128) {   // row thread/2, columns from 16 (thread % 2)
            o("v_lshrrev_b32 v6, 1, v0");
            o("v_lshlrev_b32 v6, 1, v6");
            o("v_and_b32 v7, 1, v0");
            o("v_mul_u32_u24 v7, " + S(32 * RM) + ", v7");
            o("v_add_nc_u32 v6, v6, v7");
        } else {
            o("v_lshlrev_b32 v6, 1, v0");
        }
        for (int half = 0; half < 2; half++) {
            if (half == 0) {
                o("s_bitcmp1_b32 s25, 0");
                o("s_cbranch_scc1 " + hi);
            } else {
                k += hi + ":\n";
            }
            for (int d = 0; d < QN; d++) {
                const int u = A + 2 * (d % 4), w = u + 1;   // bytes 0, 2 / 1, 3
                if (half) {
                    o("v_lshrrev_b32 " + v(u) + ", 4, " + v(Q + d));
                    o("v_and_or_b32 " + v(u) + ", " + v(u) + ", 0x000f000f, s103");
                } else {
                    o("v_and_or_b32 " + v(u) + ", " + v(Q + d) + ", 0x000f000f, s103");
                }
                o("v_lshrrev_b32 " + v(w) + ", " + S(half ? 12 : 8) + ", " + v(Q + d));
                o("v_and_or_b32 " + v(w) + ", " + v(w) + ", 0x000f000f, s103");
                o("v_pk_add_f16 " + v(u) + ", " + v(u) + ", s104");
                o("v_pk_add_f16 " + v(w) + ", " + v(w) + ", s104");
                const int c = 4 * d;
                o("ds_write_b16 v6, " + v(u) + " offset:" + S(LW + c * 2 * RM));
                o("ds_write_b16 v6, " + v(w) + " offset:" + S(LW + (c + 1) * 2 * RM));
                o("ds_write_b16_d16_hi v6, " + v(u) + " offset:" + S(LW + (c + 2) * 2 * RM));
                o("ds_write_b16_d16_hi v6, " + v(w) + " offset:" + S(LW + (c + 3) * 2 * RM));
            }
            if (half == 0) o("s_branch " + done);
        }
        k += done + ":\n";
        o("ds_write_b128 v4, " + vr(X, 4) + " offset:" + S(LX));
        if (XN == 8) o("ds_write_b128 v4, " + vr(X + 4, 4) + " offset:" + S(LX + 16));
        masked(T, [&]() { o("ds_write_b32 v5, " + v(XSV) + " offset:" + S(LXS)); });
        o("s_cmp_lg_u32 s25, 0");
        o("s_cbranch_scc1 " + nosc);
        masked(RM, [&]() {
            // header: d, dmin (f16), then 12 bytes of 6-bit scales and minimums
            auto Hb = [&](int byte) { return v(H + 1 + byte / 4); };
            auto bit = [](int byte) { return 8 * (byte % 4); };
            const int D = A, DM = A + 1, SC = A + 2, MN = A + 3, U = A + 4;
            o("v_cvt_f32_f16 " + v(D) + ", " + v(H));
            o("v_lshrrev_b32 " + v(DM) + ", 16, " + v(H));
            o("v_cvt_f32_f16 " + v(DM) + ", " + v(DM));
            for (int j = 0; j < 8; j++) {
                const std::string sc = v(SC), m = v(MN);
                if (j < 4) {
                    o("v_bfe_u32 " + sc + ", " + Hb(j) + ", " + S(bit(j)) + ", 6");
                    o("v_bfe_u32 " + m + ", " + Hb(j + 4) + ", " + S(bit(j + 4)) + ", 6");
                } else {
                    o("v_bfe_u32 " + sc + ", " + Hb(j + 4) + ", " + S(bit(j + 4)) + ", 4");
                    o("v_bfe_u32 " + v(U) + ", " + Hb(j - 4) + ", " + S(bit(j - 4) + 6) + ", 2");
                    o("v_lshl_or_b32 " + sc + ", " + v(U) + ", 4, " + sc);
                    o("v_bfe_u32 " + m + ", " + Hb(j + 4) + ", " + S(bit(j + 4) + 4) + ", 4");
                    o("v_bfe_u32 " + v(U) + ", " + Hb(j) + ", " + S(bit(j) + 6) + ", 2");
                    o("v_lshl_or_b32 " + m + ", " + v(U) + ", 4, " + m);
                }
                o("v_cvt_f32_u32 " + sc + ", " + sc);
                o("v_cvt_f32_u32 " + m + ", " + m);
                o("v_mul_f32 " + sc + ", " + v(D) + ", " + sc);
                o("v_mul_f32 " + m + ", " + v(DM) + ", " + m);
                o("ds_write_b32 v5, " + sc + " offset:" + S(LSC + j * 4 * RM));
                o("ds_write_b32 v5, " + m + " offset:" + S(LMN + j * 4 * RM));
                o("s_waitcnt lgkmcnt(0)");   // (sc, m reused)
            }
        });
        k += nosc + ":\n";
        o("s_waitcnt lgkmcnt(0)");
        o("s_barrier");
    };
    issue();
    write();
    if (NSW > 1) issue();
    k += "stage:\n";
    // 32 k steps of the sub-block in LDS
    for (int kk = 0; kk < 32; kk++) {
        o("ds_read_b128 " + vr(A, 4) + ", v1 offset:" + S(LW + kk * 2 * RM));
        o("ds_read_b128 " + vr(B, 4) + ", v2 offset:" + S(LX + kk * 2 * T));
        o("s_waitcnt lgkmcnt(0)");
        for (int i = 0; i < 8; i++)
            for (int p = 0; p < 4; p++) {
                const std::string d = v(TMP + 4 * i + p), a = v(A + i / 2), b = v(B + p), h = S(i % 2);
                if (kk == 0) o("v_pk_mul_f16 " + d + ", " + a + ", " + b + " op_sel:[" + h + ",0] op_sel_hi:[" + h + ",1]");
                else o("v_pk_fma_f16 " + d + ", " + a + ", " + b + ", " + d + " op_sel:[" + h + ",0,0] op_sel_hi:[" + h + ",1,1]");
            }
    }
    // into f32: acc += d sc 2^E sum, then - dmin m xs
    o("s_and_b32 s25, s21, 7");
    o("s_mul_i32 s26, s25, " + S(4 * RM));
    o("v_lshlrev_b32 v6, 1, v1");
    o("v_add_nc_u32 v6, s26, v6");
    o("ds_read_b128 " + vr(A, 4) + ", v6 offset:" + S(LSC));
    o("ds_read_b128 " + vr(B, 4) + ", v6 offset:" + S(LSC + 16));
    o("s_waitcnt lgkmcnt(0)");
    for (int i = 0; i < 8; i++) o("v_ldexp_f32 " + v(A + i) + ", " + v(A + i) + ", s24");
    for (int i = 0; i < 8; i++)
        for (int t = 0; t < 8; t++) {
            const std::string a = v(ACC + 8 * t + i);
            o("v_fma_mix_f32 " + a + ", " + v(A + i) + ", " + v(TMP + 4 * i + t / 2) + ", " + a + " op_sel:[0," + S(t % 2) + ",0] op_sel_hi:[0,1,0]");
        }
    o("ds_read_b128 " + vr(A, 4) + ", v6 offset:" + S(LMN));
    o("ds_read_b128 " + vr(B, 4) + ", v6 offset:" + S(LMN + 16));
    o("v_lshlrev_b32 " + v(TMP + 4) + ", 1, v2");
    o("ds_read_b128 " + vr(TMP, 4) + ", " + v(TMP + 4) + " offset:" + S(LXS));
    o("ds_read_b128 " + vr(TMP + 4, 4) + ", " + v(TMP + 4) + " offset:" + S(LXS + 16));
    o("s_waitcnt lgkmcnt(0)");
    for (int i = 0; i < 8; i++)
        for (int t = 0; t < 8; t++) {
            const std::string a = v(ACC + 8 * t + i);
            o("v_fma_f32 " + a + ", -" + v(A + i) + ", " + v(TMP + t) + ", " + a);
        }
    o("s_add_u32 s21, s21, 1");
    o("s_cmp_eq_u32 s21, " + S(NSW));
    o("s_cbranch_scc1 stages_done");
    o("s_barrier");                                         // every wave done with the LDS
    write();
    o("s_add_u32 s25, s21, 1");
    o("s_cmp_ge_u32 s25, " + S(NSW));
    o("s_cbranch_scc1 stage");
    issue();
    o("s_branch stage");
    k += "stages_done:\n";
    // y[t][row]: the thread's 8 rows for each of its 8 tokens
    o("v_lshrrev_b32 v7, 4, v2");
    o("v_lshlrev_b32 v7, 3, v7");                           // its first token
    o("v_mul_lo_u32 v7, s6, v7");
    o("v_lshrrev_b32 v8, 1, v1");                           // rg 8
    o("v_add_nc_u32 v7, v7, v8");
    o("s_mul_i32 s25, s10, " + S(RM));
    o("v_add_nc_u32 v7, s25, v7");
    o("v_lshlrev_b32 v7, 2, v7");
    o("s_mul_i32 s25, s6, " + S(T * 4));
    o("s_mul_i32 s25, s25, s11");
    o("s_add_u32 s26, s4, s25");
    o("s_addc_u32 s27, s5, 0");
    o("s_lshl_b32 s25, s6, 2");
    for (int t = 0; t < 8; t++) {
        if (t) {
            o("s_add_u32 s26, s26, s25");
            o("s_addc_u32 s27, s27, 0");
        }
        o("global_store_dwordx4 v7, " + vr(ACC + 8 * t, 4) + ", s[26:27]");
        o("global_store_dwordx4 v7, " + vr(ACC + 8 * t + 4, 4) + ", s[26:27] offset:16");
    }
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
