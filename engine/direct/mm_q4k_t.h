// mm_t: the prompt mat-mat as a tiled GEMM, for Q4_K and Q6_K weights. Each
// workgroup of 256 takes 128 rows x 128 tokens, each thread 8 rows x 8
// tokens, the weights and x of one 32-column sub-block at a time staged in
// LDS (the quants as exact f16: Q4_K 0..15, Q6_K -32..31), so every weight
// byte is read from memory once per 128 tokens instead of once per 16
// (rounds 72-77: the 16-token kernels stall on their weight and x streams at
// ~658 tok/s; round 78: this kernel is 17-26% faster a token).
//
// Math as mm_q4k_h / mm_q6k_h: f16 sums of q x16 (|x16| < 16) over each
// 32 columns (Q4_K) or 16 (Q6_K, one scale a 16), then into f32:
// Q4_K acc += d sc 2^E sum - dmin m xs; Q6_K acc += d sc 2^E sum.
// Each k step: one ds_read_b128 of 8 rows' quants (f16) and one of 8 tokens'
// x (4 pairs), 32 v_pk_fma_f16 (the row's quant broadcast by op_sel).
// A sub-block's loads go out a whole sub-block ahead (registers), and are
// written to LDS between two barriers.
//
// Arguments: s[0:1] W, s[2:3] x16 (K x TT f16, a column's TT tokens
// together), s[4:5] y (TT x R f32; with SK, SK of them R TT floats apart,
// to be summed), s6 R, s7 TT (tokens in the pass, a multiple of 128), s[8:9]
// xs32 (K/32 x TT, the sub-block sums, E in the low byte; Q6_K only uses E). Grid: R/128 x TT/128 x SK workgroups of 256.
#pragma once
#include <functional>
#include <string>

namespace ts5::direct {

inline std::string gen_mm_t(int K, bool q6, int SK = 1) {
    const int NS = K / 32, NBK = K / 256, BB = q6 ? 210 : 144;
    if (K % 256 || SK < 1 || NBK % SK) return "";
    const int NSW = NS / SK;                               // sub-blocks a workgroup (whole blocks)
    const int RM = 128, RB = NBK * BB;
    const int LW = 0, LX = 8192, LSC = 16384, LMN = LSC + 32 * RM, LXS = LMN + 32 * RM;
    const int LDS = q6 ? LSC + 8 * RM : LXS + 4 * 128;
    // VGPRs: v0 thread, v1 W read (rg 16), v2 x read (tg 16), v3 quant row
    // offset, v4 x16 offset, v5 thread 4, v6 scratch address, A v7-10 (8 rows'
    // quants), B v11-14 (8 tokens' x), staging v15-26 (quants 4 / ql 4 qh 4,
    // then x 8), header v27-30 (Q6_K: v31, its two scales and d; round 79:
    // at v27 it overlapped the staged x), xs v31 (Q4_K),
    // f16 sums v32-63 (row i, pair p: 32 + 4 i + p), f32 sums v64-127
    // (token t, row i: 64 + 8 t + i).
    const int A = 7, B = 11, Q = 15, QH = 19, X = 15 + (q6 ? 8 : 4), H = q6 ? 31 : 27, XSV = 31, TMP = 32, ACC = 64;
    // SGPRs: s10 row tile, s11 token tile, s12 SK slice; s[14:15] the block
    // being loaded, s[16:17] s[32:33] s[34:35] scratch addresses, s[18:19]
    // x16 and s[20:21] xs of the sub-block being loaded, s22 sub-blocks
    // loaded, s23 the sub-block in LDS, s24 E loaded, s28 E in LDS, s29 x16
    // bytes a sub-block, s30 xs bytes a sub-block, s25-27 scratch, s31 exec.
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto vr = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    auto v = [&](int a) { return "v" + S(a); };
    int lab = 0;
    auto label = [&](const std::string& base) { return base + S(lab++); };
    k += std::string(".kernel ") + (q6 ? "mm_q6k_t" : "mm_q4k_t") + "\n.wave32\n.user_sgprs 10\n.tgid_x\n.tgid_y\n.tgid_z\n.lds " + S(LDS) + "\n";
    o("v_lshrrev_b32 v6, 5, v0");
    o("v_readfirstlane_b32 s25, v6");                      // the wave
    o("v_and_b32 v7, 31, v0");                              // its lane
    o("v_and_b32 v8, 7, v7");
    o("s_and_b32 s26, s25, 1");
    o("s_lshl_b32 s26, s26, 3");
    o("v_add_nc_u32 v8, s26, v8");                          // rg = lane % 8 + 8 (wave % 2)
    o("v_lshrrev_b32 v9, 3, v7");
    o("s_lshr_b32 s26, s25, 1");
    o("s_lshl_b32 s26, s26, 2");
    o("v_add_nc_u32 v9, s26, v9");                          // tg = lane / 8 + 4 (wave / 2)
    o("v_lshlrev_b32 v1, 4, v8");
    o("v_lshlrev_b32 v2, 4, v9");
    // the quants each thread stages: row thread/2, bytes (Q6_K: values) 16 (thread % 2) ..
    o("s_mul_i32 s26, s10, " + S(RM));
    o("v_lshrrev_b32 v7, 1, v0");
    o("v_add_nc_u32 v7, s26, v7");
    o("v_mul_u32_u24 v3, " + S(RB) + ", v7");
    o("v_and_b32 v8, 1, v0");
    o("v_lshl_add_u32 v3, v8, 4, v3");
    // x16: thread column thread/8, tokens 16 (thread % 8) .. of the tile
    o("s_lshl_b32 s29, s7, 1");                             // TT 2: bytes a column
    o("v_lshrrev_b32 v4, 3, v0");
    o("v_mul_lo_u32 v4, s29, v4");
    o("v_and_b32 v8, 7, v0");
    o("v_lshl_add_u32 v4, v8, 5, v4");
    o("v_lshlrev_b32 v5, 2, v0");
    o("s_lshl_b32 s29, s7, 6");                             // x16 bytes a sub-block: 32 columns
    o("s_lshl_b32 s30, s7, 2");                             // xs32 bytes a sub-block
    o("s_mul_i32 s25, s12, " + S(NBK / SK * BB));
    o("s_add_u32 s14, s0, s25");
    o("s_addc_u32 s15, s1, 0");
    o("s_mul_i32 s25, s12, " + S(NSW));
    o("s_mul_i32 s26, s25, s29");
    o("s_lshl_b32 s27, s11, 8");                            // the token tile: 128 tokens, 2 bytes
    o("s_add_u32 s26, s26, s27");
    o("s_add_u32 s18, s2, s26");
    o("s_addc_u32 s19, s3, 0");
    o("s_mul_i32 s26, s25, s30");
    o("s_lshl_b32 s27, s11, 9");                            // 128 tokens, 4 bytes
    o("s_add_u32 s26, s26, s27");
    o("s_add_u32 s20, s8, s26");
    o("s_addc_u32 s21, s9, 0");
    o("s_mov_b32 s22, 0");
    o("s_mov_b32 s23, 0");
    o("s_mov_b32 s103, 0x64006400");                        // f16 1024, 1024 (the quants' exponent trick)
    o(q6 ? "s_mov_b32 s104, 0xe420e420" : "s_mov_b32 s104, 0xe400e400");   // f16 -1056 (Q6_K: q - 32) / -1024
    for (int i = 0; i < 64; i++) o("v_mov_b32 " + v(ACC + i) + ", 0");
    auto masked = [&](int n, const std::function<void()>& body) {   // threads < n only
        o("v_cmp_gt_u32 vcc_lo, " + S(n) + ", v0");
        o("s_and_saveexec_b32 s31, vcc_lo");
        body();
        o("s_mov_b32 exec_lo, s31");
    };
    auto row_offset = [&]() {   // v6: row thread's offset (threads < 128)
        o("s_mul_i32 s26, s10, " + S(RM));
        o("v_add_nc_u32 v6, s26, v0");
        o("v_mul_u32_u24 v6, " + S(RB) + ", v6");
    };
    // Sub-block s22's loads into the staging registers, then the pointers on.
    auto issue = [&]() {
        const std::string noq = label("noq"), noh = label("noh"), nob = label("nob");
        o("s_and_b32 s25, s22, 7");                         // j, its place in the block
        if (!q6) {
            o("s_bitcmp1_b32 s25, 0");
            o("s_cbranch_scc1 " + noq);                     // odd: the pair's quants are staged already
            o("s_lshl_b32 s26, s25, 4");
            o("s_add_u32 s26, s26, 16");
            o("s_add_u32 s16, s14, s26");
            o("s_addc_u32 s17, s15, 0");
            o("global_load_dwordx4 " + vr(Q, 4) + ", v3, s[16:17]");
            k += noq + ":\n";
            o("s_cmp_lg_u32 s25, 0");
            o("s_cbranch_scc1 " + noh);                     // the block's header, for its scales
            masked(RM, [&]() {
                row_offset();
                o("global_load_dwordx4 " + vr(H, 4) + ", v6, s[14:15]");
            });
            k += noh + ":\n";
        } else {
            // half n = j / 4, quarter m = j % 4: ql at 64 n + 32 (m & 1), qh at
            // 128 + 32 n, the two scales at 192 + 8 n + 2 m, d at 208. Odd
            // blocks start 2 bytes into a dword: 2-byte loads throughout.
            o("s_lshr_b32 s26, s25, 2");                    // n
            o("s_and_b32 s27, s25, 1");
            o("s_lshl_b32 s27, s27, 5");
            o("s_lshl_b32 s26, s26, 6");
            o("s_add_u32 s27, s27, s26");                   // 64 n + 32 (m & 1)
            o("s_add_u32 s16, s14, s27");
            o("s_addc_u32 s17, s15, 0");
            o("s_lshr_b32 s26, s26, 1");                    // 32 n
            o("s_add_u32 s27, s26, 128");
            o("s_add_u32 s32, s14, s27");
            o("s_addc_u32 s33, s15, 0");
            for (int i = 0; i < 8; i++) {
                o(std::string(i % 2 ? "global_load_short_d16_hi " : "global_load_short_d16 ") + v(Q + i / 2) + ", v3, s[16:17] offset:" + S(2 * i));
                o(std::string(i % 2 ? "global_load_short_d16_hi " : "global_load_short_d16 ") + v(QH + i / 2) + ", v3, s[32:33] offset:" + S(2 * i));
            }
            o("s_lshr_b32 s26, s26, 2");                    // 8 n
            o("s_and_b32 s27, s25, 3");
            o("s_lshl_b32 s27, s27, 1");
            o("s_add_u32 s27, s27, s26");
            o("s_add_u32 s27, s27, 192");
            o("s_add_u32 s34, s14, s27");
            o("s_addc_u32 s35, s15, 0");
            masked(RM, [&]() {
                row_offset();
                o("global_load_short_d16 " + v(H) + ", v6, s[34:35]");
                o("global_load_short_d16_hi " + v(H) + ", v6, s[14:15] offset:208");
            });
        }
        o("global_load_dwordx4 " + vr(X, 4) + ", v4, s[18:19]");
        o("global_load_dwordx4 " + vr(X + 4, 4) + ", v4, s[18:19] offset:16");
        if (!q6) masked(128, [&]() { o("global_load_dword " + v(XSV) + ", v5, s[20:21]"); });
        o("s_load_dword s24, s[20:21], 0x0");
        o("s_add_u32 s18, s18, s29");
        o("s_addc_u32 s19, s19, 0");
        o("s_add_u32 s20, s20, s30");
        o("s_addc_u32 s21, s21, 0");
        o("s_cmp_lg_u32 s25, 7");
        o("s_cbranch_scc1 " + nob);
        o("s_add_u32 s14, s14, " + S(BB));
        o("s_addc_u32 s15, s15, 0");
        k += nob + ":\n";
        o("s_add_u32 s22, s22, 1");
    };
    // The staged sub-block s23 into LDS: quants (f16), x, (Q4_K) xs, and
    // the scales: Q4_K at a block's start all 8 sub-blocks' (d sc, dmin m);
    // Q6_K this sub-block's two (d sc).
    auto write = [&]() {
        const std::string hi = label("whi"), done = label("wdone"), nosc = label("nosc");
        o("s_waitcnt vmcnt(0) lgkmcnt(0)");
        o("s_sext_i32_i8 s28, s24");
        o("s_and_b32 s25, s23, 7");
        o("v_lshrrev_b32 v6, 1, v0");                       // row thread/2, columns from 16 (thread % 2)
        o("v_lshlrev_b32 v6, 1, v6");
        o("v_and_b32 v7, 1, v0");
        o("v_mul_u32_u24 v7, " + S(32 * RM) + ", v7");
        o("v_add_nc_u32 v6, v6, v7");
        auto put = [&](int d, int u, int w) {               // columns 4 d .. 4 d + 3: u (bytes 0, 2), w (1, 3)
            const int c = 4 * d;
            o("ds_write_b16 v6, " + v(u) + " offset:" + S(LW + c * 2 * RM));
            o("ds_write_b16 v6, " + v(w) + " offset:" + S(LW + (c + 1) * 2 * RM));
            o("ds_write_b16_d16_hi v6, " + v(u) + " offset:" + S(LW + (c + 2) * 2 * RM));
            o("ds_write_b16_d16_hi v6, " + v(w) + " offset:" + S(LW + (c + 3) * 2 * RM));
        };
        if (!q6) {
            for (int half = 0; half < 2; half++) {
                if (half == 0) {
                    o("s_bitcmp1_b32 s25, 0");
                    o("s_cbranch_scc1 " + hi);
                } else {
                    k += hi + ":\n";
                }
                for (int d = 0; d < 4; d++) {
                    const int u = A + 2 * d, w = u + 1;
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
                    put(d, u, w);
                }
                if (half == 0) o("s_branch " + done);
            }
            k += done + ":\n";
        } else {
            // q = (ql >> 4 (m >> 1)) & 15 | ((qh >> 2 m) & 3) << 4, as f16 q - 32
            o("s_and_b32 s26, s25, 3");                     // m
            o("s_lshl_b32 s27, s26, 1");                    // 2 m
            o("s_lshr_b32 s26, s26, 1");
            o("s_lshl_b32 s26, s26, 2");                    // 4 (m >> 1)
            for (int d = 0; d < 4; d++) {
                const int u = A + 2 * d, w = u + 1;
                o("v_lshrrev_b32 " + v(u) + ", s26, " + v(Q + d));
                o("v_and_b32 " + v(u) + ", 0x0f0f0f0f, " + v(u));
                o("v_lshrrev_b32 " + v(w) + ", s27, " + v(QH + d));
                o("v_and_b32 " + v(w) + ", 0x03030303, " + v(w));
                o("v_lshl_or_b32 " + v(u) + ", " + v(w) + ", 4, " + v(u));   // the 4 values' bytes
                o("v_lshrrev_b32 " + v(w) + ", 8, " + v(u));
                o("v_and_or_b32 " + v(u) + ", " + v(u) + ", 0x00ff00ff, s103");
                o("v_and_or_b32 " + v(w) + ", " + v(w) + ", 0x00ff00ff, s103");
                o("v_pk_add_f16 " + v(u) + ", " + v(u) + ", s104");
                o("v_pk_add_f16 " + v(w) + ", " + v(w) + ", s104");
                put(d, u, w);
            }
        }
        o("v_lshlrev_b32 v7, 5, v0");                       // x: thread 32 bytes, as in memory
        o("ds_write_b128 v7, " + vr(X, 4) + " offset:" + S(LX));
        o("ds_write_b128 v7, " + vr(X + 4, 4) + " offset:" + S(LX + 16));
        if (!q6) {
            masked(128, [&]() { o("ds_write_b32 v5, " + v(XSV) + " offset:" + S(LXS)); });
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
        } else {
            masked(RM, [&]() {   // d sc for values 0-15 and 16-31 of the sub-block
                o("v_lshrrev_b32 " + v(A) + ", 16, " + v(H));
                o("v_cvt_f32_f16 " + v(A) + ", " + v(A));
                o("v_bfe_i32 " + v(A + 1) + ", " + v(H) + ", 0, 8");
                o("v_bfe_i32 " + v(A + 2) + ", " + v(H) + ", 8, 8");
                o("v_cvt_f32_i32 " + v(A + 1) + ", " + v(A + 1));
                o("v_cvt_f32_i32 " + v(A + 2) + ", " + v(A + 2));
                o("v_mul_f32 " + v(A + 1) + ", " + v(A) + ", " + v(A + 1));
                o("v_mul_f32 " + v(A + 2) + ", " + v(A) + ", " + v(A + 2));
                o("ds_write_b32 v5, " + v(A + 1) + " offset:" + S(LSC));
                o("ds_write_b32 v5, " + v(A + 2) + " offset:" + S(LSC + 4 * RM));
            });
        }
        o("s_waitcnt lgkmcnt(0)");
        o("s_barrier");
    };
    // the f16 sums into f32 with the scales at LDS sc (8 rows from rg 32)
    auto fold = [&](const std::string& sc_off_sgpr, int sc_base) {
        o("v_lshlrev_b32 v6, 1, v1");
        if (!sc_off_sgpr.empty()) o("v_add_nc_u32 v6, " + sc_off_sgpr + ", v6");
        o("ds_read_b128 " + vr(A, 4) + ", v6 offset:" + S(sc_base));
        o("ds_read_b128 " + vr(B, 4) + ", v6 offset:" + S(sc_base + 16));
        o("s_waitcnt lgkmcnt(0)");
        for (int i = 0; i < 8; i++) o("v_ldexp_f32 " + v(A + i) + ", " + v(A + i) + ", s28");
        for (int i = 0; i < 8; i++)
            for (int t = 0; t < 8; t++) {
                const std::string a = v(ACC + 8 * t + i);
                o("v_fma_mix_f32 " + a + ", " + v(A + i) + ", " + v(TMP + 4 * i + t / 2) + ", " + a + " op_sel:[0," + S(t % 2) + ",0] op_sel_hi:[0,1,0]");
            }
    };
    issue();
    write();
    if (NSW > 1) issue();
    k += "stage:\n";
    // 32 k steps of the sub-block in LDS (Q6_K: folded after each 16)
    for (int kk = 0; kk < 32; kk++) {
        o("ds_read_b128 " + vr(A, 4) + ", v1 offset:" + S(LW + kk * 2 * RM));
        o("ds_read_b128 " + vr(B, 4) + ", v2 offset:" + S(LX + kk * 2 * 128));
        o("s_waitcnt lgkmcnt(0)");
        const bool first = kk == 0 || (q6 && kk == 16);
        for (int i = 0; i < 8; i++)
            for (int p = 0; p < 4; p++) {
                const std::string d = v(TMP + 4 * i + p), a = v(A + i / 2), b = v(B + p), h = S(i % 2);
                if (first) o("v_pk_mul_f16 " + d + ", " + a + ", " + b + " op_sel:[" + h + ",0] op_sel_hi:[" + h + ",1]");
                else o("v_pk_fma_f16 " + d + ", " + a + ", " + b + ", " + d + " op_sel:[" + h + ",0,0] op_sel_hi:[" + h + ",1,1]");
            }
        if (q6 && kk == 15) fold("", LSC);
    }
    if (q6) {
        fold("", LSC + 4 * RM);
    } else {
        // acc += d sc 2^E sum, then - dmin m xs
        o("s_and_b32 s25, s23, 7");
        o("s_mul_i32 s26, s25, " + S(4 * RM));
        fold("s26", LSC);
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
    }
    o("s_add_u32 s23, s23, 1");
    o("s_cmp_eq_u32 s23, " + S(NSW));
    o("s_cbranch_scc1 stages_done");
    o("s_barrier");                                         // every wave done with the LDS
    write();
    o("s_add_u32 s25, s23, 1");
    o("s_cmp_ge_u32 s25, " + S(NSW));
    o("s_cbranch_scc1 stage");
    issue();
    o("s_branch stage");
    k += "stages_done:\n";
    // y[token][row]: the thread's 8 rows for each of its 8 tokens
    o("v_lshrrev_b32 v7, 4, v2");
    o("v_lshlrev_b32 v7, 3, v7");                           // its first token in the tile
    o("s_lshl_b32 s25, s11, 7");
    o("v_add_nc_u32 v7, s25, v7");                          // .. in the pass
    o("v_mul_lo_u32 v7, s6, v7");
    o("v_lshrrev_b32 v8, 1, v1");                           // rg 8
    o("v_add_nc_u32 v7, v7, v8");
    o("s_mul_i32 s25, s10, " + S(RM));
    o("v_add_nc_u32 v7, s25, v7");
    o("v_lshlrev_b32 v7, 2, v7");
    o("s_mul_i32 s25, s6, s7");
    o("s_lshl_b32 s25, s25, 2");
    o("s_mul_i32 s25, s25, s12");
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

// Kept for the test's grid: rows a workgroup.
inline int mm_t_rows() { return 128; }

}  // namespace ts5::direct
