// Kernels for reading a prompt T tokens at a time (the batched mat-vecs are
// mm_q4k.h and mm_q6k.h; the per-token steps reuse ops.h and attn.h).
#pragma once
#include <string>

namespace ts5::direct {

// to_xT: T tokens' rows of K values (Y, T x K, f32) to the prompt kernels'
// input: xT (K x T, a column's T values together), xs32 (K/32 x T, the sums
// of each 32 columns, for Q4_K) and xs16 (K/16 x T, for Q6_K).
// A workgroup of 256 takes C = 1024 / T columns of every token: each thread
// 4 columns of one token (rows read coalesced), the sums by DPP across the
// threads of a token, then through LDS to xT, 4 tokens a thread (one
// contiguous 4 KiB a workgroup). Round 43: the first version (256 columns
// a workgroup, 16-48 workgroups, sums by 16-32 serial LDS reads) took 24%
// of a prompt's time.
// Also x16 (K x T/2 dwords: the f16 kernels' x, a column's tokens in
// pairs, low half the even one), scaled by 2^-E for each 32 columns (one E
// for all T tokens) so that |x16| < 16, with E (int8) in the low byte of xs16 and xs32
// (they lose their last 8 bits, 2^-15 relative).
// Arguments: s[0:1] Y, s[2:3] xT, s[4:5] xs32, s[6:7] xs16, s[8:9] x16;
// s10 workgroup id. Grid: K / C workgroups of NT. T is 8 or 16.
// NT 128 (T 16: C 32, a workgroup one exponent chunk): twice the workgroups
// (round 62: K 4096 is only 64 of 256, ~13 us a launch on 40 CUs).
inline std::string gen_to_xT(int K, int T, int NT = 256) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int LT = T == 16 ? 4 : 3, C = 4 * NT / T, TPT = NT / T;   // columns a workgroup, threads a token
    const int LNT = NT == 128 ? 7 : 8, LTPT = LNT - LT;
    if ((NT != 128 && NT != 256) || C < 32) return "";
    auto lg = [](int v) { int l = 0; while ((1 << l) < v) l++; return l; };
    k += ".kernel to_xT\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds 8256\n";
    o("v_lshrrev_b32 v1, " + S(LTPT) + ", v0");          // t
    o("v_and_b32 v2, " + S(TPT - 1) + ", v0");           // q: columns 4q .. 4q + 3 of the workgroup's
    o("v_mul_u32_u24 v3, " + S(K * 4) + ", v1");
    o("v_lshl_add_u32 v3, v2, 4, v3");
    o("s_mul_i32 s18, s10, " + S(C * 4));
    o("s_add_u32 s14, s0, s18");
    o("s_addc_u32 s15, s1, 0");
    o("global_load_dwordx4 v[8:11], v3, s[14:15]");
    // the C/32 chunk maxima in LDS (8192 ..) start at 0
    o("v_cmp_gt_u32 vcc_lo, " + S(C / 32) + ", v0");
    o("s_and_saveexec_b32 s16, vcc_lo");
    o("v_lshlrev_b32 v5, 2, v0");
    o("v_mov_b32 v6, 0");
    o("ds_write_b32 v5, v6 offset:8192");
    o("s_mov_b32 exec_lo, s16");
    o("v_mul_u32_u24 v4, " + S(C * 4) + ", v1");         // LDS [t][C]
    o("v_lshl_add_u32 v4, v2, 4, v4");
    o("s_waitcnt vmcnt(0)");
    o("ds_write_b128 v4, v[8:11]");
    // sums: 4 columns, then 4 threads (16 columns), then 8 (32)
    o("v_add_f32 v12, v8, v9");
    o("v_add_f32 v13, v10, v11");
    o("v_add_f32 v12, v12, v13");
    o("v_nop");
    o("v_nop");
    o("v_add_f32_dpp v12, v12, v12 row_xmask:1 row_mask:0xf bank_mask:0xf");
    o("v_nop");
    o("v_nop");
    o("v_add_f32_dpp v12, v12, v12 row_xmask:2 row_mask:0xf bank_mask:0xf");
    o("v_nop");
    o("v_nop");
    o("v_add_f32_dpp v13, v12, v12 row_xmask:4 row_mask:0xf bank_mask:0xf");
    // the 32 columns' max |x| (of all the tokens): x16 = x 2^-E with E = its
    // exponent - 3, so |x16| < 16; E into xs16's and xs32's low byte
    o("v_max3_f32 v14, |v8|, |v9|, |v10|");
    o("v_max3_f32 v14, v14, |v11|, v14");
    for (int m : {1, 2, 4}) {
        o("v_nop");
        o("v_nop");
        o("v_max_f32_dpp v14, v14, v14 row_xmask:" + S(m) + " row_mask:0xf bank_mask:0xf");
    }
    // ... and over all T tokens (LDS max: order-free, so deterministic): one
    // E a chunk for every token, so the prompt kernels fold 2^E into the
    // weights' scale once a sub-block instead of once a token (round 62)
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_lshrrev_b32 v5, 3, v2");
    o("v_lshlrev_b32 v5, 2, v5");
    o("ds_max_u32 v5, v14 offset:8192");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("ds_read_b32 v14, v5 offset:8192");
    o("s_waitcnt lgkmcnt(0)");
    o("v_lshrrev_b32 v15, 23, v14");
    o("v_add_nc_u32 v15, 0xffffff7e, v15");             // E = biased exponent - 130 (2^E <= max/8)
    o("v_max_i32 v15, 0xffffff9c, v15");                // at least -100 (zero or tiny: fits a byte)
    o("v_sub_nc_u32 v16, 0, v15");
    for (int i = 0; i < 4; i++) o("v_ldexp_f32 v" + S(18 + i) + ", v" + S(8 + i) + ", v16");
    o("ds_write_b128 v4, v[18:21] offset:4096");
    o("v_and_b32 v15, 0xff, v15");
    o("v_and_b32 v12, 0xffffff00, v12");
    o("v_or_b32 v12, v12, v15");
    o("v_and_b32 v13, 0xffffff00, v13");
    o("v_or_b32 v13, v13, v15");
    // xs16: lanes q % 4 == 0, at (wg C/16 + q/4) T + t
    o("s_mul_i32 s18, s10, " + S(C / 16 * T * 4));
    o("v_lshrrev_b32 v5, 2, v2");
    o("v_mul_u32_u24 v5, " + S(T * 4) + ", v5");
    o("v_lshl_add_u32 v5, v1, 2, v5");
    o("v_add_nc_u32 v5, s18, v5");
    o("v_and_b32 v6, 3, v2");
    o("v_cmp_eq_u32 vcc_lo, 0, v6");
    o("s_and_saveexec_b32 s16, vcc_lo");
    o("global_store_dword v5, v12, s[6:7]");
    o("s_mov_b32 exec_lo, s16");
    // xs32: lanes q % 8 == 0, at (wg C/32 + q/8) T + t
    o("s_mul_i32 s18, s10, " + S(C / 32 * T * 4));
    o("v_lshrrev_b32 v5, 3, v2");
    o("v_mul_u32_u24 v5, " + S(T * 4) + ", v5");
    o("v_lshl_add_u32 v5, v1, 2, v5");
    o("v_add_nc_u32 v5, s18, v5");
    o("v_and_b32 v6, 7, v2");
    o("v_cmp_eq_u32 vcc_lo, 0, v6");
    o("s_and_saveexec_b32 s16, vcc_lo");
    o("global_store_dword v5, v13, s[4:5]");
    o("s_mov_b32 exec_lo, s16");
    // xT: thread j takes column c = j / (T/4), tokens t0 = 4 (j % (T/4)) .. + 3
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_lshrrev_b32 v1, " + S(LT - 2) + ", v0");        // c
    o("v_and_b32 v2, " + S(T / 4 - 1) + ", v0");
    o("v_mul_u32_u24 v2, " + S(4 * C * 4) + ", v2");     // t0 C 4
    o("v_lshl_add_u32 v2, v1, 2, v2");
    for (int i = 0; i < 4; i++) o("ds_read_b32 v" + S(8 + i) + ", v2 offset:" + S(i * C * 4));
    for (int i = 0; i < 4; i++) o("ds_read_b32 v" + S(14 + i) + ", v2 offset:" + S(4096 + i * C * 4));
    o("s_lshl_b32 s18, s10, " + S(lg(C * T * 4)));
    o("s_add_u32 s12, s2, s18");
    o("s_addc_u32 s13, s3, 0");
    o("v_lshlrev_b32 v3, 4, v0");
    o("s_waitcnt lgkmcnt(0)");
    o("global_store_dwordx4 v3, v[8:11], s[12:13]");
    o("v_cvt_pkrtz_f16_f32 v12, v14, v15");
    o("v_cvt_pkrtz_f16_f32 v13, v16, v17");
    o("s_lshl_b32 s17, s10, " + S(lg(C * T * 2)));
    o("s_add_u32 s12, s8, s17");
    o("s_addc_u32 s13, s9, 0");
    o("v_lshlrev_b32 v3, 3, v0");
    o("global_store_dwordx2 v3, v[12:13], s[12:13]");
    o("s_endpgm");
    return k;
}

// to_xT2: the same outputs as to_xT with no LDS and no barriers (round 67:
// the cross-token exponent made to_xT 7.0 / 16.0 us at K 4096 / 12288, two
// barriers on 64-192 workgroups). Each thread owns one column for all T
// tokens (T loads of 4 bytes, coalesced across the wave), so the max over
// tokens is its own and a wave is exactly one 32-column exponent chunk: the
// max and the 16- and 32-column sums by DPP and permlanex16. xT (a column's
// T values) and x16 (its T/2 dwords) are contiguous a thread; xs16 / xs32
// by the chunk's first lane (16 / 32 consecutive dwords). WXT 0 skips xT
// (the f16 prompt kernels never read it).
// Arguments as to_xT. Grid: K / 256 workgroups of 256. T is 8 or 16.
inline std::string gen_to_xT2(int K, int T, bool WXT = true) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto lg = [](int v) { int l = 0; while ((1 << l) < v) l++; return l; };
    if (T != 8 && T != 16) return "";
    const int X = 8, Y = X + T, M = Y + T, SM = M + 2, E = SM + 2;   // x, scaled / sums, max, temps, exponent
    k += ".kernel to_xT\n.wave32\n.user_sgprs 10\n.tgid_x\n";
    o("s_mov_b32 s86, 0x76543210");
    o("s_mov_b32 s87, 0xfedcba98");
    o("v_lshlrev_b32 v3, 2, v0");
    o("s_lshl_b32 s18, s10, 10");                       // 256 columns, 4 bytes
    o("s_add_u32 s14, s0, s18");
    o("s_addc_u32 s15, s1, 0");
    for (int t = 0; t < T; t++) {
        o("global_load_dword v" + S(X + t) + ", v3, s[14:15]");
        if (t + 1 < T) {
            o("s_add_u32 s14, s14, " + S(K * 4));
            o("s_addc_u32 s15, s15, 0");
        }
    }
    o("s_waitcnt vmcnt(0)");
    // max |x| over the tokens, then the 32 columns of the wave
    o("v_max3_f32 v" + S(M) + ", |v" + S(X) + "|, |v" + S(X + 1) + "|, 0");
    for (int t = 2; t < T; t += 2)
        o("v_max3_f32 v" + S(M) + ", v" + S(M) + ", |v" + S(X + t) + "|, |v" + S(X + t + 1) + "|");
    auto red = [&](const std::string& op, int r) {   // v r over 16 lanes (xmask), into every lane of the row
        for (int m : {1, 2, 4, 8}) {
            o("v_nop");
            o("v_nop");
            o("v_" + op + "_f32_dpp v" + S(r) + ", v" + S(r) + ", v" + S(r) + " row_xmask:" + S(m) + " row_mask:0xf bank_mask:0xf");
        }
    };
    red("max", M);
    o("v_permlanex16_b32 v" + S(M + 1) + ", v" + S(M) + ", s86, s87");
    o("v_max_f32 v" + S(M) + ", v" + S(M) + ", v" + S(M + 1));
    o("v_lshrrev_b32 v" + S(E) + ", 23, v" + S(M));
    o("v_add_nc_u32 v" + S(E) + ", 0xffffff7e, v" + S(E));   // E = biased exponent - 130
    o("v_max_i32 v" + S(E) + ", 0xffffff9c, v" + S(E));      // at least -100
    o("v_sub_nc_u32 v" + S(E + 1) + ", 0, v" + S(E));
    o("v_and_b32 v" + S(E) + ", 0xff, v" + S(E));
    // x16: x 2^-E, token pairs as f16 (low half the even token)
    for (int t = 0; t < T; t++) o("v_ldexp_f32 v" + S(Y + t) + ", v" + S(X + t) + ", v" + S(E + 1));
    for (int p = 0; p < T / 2; p++) o("v_cvt_pkrtz_f16_f32 v" + S(Y + p) + ", v" + S(Y + 2 * p) + ", v" + S(Y + 2 * p + 1));
    o("s_lshl_b32 s17, s10, " + S(8 + lg(T * 2)));
    o("s_add_u32 s12, s8, s17");
    o("s_addc_u32 s13, s9, 0");
    o("v_lshlrev_b32 v4, " + S(lg(T * 2)) + ", v0");
    for (int i = 0; i < T / 8; i++) o("global_store_dwordx4 v4, v[" + S(Y + 4 * i) + ":" + S(Y + 4 * i + 3) + "], s[12:13] offset:" + S(16 * i));
    if (WXT) {
        o("s_lshl_b32 s17, s10, " + S(8 + lg(T * 4)));
        o("s_add_u32 s12, s2, s17");
        o("s_addc_u32 s13, s3, 0");
        o("v_lshlrev_b32 v4, " + S(lg(T * 4)) + ", v0");
        for (int i = 0; i < T / 4; i++) o("global_store_dwordx4 v4, v[" + S(X + 4 * i) + ":" + S(X + 4 * i + 3) + "], s[12:13] offset:" + S(16 * i));
    }
    // sums of 16 columns (xs16) into Y, then of 32 (xs32) into X, E in the low byte
    for (int t = 0; t < T; t++) o("v_mov_b32 v" + S(Y + t) + ", v" + S(X + t));
    o("v_nop");
    o("v_nop");
    for (int m : {1, 2, 4, 8})   // a token's step T instructions after its last: no waits
        for (int t = 0; t < T; t++)
            o("v_add_f32_dpp v" + S(Y + t) + ", v" + S(Y + t) + ", v" + S(Y + t) + " row_xmask:" + S(m) + " row_mask:0xf bank_mask:0xf");
    for (int t = 0; t < T; t++) {
        o("v_permlanex16_b32 v" + S(SM) + ", v" + S(Y + t) + ", s86, s87");
        o("v_add_f32 v" + S(X + t) + ", v" + S(Y + t) + ", v" + S(SM));
    }
    for (int t = 0; t < T; t++) {
        o("v_and_or_b32 v" + S(Y + t) + ", v" + S(Y + t) + ", 0xffffff00, v" + S(E));
        o("v_and_or_b32 v" + S(X + t) + ", v" + S(X + t) + ", 0xffffff00, v" + S(E));
    }
    // xs16: lanes c % 16 == 0, at (c / 16) T
    o("v_and_b32 v5, 15, v0");
    o("v_cmp_eq_u32 vcc_lo, 0, v5");
    o("s_and_saveexec_b32 s16, vcc_lo");
    o("s_lshl_b32 s17, s10, " + S(4 + lg(T * 4)));
    o("s_add_u32 s12, s6, s17");
    o("s_addc_u32 s13, s7, 0");
    o("v_lshlrev_b32 v4, " + S(lg(T * 4) - 4) + ", v0");
    for (int i = 0; i < T / 4; i++) o("global_store_dwordx4 v4, v[" + S(Y + 4 * i) + ":" + S(Y + 4 * i + 3) + "], s[12:13] offset:" + S(16 * i));
    o("s_mov_b32 exec_lo, s16");
    // xs32: lanes c % 32 == 0, at (c / 32) T
    o("v_and_b32 v5, 31, v0");
    o("v_cmp_eq_u32 vcc_lo, 0, v5");
    o("s_and_saveexec_b32 s16, vcc_lo");
    o("s_lshl_b32 s17, s10, " + S(3 + lg(T * 4)));
    o("s_add_u32 s12, s4, s17");
    o("s_addc_u32 s13, s5, 0");
    o("v_lshlrev_b32 v4, " + S(lg(T * 4) - 5) + ", v0");
    for (int i = 0; i < T / 4; i++) o("global_store_dwordx4 v4, v[" + S(X + 4 * i) + ":" + S(X + 4 * i + 3) + "], s[12:13] offset:" + S(16 * i));
    o("s_mov_b32 exec_lo, s16");
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
