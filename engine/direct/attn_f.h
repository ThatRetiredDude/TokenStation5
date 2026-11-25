// attn_f: a big pass's attention as two tiled products (round 87). Round 86's
// profile at depth 7680: attention 2.1 ms of 3.1 ms a prompt token, ~2
// TFLOP/s (attn_part2 and attn_t alike: one row a thread, every K/V value
// read from LDS for one FMA, ~40 KB of unrolled code). Here, as mm_q4k_t
// does for the weights, each thread works a 4 x 4 block with packed f16 FMAs
// (f16 sums over 32 dims or 32 positions, folded into f32), so an LDS read
// feeds 8-16 FMAs, and the code stays ~20 KB.
//
// Workgroup: 32 tokens x the 4 query heads of KV head g = 128 rows (row
// 4 tt + h: a thread's 4 rows are one token's), one split of the positions,
// 32 positions a chunk. Thread i: rg = i >> 3 (token tt = rg, rows 4 rg ..
// +3), lo = i & 7: scores for positions 4 lo .. +3 of the chunk, outputs for
// dims 16 lo .. +15. A row's max goes over its 8 lanes by DPP; the 2^ sums
// stay per lane until the end.
// LDS: Q f16 [128][128] at 0 (pre-scaled by s7), K [32][128] at 32768,
// V at 40960, P f16 [128][32] at 49152.
//
// Arguments, splits, partials and the reduce: as attn_t (attn_t.h).
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_attn_f() {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto v = [&](int a) { return "v" + S(a); };
    auto vr = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    const int LQ = 0, LK = 32768, LV = 40960, LP = 49152;
    // VGPRs: v0 tid, v1 rg, v2 lo, v3 Q read (rg 1024), v4 K read (LK + lo 1024),
    // v5 P write (LP + rg 256 + lo 8), v6 P read (LP + rg 256), v7 V read
    // (LV + lo 32), v8 K/V write (tid 32), v9 address, v10 (tid & 7) 32 + 256 g,
    // v11 tid >> 3, v12 lim, v13 lim - c - 4 lo, v14-15 temps.
    const int M = 16, L = 20, AL = 24, SC = 28, PF = 44, ACC = 60, O = 92, B = 156;
    k += ".kernel attn_f\n.wave32\n.user_sgprs 14\n.tgid_x\n.tgid_y\n.lds 57344\n";
    o("s_and_b32 s16, s14, 7");                    // g
    o("s_lshr_b32 s17, s14, 3");                   // split
    o("s_mul_i32 s18, s17, s10");                  // ps
    o("s_add_u32 s19, s18, s10");
    o("s_min_u32 s19, s19, s12");                  // pe
    o("s_lshl_b32 s20, s15, 5");                   // t0
    o("s_add_u32 s21, s6, s20");
    o("s_add_u32 s23, s21, 1");
    o("s_min_u32 s23, s23, s19");                  // the tile's smallest limit
    o("s_add_u32 s21, s21, 32");
    o("s_min_u32 s22, s21, s19");                  // E: the tile's end
    o("s_sub_u32 s25, s12, 1");                    // last position
    o("v_lshrrev_b32 v1, 3, v0");                  // rg = tt
    o("v_and_b32 v2, 7, v0");                      // lo
    o("s_add_u32 s26, s6, s20");
    o("s_add_u32 s26, s26, 1");
    o("v_add_nc_u32 v12, s26, v1");
    o("v_min_u32 v12, s19, v12");                  // lim
    o("v_lshlrev_b32 v3, 10, v1");
    o("v_lshlrev_b32 v4, 10, v2");
    o("v_add_nc_u32 v4, " + S(LK) + ", v4");
    o("v_lshlrev_b32 v6, 8, v1");
    o("v_add_nc_u32 v6, " + S(LP) + ", v6");
    o("v_lshl_add_u32 v5, v2, 3, v6");
    o("v_lshlrev_b32 v7, 5, v2");
    o("v_add_nc_u32 v7, " + S(LV) + ", v7");
    o("v_lshlrev_b32 v8, 5, v0");
    o("v_lshrrev_b32 v11, 3, v0");
    o("v_lshlrev_b32 v10, 5, v2");
    o("s_lshl_b32 s26, s16, 8");
    o("v_add_nc_u32 v10, s26, v10");
    for (int i = 0; i < 4; i++) {
        o("v_mov_b32 " + v(M + i) + ", 0xf149f2ca");   // -1e30
        o("v_mov_b32 " + v(L + i) + ", 0");
    }
    for (int d = 0; d < 64; d++) o("v_mov_b32 " + v(O + d) + ", 0");
    o("s_mov_b32 s24, s18");                       // c
    o("s_cmp_ge_u32 s18, s22");
    o("s_cbranch_scc1 store");
    // ---- Q: thread i stages row i >> 1 (token (i >> 1) >> 2, head (i >> 1) & 3),
    // dims 64 (i & 1) ..: f32, times s7, f16 into LDS
    o("v_lshrrev_b32 v14, 1, v0");
    o("v_lshrrev_b32 v15, 2, v14");
    o("v_add_nc_u32 v15, s20, v15");
    o("v_lshlrev_b32 v9, 14, v15");                // token 16384
    o("v_and_b32 v15, 3, v14");
    o("v_lshl_add_u32 v9, v15, 9, v9");            // + head 512
    o("v_and_b32 v15, 1, v0");
    o("v_lshl_add_u32 v9, v15, 8, v9");            // + half 256
    o("s_lshl_b32 s26, s16, 11");
    o("v_add_nc_u32 v9, s26, v9");                 // + 2048 g
    for (int i = 0; i < 16; i++) o("global_load_dwordx4 " + vr(B + 4 * i, 4) + ", v9, s[0:1] offset:" + S(16 * i));
    o("v_lshlrev_b32 v14, 8, v14");
    o("v_lshl_add_u32 v14, v15, 7, v14");          // LDS: row 256 + half 128
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 64; i++) o("v_mul_f32 " + v(B + i) + ", s7, " + v(B + i));
    for (int i = 0; i < 32; i++) o("v_cvt_pkrtz_f16_f32 " + v(B + i) + ", " + v(B + 2 * i) + ", " + v(B + 2 * i + 1));
    for (int i = 0; i < 8; i++) o("ds_write_b128 v14, " + vr(B + 4 * i, 4) + " offset:" + S(LQ + 16 * i));
    // ---- the chunk loop: K and V of chunk s24 into v[PF..PF+15], then LDS
    auto prefetch = [&]() {
        o("v_add_nc_u32 v9, s24, v11");
        o("v_min_u32 v9, s25, v9");
        o("v_lshlrev_b32 v9, 11, v9");
        o("v_add_nc_u32 v9, v9, v10");
        o("global_load_dwordx4 " + vr(PF, 4) + ", v9, s[2:3]");
        o("global_load_dwordx4 " + vr(PF + 4, 4) + ", v9, s[2:3] offset:16");
        o("global_load_dwordx4 " + vr(PF + 8, 4) + ", v9, s[4:5]");
        o("global_load_dwordx4 " + vr(PF + 12, 4) + ", v9, s[4:5] offset:16");
    };
    prefetch();
    k += "chunk:\n";
    o("s_waitcnt vmcnt(0)");
    o("s_barrier");                                // everyone done with the last chunk's K, V
    o("ds_write_b128 v8, " + vr(PF, 4) + " offset:" + S(LK));
    o("ds_write_b128 v8, " + vr(PF + 4, 4) + " offset:" + S(LK + 16));
    o("ds_write_b128 v8, " + vr(PF + 8, 4) + " offset:" + S(LV));
    o("ds_write_b128 v8, " + vr(PF + 12, 4) + " offset:" + S(LV + 16));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("s_add_u32 s27, s24, 32");
    o("s_cmp_ge_u32 s27, s22");
    o("s_cbranch_scc1 nopf");
    o("s_mov_b32 s28, s24");
    o("s_mov_b32 s24, s27");
    prefetch();
    o("s_mov_b32 s24, s28");
    k += "nopf:\n";
    // ---- scores: 16 steps of 8 dims; q rows 4 rg + i, k rows 4 lo + j (v4 + 256 j)
    auto qk_read = [&](int st) {
        const int b = B + 32 * (st % 2);
        for (int i = 0; i < 4; i++) o("ds_read_b128 " + vr(b + 4 * i, 4) + ", v3 offset:" + S(LQ + 256 * i + 16 * st));
        for (int j = 0; j < 4; j++) o("ds_read_b128 " + vr(b + 16 + 4 * j, 4) + ", v4 offset:" + S(256 * j + 16 * st));
    };
    for (int x = 0; x < 16; x++) o("v_mov_b32 " + v(SC + x) + ", 0");
    qk_read(0);
    for (int st = 0; st < 16; st++) {
        if (st + 1 < 16) {
            qk_read(st + 1);
            o("s_waitcnt lgkmcnt(8)");
        } else {
            o("s_waitcnt lgkmcnt(0)");
        }
        const int b = B + 32 * (st % 2);
        for (int p = 0; p < 4; p++)
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) {
                    const std::string a = v(ACC + 4 * i + j);
                    o("v_pk_fma_f16 " + a + ", " + v(b + 4 * i + p) + ", " + v(b + 16 + 4 * j + p) + ", " + (st % 4 == 0 && p == 0 ? "0" : a));
                }
        if (st % 4 == 3)                           // 32 dims: into f32
            for (int x = 0; x < 16; x++) {
                o("v_fma_mix_f32 " + v(SC + x) + ", " + v(ACC + x) + ", 1.0, " + v(SC + x) + " op_sel:[0,0,0] op_sel_hi:[1,0,0]");
                o("v_fma_mix_f32 " + v(SC + x) + ", " + v(ACC + x) + ", 1.0, " + v(SC + x) + " op_sel:[1,0,0] op_sel_hi:[1,0,0]");
            }
    }
    // ---- mask the chunk that crosses a limit (a thread's rows share one)
    o("s_add_u32 s27, s24, 32");
    o("s_cmp_le_u32 s27, s23");
    o("s_cbranch_scc1 nomask");
    o("v_subrev_nc_u32 v13, s24, v12");
    o("v_lshlrev_b32 v14, 2, v2");
    o("v_sub_nc_u32 v13, v13, v14");               // lim - c - 4 lo
    for (int j = 0; j < 4; j++) {
        o("v_cmp_lt_i32 vcc_lo, " + S(j) + ", v13");
        for (int i = 0; i < 4; i++) o("v_cndmask_b32 " + v(SC + 4 * i + j) + ", 0xff800000, " + v(SC + 4 * i + j) + ", vcc_lo");
    }
    k += "nomask:\n";
    // ---- softmax step: row max over the 8 lanes, scale, p
    for (int i = 0; i < 4; i++) {
        o("v_max_f32 " + v(AL + i) + ", " + v(SC + 4 * i) + ", " + v(SC + 4 * i + 1));
        o("v_max3_f32 " + v(AL + i) + ", " + v(AL + i) + ", " + v(SC + 4 * i + 2) + ", " + v(SC + 4 * i + 3));
    }
    o("v_nop");
    o("v_nop");
    for (int i = 0; i < 4; i++)
        for (int x : {1, 2, 4}) o("v_max_f32_dpp " + v(AL + i) + ", " + v(AL + i) + ", " + v(AL + i) + " row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
    for (int i = 0; i < 4; i++) {
        o("v_max_f32 v14, " + v(M + i) + ", " + v(AL + i));   // new max
        o("v_sub_f32 " + v(AL + i) + ", " + v(M + i) + ", v14");
        o("v_exp_f32 " + v(AL + i) + ", " + v(AL + i));       // the old sums' scale
        o("v_mov_b32 " + v(M + i) + ", v14");
        for (int j = 0; j < 4; j++) {
            o("v_sub_f32 " + v(SC + 4 * i + j) + ", " + v(SC + 4 * i + j) + ", v14");
            o("v_exp_f32 " + v(SC + 4 * i + j) + ", " + v(SC + 4 * i + j));
        }
        o("v_mul_f32 " + v(L + i) + ", " + v(L + i) + ", " + v(AL + i));
        for (int j = 0; j < 4; j++) o("v_add_f32 " + v(L + i) + ", " + v(L + i) + ", " + v(SC + 4 * i + j));
        for (int d = 0; d < 16; d++) o("v_mul_f32 " + v(O + 16 * i + d) + ", " + v(O + 16 * i + d) + ", " + v(AL + i));
        o("v_cvt_pkrtz_f16_f32 v14, " + v(SC + 4 * i) + ", " + v(SC + 4 * i + 1));
        o("v_cvt_pkrtz_f16_f32 v15, " + v(SC + 4 * i + 2) + ", " + v(SC + 4 * i + 3));
        o("ds_write_b64 v5, v[14:15] offset:" + S(64 * i));
    }
    o("s_waitcnt lgkmcnt(0)");                     // P rows: this wave's own lanes
    // ---- out += p v: 8 steps of 4 positions; P rows 4 rg + i (v6 + 64 i), V rows (v7 + 256 pos)
    auto pv_read = [&](int st) {
        const int b = B + 40 * (st % 2);
        for (int i = 0; i < 4; i++) o("ds_read_b64 " + vr(b + 2 * i, 2) + ", v6 offset:" + S(64 * i + 8 * st));
        for (int q = 0; q < 4; q++) {
            o("ds_read_b128 " + vr(b + 8 + 8 * q, 4) + ", v7 offset:" + S(256 * (4 * st + q)));
            o("ds_read_b128 " + vr(b + 12 + 8 * q, 4) + ", v7 offset:" + S(256 * (4 * st + q) + 16));
        }
    };
    pv_read(0);
    for (int st = 0; st < 8; st++) {
        if (st + 1 < 8) {
            pv_read(st + 1);
            o("s_waitcnt lgkmcnt(12)");
        } else {
            o("s_waitcnt lgkmcnt(0)");
        }
        const int b = B + 40 * (st % 2);
        for (int q = 0; q < 4; q++)
            for (int i = 0; i < 4; i++)
                for (int dp = 0; dp < 8; dp++) {
                    const std::string a = v(ACC + 8 * i + dp), h = S(q % 2);
                    o("v_pk_fma_f16 " + a + ", " + v(b + 2 * i + q / 2) + ", " + v(b + 8 + 8 * q + dp) + ", " + (st == 0 && q == 0 ? "0" : a) +
                      " op_sel:[" + h + ",0,0] op_sel_hi:[" + h + ",1,1]");
                }
    }
    for (int i = 0; i < 4; i++)
        for (int dp = 0; dp < 8; dp++)
            for (int h = 0; h < 2; h++)
                o("v_fma_mix_f32 " + v(O + 16 * i + 2 * dp + h) + ", " + v(ACC + 8 * i + dp) + ", 1.0, " + v(O + 16 * i + 2 * dp + h) +
                  " op_sel:[" + S(h) + ",0,0] op_sel_hi:[1,0,0]");
    o("s_add_u32 s24, s24, 32");
    o("s_cmp_lt_u32 s24, s22");
    o("s_cbranch_scc1 chunk");
    k += "store:\n";
    // the sums over the row's 8 lanes
    o("v_nop");
    o("v_nop");
    for (int i = 0; i < 4; i++)
        for (int x : {1, 2, 4}) o("v_add_f32_dpp " + v(L + i) + ", " + v(L + i) + ", " + v(L + i) + " row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
    // base = (t0 + tt) s11 + 2112 (split 8 + g); row i = head i: out + 512 i + 64 lo
    o("v_add_nc_u32 v9, s20, v1");
    o("v_mul_lo_u32 v9, s11, v9");
    o("s_mul_i32 s26, s14, 2112");
    o("v_add_nc_u32 v9, s26, v9");
    o("v_lshl_add_u32 v14, v2, 6, v9");
    for (int i = 0; i < 4; i++)
        for (int x = 0; x < 4; x++)
            o("global_store_dwordx4 v14, " + vr(O + 16 * i + 4 * x, 4) + ", s[8:9] offset:" + S(32 + 512 * i + 16 * x));
    o("v_cmp_eq_u32 vcc_lo, 0, v2");
    o("s_and_b32 exec_lo, exec_lo, vcc_lo");
    o("global_store_dwordx4 v9, " + vr(M, 4) + ", s[8:9]");
    o("global_store_dwordx4 v9, " + vr(L, 4) + ", s[8:9] offset:16");
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
