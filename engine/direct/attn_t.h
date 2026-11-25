// attn_t: a big pass's attention, tiled (round 85). attn_part2 read the KV
// cache once per 2 prompt tokens, so a long prompt's attention grew to
// ~0.27 ms a token per 1k positions (rounds 83-84: Vulkan 13-33% faster
// from 2k tokens up). Here a workgroup takes 32 consecutive tokens x the 4
// query heads of one KV head (128 rows) over one split of the positions and
// stages K and V 32 positions at a time in LDS, so each chunk is read once
// for 32 tokens.
//
// Threads: thread i takes row i >> 1 (head i >> 6, token (i >> 1) & 31) and
// dims 64 (i & 1) .. +63; the two halves of a row add their scores with one
// DPP swap. q (f32, pre-scaled by s7) stays in VGPRs; scores q.k and the
// output sums p.v are f32 (v_fma_mix with K/V f16 from LDS).
//
// Splits as the runner's (one layout for the whole pass): positions
// [split * s10, min(+s10, s12)). Token t sees positions < s6 + t + 1. A row
// with no visible position in its split writes max -1e30, sum 0, out 0.
// Partials as attn_part's: per token (s11 bytes apart) and workgroup
// split * 8 + g: max[4], sum[4], out[4][128], 2112 bytes.
//
// Arguments: s[0:1] q (the pass's tokens, 16 KiB each), s[2:3] K, s[4:5] V
// (the layer's cache), s6 pos0, s7 scale, s[8:9] partials, s10 positions a
// split (a multiple of 256), s11 partial bytes a token, s12 positions after
// the pass (s6 + tokens). tgid x: split * 8 + g (s14), y: 32-token tile (s15).
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_attn_t() {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto v = [&](int a) { return "v" + S(a); };
    auto vr = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    // VGPRs: v0 tid, v1 half, v2 row, v3 head in group, v4 token in tile,
    // v5 lim - chunk, v6 lim, v7 LDS write, v8 LDS read (128 half), v9 tid >> 3,
    // v10 address, v11 (tid & 7) 32 + 256 g, v12 max, v13 sum, v14-15 temps,
    // v16-31 the next chunk (K 16..23, V 24..31), v32-63 scores / p,
    // v64-127 q, v128-191 out, v192-223 / v224-255 a K or V row (two buffers).
    const int PF = 16, SC = 32, Q = 64, O = 128, KB0 = 192, KB1 = 224;
    k += ".kernel attn_t\n.wave32\n.user_sgprs 14\n.tgid_x\n.tgid_y\n.lds 16384\n";
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
    o("v_and_b32 v1, 1, v0");
    o("v_lshrrev_b32 v2, 1, v0");
    o("v_lshrrev_b32 v3, 5, v2");
    o("v_and_b32 v4, 31, v2");
    // lim = min(pe, pos0 + t0 + tt + 1)
    o("s_add_u32 s26, s6, s20");
    o("s_add_u32 s26, s26, 1");
    o("v_add_nc_u32 v6, s26, v4");
    o("v_min_u32 v6, s19, v6");
    o("v_lshlrev_b32 v7, 5, v0");
    o("v_lshlrev_b32 v8, 7, v1");
    o("v_lshrrev_b32 v9, 3, v0");
    o("v_and_b32 v11, 7, v0");
    o("v_lshlrev_b32 v11, 5, v11");
    o("s_lshl_b32 s26, s16, 8");
    o("v_add_nc_u32 v11, s26, v11");
    // q: (t0 + tt) 16384 + 2048 g + 512 hl + 256 half
    o("v_add_nc_u32 v10, s20, v4");
    o("v_lshlrev_b32 v10, 14, v10");
    o("v_lshl_add_u32 v10, v3, 9, v10");
    o("v_lshl_add_u32 v10, v1, 8, v10");
    o("s_lshl_b32 s26, s16, 11");
    o("v_add_nc_u32 v10, s26, v10");
    for (int i = 0; i < 16; i++) o("global_load_dwordx4 " + vr(Q + 4 * i, 4) + ", v10, s[0:1] offset:" + S(16 * i));
    o("v_mov_b32 v12, 0xf149f2ca");                // -1e30
    o("v_mov_b32 v13, 0");
    for (int d = 0; d < 64; d++) o("v_mov_b32 " + v(O + d) + ", 0");
    o("s_waitcnt vmcnt(0)");
    for (int d = 0; d < 64; d++) o("v_mul_f32 " + v(Q + d) + ", s7, " + v(Q + d));
    o("s_mov_b32 s24, s18");                       // c
    o("s_cmp_ge_u32 s18, s22");
    o("s_cbranch_scc1 store");
    auto prefetch = [&]() {                        // chunk s24 into v16-31
        o("v_add_nc_u32 v10, s24, v9");
        o("v_min_u32 v10, s25, v10");
        o("v_lshlrev_b32 v10, 11, v10");
        o("v_add_nc_u32 v10, v10, v11");
        o("global_load_dwordx4 " + vr(PF, 4) + ", v10, s[2:3]");
        o("global_load_dwordx4 " + vr(PF + 4, 4) + ", v10, s[2:3] offset:16");
        o("global_load_dwordx4 " + vr(PF + 8, 4) + ", v10, s[4:5]");
        o("global_load_dwordx4 " + vr(PF + 12, 4) + ", v10, s[4:5] offset:16");
    };
    prefetch();
    k += "chunk:\n";
    o("s_waitcnt vmcnt(0)");
    o("s_barrier");                                // everyone done with the last chunk
    o("ds_write_b128 v7, " + vr(PF, 4));
    o("ds_write_b128 v7, " + vr(PF + 4, 4) + " offset:16");
    o("ds_write_b128 v7, " + vr(PF + 8, 4) + " offset:8192");
    o("ds_write_b128 v7, " + vr(PF + 12, 4) + " offset:8208");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("s_add_u32 s27, s24, 32");                   // the next chunk, if any, on its way
    o("s_cmp_ge_u32 s27, s22");
    o("s_cbranch_scc1 nopf");
    o("s_mov_b32 s28, s24");
    o("s_mov_b32 s24, s27");
    prefetch();
    o("s_mov_b32 s24, s28");
    k += "nopf:\n";
    // K row j (this half's 64 dims) into buffer j % 2
    auto read_row = [&](int j, int base) {
        const int kb = j % 2 ? KB1 : KB0;
        for (int q = 0; q < 8; q++) o("ds_read_b128 " + vr(kb + 4 * q, 4) + ", v8 offset:" + S(base + 256 * j + 16 * q));
    };
    // ---- scores
    read_row(0, 0);
    for (int j = 0; j < 32; j++) {
        if (j + 1 < 32) {
            read_row(j + 1, 0);
            o("s_waitcnt lgkmcnt(8)");
        } else {
            o("s_waitcnt lgkmcnt(0)");
        }
        const int kb = j % 2 ? KB1 : KB0, a = SC + j;
        for (int d = 0; d < 64; d++) {
            const std::string acc = d % 2 ? "v14" : v(a);
            const std::string c = d < 2 ? "0" : acc;
            o("v_fma_mix_f32 " + acc + ", " + v(Q + d) + ", " + v(kb + d / 2) + ", " + c + " op_sel:[0," + S(d % 2) + ",0] op_sel_hi:[0,1,0]");
        }
        o("v_add_f32 " + v(a) + ", " + v(a) + ", v14");
    }
    o("v_nop");
    o("v_nop");
    for (int j = 0; j < 32; j++) o("v_add_f32_dpp " + v(SC + j) + ", " + v(SC + j) + ", " + v(SC + j) + " row_xmask:1 row_mask:0xf bank_mask:0xf");
    // ---- mask the chunk that crosses a limit
    o("s_add_u32 s27, s24, 32");
    o("s_cmp_le_u32 s27, s23");
    o("s_cbranch_scc1 nomask");
    o("v_subrev_nc_u32 v5, s24, v6");
    for (int j = 0; j < 32; j++) {
        o("v_cmp_lt_i32 vcc_lo, " + S(j) + ", v5");
        o("v_cndmask_b32 " + v(SC + j) + ", 0xff800000, " + v(SC + j) + ", vcc_lo");
    }
    k += "nomask:\n";
    // ---- softmax step
    o("v_max_f32 v14, " + v(SC) + ", " + v(SC + 1));
    for (int j = 2; j < 32; j++) o("v_max_f32 v14, v14, " + v(SC + j));
    o("v_max_f32 v15, v12, v14");                  // new max
    o("v_sub_f32 v14, v12, v15");
    o("v_exp_f32 v14, v14");                       // the old sums' scale
    o("v_mov_b32 v12, v15");
    o("v_mul_f32 v13, v13, v14");
    for (int d = 0; d < 64; d++) o("v_mul_f32 " + v(O + d) + ", " + v(O + d) + ", v14");
    for (int j = 0; j < 32; j++) {
        o("v_sub_f32 " + v(SC + j) + ", " + v(SC + j) + ", v15");
        o("v_exp_f32 " + v(SC + j) + ", " + v(SC + j));
    }
    for (int j = 0; j < 32; j++) o("v_add_f32 v13, v13, " + v(SC + j));
    // ---- out += p v
    read_row(0, 8192);
    for (int j = 0; j < 32; j++) {
        if (j + 1 < 32) {
            read_row(j + 1, 8192);
            o("s_waitcnt lgkmcnt(8)");
        } else {
            o("s_waitcnt lgkmcnt(0)");
        }
        const int kb = j % 2 ? KB1 : KB0;
        for (int d = 0; d < 64; d++)
            o("v_fma_mix_f32 " + v(O + d) + ", " + v(SC + j) + ", " + v(kb + d / 2) + ", " + v(O + d) + " op_sel:[0," + S(d % 2) + ",0] op_sel_hi:[0,1,0]");
    }
    o("s_add_u32 s24, s24, 32");
    o("s_cmp_lt_u32 s24, s22");
    o("s_cbranch_scc1 chunk");
    k += "store:\n";
    // base = (t0 + tt) s11 + 2112 (split 8 + g)
    o("v_add_nc_u32 v10, s20, v4");
    o("v_mul_lo_u32 v10, s11, v10");
    o("s_mul_i32 s26, s14, 2112");
    o("v_add_nc_u32 v10, s26, v10");
    o("v_lshlrev_b32 v14, 9, v3");
    o("v_lshl_add_u32 v14, v1, 8, v14");
    o("v_add_nc_u32 v14, v10, v14");               // out + 512 hl + 256 half (+ 32)
    for (int i = 0; i < 16; i++) o("global_store_dwordx4 v14, " + vr(O + 4 * i, 4) + ", s[8:9] offset:" + S(32 + 16 * i));
    o("v_cmp_eq_u32 vcc_lo, 0, v1");
    o("s_and_b32 exec_lo, exec_lo, vcc_lo");
    o("v_lshl_add_u32 v10, v3, 2, v10");
    o("global_store_dword v10, v12, s[8:9]");
    o("global_store_dword v10, v13, s[8:9] offset:16");
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
