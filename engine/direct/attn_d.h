// attn_d: one generated token's attention, built to stream the KV cache
// (round 91). Round 89-90: at depth 8160 attn_part took ~6.1 ms a token for
// 1.2 GB of f16 K/V (~200 GB/s): load K, compute, load V, compute, one chunk
// at a time. Here every thread loads its K and V rows for 4 positions at
// once and keeps its own running max / sums (no workgroup barrier until the
// end), so the loads of a 64-position step are all in flight together.
//
// Workgroup: 128 threads, KV head g = s12 & 7, split s12 >> 3 (positions
// [split s10, min(+s10, s6))). Thread i: lane8 = i & 7 takes dims 16 lane8
// .. +15, pr = i >> 3 positions c + pr + 16 b (b = 0..3) of each 64-step.
// Scores: packed f16 FMAs over the 16 dims, then the 8 lanes add by DPP.
// At the end the 16 position groups fold (DPP, permlanex16, LDS) into
// attn_part's partials: max[4], sum[4], out[4][128] at s12 * 2112.
// Arguments as attn_part's (s[0:1] q f32 [32][128], s[2:3] K, s[4:5] V, s6
// positions, s7 scale, s[8:9] partials, s10 positions a split).
#pragma once
#include <algorithm>
#include <string>

namespace ts5::direct {

// batched (round 107: a verify pass's T tokens in one launch): token tgid_y
// (s13), whose q is s[0:1] + t q_bytes, positions s6 + t, partials s[8:9] +
// t part_bytes; the rest as one token's (every token of the launch must have
// the same s10, so each is bit for bit its own launch).
//
// share (round 113): the nt tokens of a verify pass in ONE workgroup of
// 128 nt threads sharing each K/V load (batched read the cache nt times:
// at 5-7k positions checking 2-4 tokens cost 1.3-2x one token's pass).
// Waves 4t..4t+3 run token t exactly as its own launch would; token 0's
// waves load each 64-position step (clamped to the last token's last
// position) and stage it in LDS, the others read it from there, so every
// token's arithmetic sees the same values in the same order. Positions
// past a token's own end, and whole steps past it (the last token's range
// can be longer), are masked: they add exp(-inf) = 0 times a finite value
// and rescale by exp(0) = 1, which leaves its sums bit for bit as they were.
// Token t's q, positions and partials as batched's; one workgroup a split.
// pairs (round 122, with nt 2): workgroup (x, y) shares K/V between tokens
// 2y and 2y + 1 only, so a T 4 pass is two 256-thread workgroups a split
// (round 121: one 512-thread workgroup a split left a WGP one workgroup and
// no overlap: T 3 / T 4 attention at 4,000 positions 2.2-2.4x one token).
inline std::string gen_attn_d(bool batched = false, unsigned q_bytes = 16384, unsigned part_bytes = 0, int nt = 0, bool pairs = false) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto v = [&](int a) { return "v" + S(a); };
    auto vr = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    const int Q = 8, M = 40, L = 44, SC = 48, KV = 64, O = 128, AD = 192, T = 196;
    const bool share = nt >= 2;
    if (share && (nt > 4 || batched)) return "";
    if (pairs && nt != 2) return "";
    const int lds = share ? std::max(32768, nt * 8448) : 8448;
    k += ".kernel attn_d\n.wave32\n.user_sgprs 12\n.tgid_x\n" + std::string(batched ? ".tgid_y\n" : "") + ".lds " + S(lds) + "\n";
    if (share && pairs) {
        o("v_lshrrev_b32 v5, 7, v0");
        o("v_readfirstlane_b32 s30, v5");          // the token in the pair
        o("v_and_b32 v0, 127, v0");
        o("s_lshl_b32 s31, s13, 1");               // the pair's first token (tgid_y 2)
        o("s_add_u32 s28, s6, s31");
        o("s_add_u32 s28, s28, 1");                // the pair's last token's positions
        o("s_add_u32 s13, s31, s30");              // token t of the pass
        o("s_mul_i32 s29, s30, 8448");             // its fold's LDS
    } else if (share) {
        o("v_lshrrev_b32 v5, 7, v0");
        o("v_readfirstlane_b32 s13, v5");          // token t
        o("v_and_b32 v0, 127, v0");                // the thread in its token's 128
        o("s_add_u32 s28, s6, " + S(nt - 1));      // the last token's positions
        o("s_mul_i32 s29, s13, 8448");             // its fold's LDS
        o("s_mov_b32 s30, s13");                   // (s13 becomes g below: the token stays in s30)
    }
    if (batched || share) {
        o("s_mul_i32 s26, s13, " + S(q_bytes));
        o("s_add_u32 s0, s0, s26");
        o("s_addc_u32 s1, s1, 0");
        o("s_add_u32 s6, s6, s13");
        o("s_mul_i32 s26, s13, " + S(part_bytes));
        o("s_add_u32 s8, s8, s26");
        o("s_addc_u32 s9, s9, 0");
    }
    o("s_and_b32 s13, s12, 7");                    // g
    o("s_lshr_b32 s14, s12, 3");                   // split
    o("s_mul_i32 s16, s14, s10");                  // ps
    o("s_add_u32 s15, s16, s10");
    o("s_min_u32 s15, s15, s6");                   // pe
    if (share) {
        o("s_add_u32 s27, s16, s10");
        o("s_min_u32 s27, s27, s28");              // the steps' end: the last token's pe
        o("s_sub_u32 s17, s28, 1");                // loads clamp at the last token's last position
        o("v_lshlrev_b32 v6, 4, v0");
        o("v_mov_b32 v7, v6");                     // (staging address, kept in v7 until the fold)
    } else {
        o("s_mov_b32 s27, s15");
        o("s_sub_u32 s17, s6, 1");                 // last position
    }
    o("v_and_b32 v1, 7, v0");                      // lane8
    o("v_lshrrev_b32 v2, 3, v0");                  // pr
    o("v_lshlrev_b32 v3, 5, v1");
    o("s_lshl_b32 s18, s13, 8");
    o("v_add_nc_u32 v3, s18, v3");                 // g 256 + lane8 32
    // q: head 4 g + h, dims 16 lane8 ..: f32 x s7 -> f16 pairs
    o("v_lshlrev_b32 v4, 6, v1");
    o("s_lshl_b32 s18, s13, 11");
    o("v_add_nc_u32 v4, s18, v4");
    for (int h = 0; h < 4; h++)
        for (int x = 0; x < 4; x++) o("global_load_dwordx4 " + vr(O + 16 * h + 4 * x, 4) + ", v4, s[0:1] offset:" + S(512 * h + 16 * x));
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 64; i++) o("v_mul_f32 " + v(O + i) + ", s7, " + v(O + i));
    for (int i = 0; i < 32; i++) o("v_cvt_pkrtz_f16_f32 " + v(Q + i) + ", " + v(O + 2 * i) + ", " + v(O + 2 * i + 1));
    for (int i = 0; i < 64; i++) o("v_mov_b32 " + v(O + i) + ", 0");
    for (int h = 0; h < 4; h++) {
        o("v_mov_b32 " + v(M + h) + ", 0xf149f2ca");   // -1e30
        o("v_mov_b32 " + v(L + h) + ", 0");
    }
    o("s_mov_b32 s19, s16");                       // c
    o("s_cmp_ge_u32 s16, s27");
    o("s_cbranch_scc1 fold");
    k += "step:\n";
    o("v_add_nc_u32 v4, s19, v2");                 // c + pr
    for (int b = 0; b < 4; b++) {
        o("v_add_nc_u32 v5, " + S(16 * b) + ", v4");
        o("v_min_u32 v5, s17, v5");
        o("v_lshl_add_u32 " + v(AD + b) + ", v5, 11, v3");
    }
    if (share) {
        o("s_cmp_eq_u32 s30, 0");
        o("s_cbranch_scc0 staged");
    }
    for (int b = 0; b < 4; b++) {
        o("global_load_dwordx4 " + vr(KV + 16 * b, 4) + ", " + v(AD + b) + ", s[2:3]");
        o("global_load_dwordx4 " + vr(KV + 16 * b + 4, 4) + ", " + v(AD + b) + ", s[2:3] offset:16");
        o("global_load_dwordx4 " + vr(KV + 16 * b + 8, 4) + ", " + v(AD + b) + ", s[4:5]");
        o("global_load_dwordx4 " + vr(KV + 16 * b + 12, 4) + ", " + v(AD + b) + ", s[4:5] offset:16");
    }
    if (share) {
        // token 0: the step into LDS, [group x of 4 dwords][thread] (16 B
        // each, so a wave's writes are 512 contiguous bytes); the others read
        // their thread's groups back into the same VGPRs
        o("s_waitcnt vmcnt(0)");
        for (int x = 0; x < 16; x++) o("ds_write_b128 v7, " + vr(KV + 4 * x, 4) + " offset:" + S(2048 * x));
        o("s_waitcnt lgkmcnt(0)");
        k += "staged:\n";
        o("s_barrier");
        o("s_cmp_eq_u32 s30, 0");
        o("s_cbranch_scc1 have");
        for (int x = 0; x < 16; x++) o("ds_read_b128 " + vr(KV + 4 * x, 4) + ", v7 offset:" + S(2048 * x));
        o("s_waitcnt lgkmcnt(0)");
        k += "have:\n";
    }
    // scores of position b as K arrives (vmcnt counts down in order)
    for (int b = 0; b < 4; b++) {
        o("s_waitcnt vmcnt(" + S(4 * (3 - b) + 2) + ")");
        for (int h = 0; h < 4; h++) {
            for (int j = 0; j < 8; j++)
                o("v_pk_fma_f16 " + v(T) + ", " + v(Q + 8 * h + j) + ", " + v(KV + 16 * b + j) + ", " + (j ? v(T) : std::string("0")));
            const std::string s = v(SC + 4 * b + h);
            o("v_fma_mix_f32 " + s + ", " + v(T) + ", 1.0, 0 op_sel:[0,0,0] op_sel_hi:[1,0,0]");
            o("v_fma_mix_f32 " + s + ", " + v(T) + ", 1.0, " + s + " op_sel:[1,0,0] op_sel_hi:[1,0,0]");
        }
    }
    o("v_nop");
    o("v_nop");
    for (int x = 0; x < 16; x++)
        for (int m : {1, 2, 4}) o("v_add_f32_dpp " + v(SC + x) + ", " + v(SC + x) + ", " + v(SC + x) + " row_xmask:" + S(m) + " row_mask:0xf bank_mask:0xf");
    // positions at or past the split's end
    o("s_add_u32 s20, s19, 64");
    o("s_cmp_le_u32 s20, s15");
    o("s_cbranch_scc1 nomask");
    for (int b = 0; b < 4; b++) {
        o("v_add_nc_u32 v5, " + S(16 * b) + ", v4");
        o("v_cmp_gt_u32 vcc_lo, s15, v5");
        for (int h = 0; h < 4; h++) o("v_cndmask_b32 " + v(SC + 4 * b + h) + ", 0xff800000, " + v(SC + 4 * b + h) + ", vcc_lo");
    }
    k += "nomask:\n";
    for (int h = 0; h < 4; h++) {
        o("v_max3_f32 v5, " + v(SC + h) + ", " + v(SC + 4 + h) + ", " + v(SC + 8 + h));
        o("v_max3_f32 v5, v5, " + v(SC + 12 + h) + ", " + v(M + h));   // new max
        o("v_sub_f32 v6, " + v(M + h) + ", v5");
        o("v_exp_f32 v6, v6");
        o("v_mov_b32 " + v(M + h) + ", v5");
        o("v_mul_f32 " + v(L + h) + ", " + v(L + h) + ", v6");
        for (int d = 0; d < 16; d++) o("v_mul_f32 " + v(O + 16 * h + d) + ", " + v(O + 16 * h + d) + ", v6");
        for (int b = 0; b < 4; b++) {
            const std::string s = v(SC + 4 * b + h);
            o("v_sub_f32 " + s + ", " + s + ", v5");
            o("v_exp_f32 " + s + ", " + s);
            o("v_add_f32 " + v(L + h) + ", " + v(L + h) + ", " + s);
        }
    }
    o("s_waitcnt vmcnt(0)");
    for (int b = 0; b < 4; b++)
        for (int h = 0; h < 4; h++)
            for (int d = 0; d < 16; d++)
                o("v_fma_mix_f32 " + v(O + 16 * h + d) + ", " + v(SC + 4 * b + h) + ", " + v(KV + 16 * b + 8 + d / 2) + ", " + v(O + 16 * h + d) +
                  " op_sel:[0," + S(d % 2) + ",0] op_sel_hi:[0,1,0]");
    if (share) o("s_barrier");                     // every token done with the step's LDS
    o("s_add_u32 s19, s19, 64");
    o("s_cmp_lt_u32 s19, s27");
    o("s_cbranch_scc1 step");
    // ---- fold the 16 position groups: lanes ^8 (DPP), ^16 (permlanex16), 4 waves (LDS)
    k += "fold:\n";
    o("s_mov_b32 s22, 0x76543210");
    o("s_mov_b32 s23, 0xfedcba98");
    o("v_lshrrev_b32 v5, 5, v0");
    o("v_readfirstlane_b32 s21, v5");              // wave
    // wave max of each head into v196 + h (T region is free now)
    for (int h = 0; h < 4; h++) {
        o("v_mov_b32 " + v(T + h) + ", " + v(M + h));
    }
    o("v_nop");
    o("v_nop");
    for (int h = 0; h < 4; h++) {
        o("v_max_f32_dpp " + v(T + h) + ", " + v(T + h) + ", " + v(T + h) + " row_xmask:8 row_mask:0xf bank_mask:0xf");
        o("v_permlanex16_b32 v6, " + v(T + h) + ", s22, s23");
        o("v_max_f32 " + v(T + h) + ", " + v(T + h) + ", v6");
    }
    // workgroup max through LDS [wave][h] at 0
    o("v_cmp_eq_u32 vcc_lo, 0, v0");
    o("s_mov_b32 s24, exec_lo");
    o("v_lshlrev_b32 v5, 5, v0");                  // (only lane 0 of each wave: tid & 31 == 0)
    o("v_and_b32 v6, 31, v0");
    o("v_cmp_eq_u32 vcc_lo, 0, v6");
    o("s_and_b32 exec_lo, exec_lo, vcc_lo");
    o("s_lshl_b32 s25, s21, 4");
    if (share) o("s_add_u32 s25, s25, s29");
    o("v_mov_b32 v6, s25");
    o("ds_write_b128 v6, " + vr(T, 4));
    o("s_mov_b32 exec_lo, s24");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o(share ? "v_mov_b32 v6, s29" : "v_mov_b32 v6, 0");
    o("ds_read_b128 v[200:203], v6");
    o("ds_read_b128 v[204:207], v6 offset:16");
    o("ds_read_b128 v[208:211], v6 offset:32");
    o("ds_read_b128 v[212:215], v6 offset:48");
    o("s_waitcnt lgkmcnt(0)");
    for (int h = 0; h < 4; h++) {
        o("v_max3_f32 " + v(T + h) + ", " + v(200 + h) + ", " + v(204 + h) + ", " + v(208 + h));
        o("v_max_f32 " + v(T + h) + ", " + v(T + h) + ", " + v(212 + h));   // the split's max
        o("v_sub_f32 v6, " + v(M + h) + ", " + v(T + h));
        o("v_exp_f32 v6, v6");
        o("v_mul_f32 " + v(L + h) + ", " + v(L + h) + ", v6");
        for (int d = 0; d < 16; d++) o("v_mul_f32 " + v(O + 16 * h + d) + ", " + v(O + 16 * h + d) + ", v6");
    }
    // sums over the wave's 4 position groups
    o("v_nop");
    o("v_nop");
    auto wsum = [&](int r) {
        o("v_add_f32_dpp " + v(r) + ", " + v(r) + ", " + v(r) + " row_xmask:8 row_mask:0xf bank_mask:0xf");
        o("v_permlanex16_b32 v6, " + v(r) + ", s22, s23");
        o("v_add_f32 " + v(r) + ", " + v(r) + ", v6");
    };
    for (int h = 0; h < 4; h++) wsum(L + h);
    for (int i = 0; i < 64; i++) wsum(O + i);
    // lanes 0-7 of each wave: LDS sums [wave] at 64 + 16 wave, outs at 256 + (8 wave + lane8) 256
    o("v_and_b32 v6, 31, v0");
    o("v_cmp_gt_u32 vcc_lo, 8, v6");
    o("s_and_b32 exec_lo, exec_lo, vcc_lo");
    o("s_lshl_b32 s25, s21, 4");
    if (share) o("s_add_u32 s25, s25, s29");
    o("v_mov_b32 v6, s25");
    o("ds_write_b128 v6, " + vr(L, 4) + " offset:64");
    o("s_lshl_b32 s25, s21, 3");
    o("v_add_nc_u32 v7, s25, v1");
    o("v_lshlrev_b32 v7, 8, v7");
    if (share) o("v_add_nc_u32 v7, s29, v7");
    for (int x = 0; x < 16; x++) o("ds_write_b128 v7, " + vr(O + 4 * x, 4) + " offset:" + S(256 + 16 * x));
    o("s_mov_b32 exec_lo, s24");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    // wave 0, lanes 0-7: add the 4 waves and write the partials
    o("s_cmp_eq_u32 s21, 0");
    o("s_cbranch_scc0 done");
    o("v_cmp_gt_u32 vcc_lo, 8, v0");
    o("s_and_b32 exec_lo, exec_lo, vcc_lo");
    o("v_lshlrev_b32 v7, 8, v1");
    if (share) o("v_add_nc_u32 v7, s29, v7");
    for (int w = 1; w < 4; w++) {
        o(share ? "v_mov_b32 v6, s29" : "v_mov_b32 v6, 0");
        o("ds_read_b128 v[200:203], v6 offset:" + S(64 + 16 * w));
        o("s_waitcnt lgkmcnt(0)");
        for (int h = 0; h < 4; h++) o("v_add_f32 " + v(L + h) + ", " + v(L + h) + ", " + v(200 + h));
        for (int x = 0; x < 16; x += 4) {
            for (int y = 0; y < 4; y++) o("ds_read_b128 " + vr(200 + 4 * y, 4) + ", v7 offset:" + S(256 + 2048 * w + 16 * (x + y)));
            o("s_waitcnt lgkmcnt(0)");
            for (int y = 0; y < 16; y++) o("v_add_f32 " + v(O + 4 * x + y) + ", " + v(O + 4 * x + y) + ", " + v(200 + y));
        }
    }
    // partials at s12 2112: max[4], sum[4], out[h] at 32 + 512 h + 64 lane8
    o("s_mul_i32 s25, s12, 2112");
    o("v_lshlrev_b32 v7, 6, v1");
    o("v_add_nc_u32 v7, s25, v7");
    for (int h = 0; h < 4; h++)
        for (int x = 0; x < 4; x++) o("global_store_dwordx4 v7, " + vr(O + 16 * h + 4 * x, 4) + ", s[8:9] offset:" + S(32 + 512 * h + 16 * x));
    o("v_cmp_eq_u32 vcc_lo, 0, v0");
    o("s_and_b32 exec_lo, exec_lo, vcc_lo");
    o("v_mov_b32 v6, s25");
    o("global_store_dwordx4 v6, " + vr(T, 4) + ", s[8:9]");
    o("global_store_dwordx4 v6, " + vr(L, 4) + ", s[8:9] offset:16");
    k += "done:\n";
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
