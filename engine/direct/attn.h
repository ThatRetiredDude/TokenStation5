// attn: one token's attention over the KV cache, Qwen3 8B's shape: 32 query
// heads, 8 KV heads (4 query heads share one), head size 128. q is f32
// [32][128], already normed and rotated; K and V are f16 [position][8][128]
// (one 256-byte row per position and KV head).
//
// Two kernels, split over positions as flash-decoding does:
//
// attn_part: one workgroup of 256 threads per (split of the positions, KV
// head): workgroup w takes KV head w & 7 and split w >> 3, s10 positions
// (a multiple of 256) each, worked 256 at a time with the running max and
// sums rescaled as the max grows (round 23 had one workgroup per 256
// positions and a long serial fold in attn_reduce). Per 256: thread i scores
// position 256 chunk + i against the 4 query heads (its K row: 16 loads of
// 16 bytes; q comes through scalar loads, the same for every lane), the
// workgroup takes each head's max and the sum of 2^(s - max), then thread
// i takes dims 2 (i & 63), +1 of V for positions i >> 6, +4, ... and the
// four groups are added in LDS. Writes per workgroup (2112 bytes): max[4],
// sum[4], out[4][128] (not yet divided by the sum).
//
// attn_reduce: one workgroup per KV head: thread i takes head i >> 6, dims
// 2 (i & 63), +1, and folds every chunk in (rescaling as the max grows),
// then writes out[32][128] f32.
//
// Scores are scaled by s7 (1/sqrt(128) log2 e) so exp2 does the softmax.
//
// attn_part arguments: s[0:1] q, s[2:3] K, s[4:5] V, s6 positions, s7 scale,
// s[8:9] partials, s10 positions per split, s11 unused; s12 workgroup id.
// attn_reduce arguments: s[0:1] partials, s[2:3] out, s4 splits; s10 KV head.
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_attn_part() {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto V = [&](int r) { return "v" + S(r); };
    k += ".kernel attn_part\n.wave32\n.user_sgprs 12\n.tgid_x\n.lds 16384\n";
    // LDS: wave maxima 0..127 ([head][wave]), wave sums 128..255, p 256 + 16 t
    // ([position][head]), V partials 8192 + 32 (64 group + dim pair).
    // VGPRs: v4-7 scores / p, v12-15 running max, v16-23 V sums, v120-123
    // this thread's share of the 2^ sums (round 24 kept them in v100-103,
    // inside the p registers, and every value came out wrong), v[48:111] K
    // and then p, v[32:47] V (and reduction temporaries), v112-119 temporaries:
    // 124 in all. At 132 (round 25) only one workgroup fit on a CU.
    o("s_and_b32 s13, s12, 7");            // g
    o("s_lshr_b32 s14, s12, 3");           // split
    o("s_mul_i32 s16, s14, s10");          // first position of the split
    o("s_add_u32 s15, s16, s10");
    o("s_min_u32 s15, s15, s6");           // its end
    o("s_lshl_b32 s18, s13, 11");          // q + 2048 g: this KV head's 4 query heads
    o("s_add_u32 s18, s0, s18");
    o("s_addc_u32 s19, s1, 0");
    o("s_sub_u32 s17, s6, 1");             // last position
    o("s_mov_b32 s88, 0xff800000");        // -inf
    o("s_mov_b32 s86, 0x76543210");
    o("s_mov_b32 s87, 0xfedcba98");
    o("v_lshrrev_b32 v3, 5, v0");
    o("v_lshlrev_b32 v3, 2, v3");          // 4 wave
    o("v_lshlrev_b32 v10, 4, v0");         // 16 i
    o("v_and_b32 v24, 63, v0");            // dp
    o("v_lshrrev_b32 v25, 6, v0");         // pg
    o("v_lshlrev_b32 v26, 2, v24");        // 4 dp
    o("v_lshlrev_b32 v28, 4, v25");        // 16 pg
    o("v_lshlrev_b32 v29, 5, v0");         // 32 i
    for (int h = 0; h < 4; h++) {
        o("v_mov_b32 v" + S(12 + h) + ", s88");
        o("v_mov_b32 v" + S(120 + h) + ", 0");
    }
    for (int r = 16; r < 24; r++) o("v_mov_b32 v" + S(r) + ", 0");
    auto wave_reduce = [&](const std::string& op) {   // v8 over the wave, into every lane
        o("v_nop");
        o("v_nop");
        for (int x : {1, 2, 4, 8}) o("v_" + op + "_f32_dpp v8, v8, v8 row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
        o("v_permlanex16_b32 v9, v8, s86, s87");
        o("v_" + op + "_f32 v8, v8, v9");
    };
    // K rows for the 256 positions from s<pos>: thread i's row, position
    // min(pos + i, n - 1), into v[48:111]. (Round 28 loaded the next chunk's
    // during V: 20-30% slower, V's first wait then waits for all of K.)
    auto kloads = [&](const std::string& pos) {
        o("v_add_nc_u32 v2, " + pos + ", v0");
        o("v_min_u32 v2, s17, v2");
        o("v_lshlrev_b32 v2, 3, v2");
        o("v_add_nc_u32 v2, s13, v2");
        o("v_lshlrev_b32 v2, 8, v2");      // (8 t + g) * 256
        for (int c = 0; c < 16; c++)
            o("global_load_dwordx4 v[" + S(48 + 4 * c) + ":" + S(51 + 4 * c) + "], v2, s[2:3] offset:" + S(16 * c));
    };
    k += "chunk:\n";
    o("s_cmp_ge_u32 s16, s15");
    o("s_cbranch_scc1 finish");
    kloads("s16");
    o("v_add_nc_u32 v1, s16, v0");         // t
    for (int h = 0; h < 4; h++) o("v_mov_b32 v" + S(4 + h) + ", 0");
    for (int c = 0; c < 16; c += 2) {      // two chunks of 8 dims at a time: q in s[20:83]
        for (int cc = 0; cc < 2; cc++)
            for (int h = 0; h < 4; h++)
                o("s_load_dwordx8 s[" + S(20 + 32 * cc + 8 * h) + ":" + S(27 + 32 * cc + 8 * h) + "], s[18:19], " +
                  S(512 * h + 32 * (c + cc)));
        o("s_waitcnt vmcnt(" + S(14 - c) + ") lgkmcnt(0)");
        for (int cc = 0; cc < 2; cc++)
            for (int j = 0; j < 4; j++) {
                const int r = 48 + 4 * (c + cc) + j;
                o("v_cvt_f32_f16 v112, v" + S(r));
                o("v_cvt_f32_f16_sdwa v113, v" + S(r) + " src0_sel:WORD_1");
                for (int h = 0; h < 4; h++) {
                    o("v_fmac_f32 v" + S(4 + h) + ", s" + S(20 + 32 * cc + 8 * h + 2 * j) + ", v112");
                    o("v_fmac_f32 v" + S(4 + h) + ", s" + S(21 + 32 * cc + 8 * h + 2 * j) + ", v113");
                }
            }
    }
    o("v_cmp_gt_u32 vcc_lo, s15, v1");     // t inside this split
    for (int h = 0; h < 4; h++) {
        o("v_mul_f32 v" + S(4 + h) + ", s7, v" + S(4 + h));
        o("v_cndmask_b32 v" + S(4 + h) + ", s88, v" + S(4 + h) + ", vcc_lo");
    }
    for (int h = 0; h < 4; h++) {
        o("v_mov_b32 v8, v" + S(4 + h));
        wave_reduce("max");
        o("ds_write_b32 v3, v8 offset:" + S(32 * h));
    }
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_mov_b32 v9, 0");
    for (int h = 0; h < 4; h++) {         // new max into v12 + h, the old sums' scale into v116 + h
        o("ds_read_b128 v[32:35], v9 offset:" + S(32 * h));
        o("ds_read_b128 v[36:39], v9 offset:" + S(32 * h + 16));
        o("s_waitcnt lgkmcnt(0)");
        o("v_max3_f32 v114, v32, v33, v34");
        o("v_max3_f32 v114, v114, v35, v36");
        o("v_max3_f32 v114, v114, v37, v38");
        o("v_max3_f32 v114, v114, v39, v" + S(12 + h));
        o("v_sub_f32 v" + S(116 + h) + ", v" + S(12 + h) + ", v114");
        o("v_exp_f32 v" + S(116 + h) + ", v" + S(116 + h));
        o("v_mov_b32 v" + S(12 + h) + ", v114");
    }
    for (int h = 0; h < 4; h++) {         // p = 2^(s - M); this thread's sum share
        o("v_sub_f32 v" + S(4 + h) + ", v" + S(4 + h) + ", v" + S(12 + h));
        o("v_exp_f32 v" + S(4 + h) + ", v" + S(4 + h));
        o("v_mul_f32 v" + S(120 + h) + ", v" + S(120 + h) + ", v" + S(116 + h));
        o("v_add_f32 v" + S(120 + h) + ", v" + S(120 + h) + ", v" + S(4 + h));
    }
    for (int h = 0; h < 4; h++) {         // rescale the V sums
        o("v_mul_f32 v" + S(16 + 2 * h) + ", v" + S(16 + 2 * h) + ", v" + S(116 + h));
        o("v_mul_f32 v" + S(17 + 2 * h) + ", v" + S(17 + 2 * h) + ", v" + S(116 + h));
    }
    o("ds_write_b128 v10, v[4:7] offset:256");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    // ---- V: dims 2 dp, +1, positions s16 + pg + 4 it; p one iteration
    // ahead in v[116:119] / v[4:7] (free once p is in LDS)
    o("v_add_nc_u32 v27, s16, v25");
    for (int b = 0; b < 4; b++) {          // 4 batches of 16: all loads and p reads first
        for (int i = 0; i < 16; i++) {
            const int it = 16 * b + i;
            o("v_add_nc_u32 v30, " + S(4 * it) + ", v27");
            o("v_min_u32 v30, s17, v30");
            o("v_lshlrev_b32 v30, 3, v30");
            o("v_add_nc_u32 v30, s13, v30");
            o("v_lshl_add_u32 v30, v30, 8, v26");
            o("global_load_dword v" + S(32 + i) + ", v30, s[4:5]");
        }
        const int buf[2] = {116, 4};
        auto pread = [&](int i) {
            o("ds_read_b128 v[" + S(buf[i & 1]) + ":" + S(buf[i & 1] + 3) + "], v28 offset:" + S(256 + 64 * (16 * b + i)));
        };
        pread(0);
        for (int i = 0; i < 16; i++) {
            if (i < 15) pread(i + 1);
            o("s_waitcnt vmcnt(" + S(15 - i) + ") lgkmcnt(" + S(i < 15 ? 1 : 0) + ")");
            o("v_cvt_f32_f16 v112, v" + S(32 + i));
            o("v_cvt_f32_f16_sdwa v113, v" + S(32 + i) + " src0_sel:WORD_1");
            for (int h = 0; h < 4; h++) {
                o("v_fmac_f32 v" + S(16 + 2 * h) + ", " + V(buf[i & 1] + h) + ", v112");
                o("v_fmac_f32 v" + S(17 + 2 * h) + ", " + V(buf[i & 1] + h) + ", v113");
            }
        }
    }
    o("s_add_u32 s16, s16, 256");
    o("s_branch chunk");
    k += "finish:\n";
    for (int h = 0; h < 4; h++) {
        o("v_mov_b32 v8, v" + S(120 + h));
        wave_reduce("add");
        o("ds_write_b32 v3, v8 offset:" + S(128 + 32 * h));
    }
    // ---- add the 4 position groups
    o("ds_write_b128 v29, v[16:19] offset:8192");
    o("ds_write_b128 v29, v[20:23] offset:8208");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_cmp_gt_u32 vcc_lo, 64, v0");
    o("s_and_saveexec_b32 s89, vcc_lo");
    o("s_cbranch_execz done");
    for (int g = 1; g < 4; g++) {
        o("ds_read_b128 v[32:35], v29 offset:" + S(8192 + 2048 * g));
        o("ds_read_b128 v[36:39], v29 offset:" + S(8208 + 2048 * g));
        o("s_waitcnt lgkmcnt(0)");
        for (int r = 0; r < 8; r++) o("v_add_f32 v" + S(16 + r) + ", v" + S(16 + r) + ", v" + S(32 + r));
    }
    o("v_mov_b32 v9, 0");
    for (int h = 0; h < 4; h++) {         // sums L_h into v40 + h
        o("ds_read_b128 v[32:35], v9 offset:" + S(128 + 32 * h));
        o("ds_read_b128 v[36:39], v9 offset:" + S(144 + 32 * h));
        o("s_waitcnt lgkmcnt(0)");
        o("v_add_f32 v" + S(40 + h) + ", v32, v33");
        for (int r = 34; r <= 39; r++) o("v_add_f32 v" + S(40 + h) + ", v" + S(40 + h) + ", v" + S(r));
    }
    o("s_mul_i32 s90, s12, 2112");
    o("v_lshl_add_u32 v30, v24, 3, s90");  // this thread's dims of head 0: + 32 + 512 h
    for (int h = 0; h < 4; h++)
        o("global_store_dwordx2 v30, v[" + S(16 + 2 * h) + ":" + S(17 + 2 * h) + "], s[8:9] offset:" + S(32 + 512 * h));
    o("v_cmp_eq_u32 vcc_lo, 0, v0");
    o("s_and_b32 exec_lo, exec_lo, vcc_lo");
    o("v_mov_b32 v31, s90");
    o("global_store_dwordx4 v31, v[12:15], s[8:9]");
    o("global_store_dwordx4 v31, v[40:43], s[8:9] offset:16");
    k += "done:\n";
    o("s_endpgm");
    return k;
}

inline std::string gen_attn_reduce() {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    k += ".kernel attn_reduce\n.wave32\n.user_sgprs 10\n.tgid_x\n";
    o("v_and_b32 v1, 63, v0");             // dp
    o("v_lshrrev_b32 v2, 6, v0");          // h
    o("s_mul_i32 s11, s10, 2112");         // chunk 0's partials for this KV head
    o("v_lshlrev_b32 v3, 2, v2");
    o("v_add_nc_u32 v3, s11, v3");         // m: + 4h (l: + 16)
    o("v_lshlrev_b32 v4, 9, v2");
    o("v_lshl_add_u32 v4, v1, 3, v4");
    o("v_add3_u32 v4, v4, s11, 32");       // o: + 32 + 512 h + 8 dp
    o("v_mov_b32 v10, 0xff800000");        // M
    o("v_mov_b32 v11, 0");                 // L
    o("v_mov_b32 v12, 0");                 // O
    o("v_mov_b32 v13, 0");
    o("s_mov_b32 s12, 0");
    k += "loop:\n";
    o("global_load_dword v5, v3, s[0:1]");
    o("global_load_dword v6, v3, s[0:1] offset:16");
    o("global_load_dwordx2 v[7:8], v4, s[0:1]");
    o("s_waitcnt vmcnt(0)");
    o("v_max_f32 v9, v10, v5");            // new M
    o("v_sub_f32 v14, v10, v9");
    o("v_exp_f32 v14, v14");               // old scale
    o("v_sub_f32 v15, v5, v9");
    o("v_exp_f32 v15, v15");               // this chunk's scale
    o("v_mov_b32 v10, v9");
    o("v_mul_f32 v11, v11, v14");
    o("v_fmac_f32 v11, v6, v15");
    o("v_mul_f32 v12, v12, v14");
    o("v_fmac_f32 v12, v7, v15");
    o("v_mul_f32 v13, v13, v14");
    o("v_fmac_f32 v13, v8, v15");
    o("v_add_nc_u32 v3, 0x4200, v3");      // next chunk: + 8 * 2112
    o("v_add_nc_u32 v4, 0x4200, v4");
    o("s_add_u32 s12, s12, 1");
    o("s_cmp_lt_u32 s12, s4");
    o("s_cbranch_scc1 loop");
    o("v_rcp_f32 v11, v11");
    o("v_mul_f32 v12, v12, v11");
    o("v_mul_f32 v13, v13, v11");
    o("s_lshl_b32 s13, s10, 11");          // out + 2048 g + 512 h + 8 dp
    o("v_lshlrev_b32 v5, 9, v2");
    o("v_lshl_add_u32 v5, v1, 3, v5");
    o("v_add_nc_u32 v5, s13, v5");
    o("global_store_dwordx2 v5, v[12:13], s[2:3]");
    o("s_endpgm");
    (void)S;
    return k;
}

}  // namespace ts5::direct
