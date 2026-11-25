// mvt_q6k: y_t = W x_t for T tokens at once, W Q6_K, every y_t bit for bit
// what mv_q6k gives for x_t alone. The frame is mvt_q4k's (column chunks of
// one step staged in LDS for every token, K row slots per wave, a lane sum
// per slot and token kept across chunks); the arithmetic is mv_q6k's, lane
// for lane: each lane's 64 values of a step (4 blocks' quarters) as f32 once,
// then for each token the four 16-value chains, "- 32 sum(x)", and the
// quarter's d * scale into the lane sum, quarters in order, steps in order.
//
// Arguments as mvt_q4k. W must be readable 2 bytes past its end (as mv_q6k).
#pragma once
#include <algorithm>
#include <string>

namespace ts5::direct {

inline std::string gen_mvt_q6k(int ncols, int T, int K = 8) {
    const int nit = ncols / 2048;
    const int row_bytes = ncols / 256 * 210;
    const int SUMS = T * 8192;
    const int lds = std::max(SUMS + T * 512, 33 * 1024);
    auto S = [](long v) { return std::to_string(v); };
    auto V = [&](int r) { return "v" + S(r); };
    auto VR = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto label = [&](const std::string& l) { k += l + ":\n"; };
    // v1 scale byte shift, v2/v3/v4 ql/qh/scale offsets, v5 misalignment, v6
    // sums address, v7-14 x addresses (k1, c), v15 staging LDS address,
    // v16/v17 staging global offsets, v18 sums read, v19 sums write, v20-23
    // temps; then lane sums, two weight sets of 18, scales and temps, the
    // 64 values as f32, x of two (token, quarter) items (17 each).
    const int ACC = 24, WA = ACC + K * T, WB = WA + 18, DS = WB + 18, VF = DS + 12, XQ = VF + 64, XE = XQ + 34;
    if (XE > 256) return "";
    auto ds = [&](int i) { return V(DS + i); };   // 0-3 d * scale, 4 d, 5-6 shifted scales, 7-10 temps, 11 the chain
    k += ".kernel mvt_q6k\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(lds) + "\n";
    // ---- lane constants (mv_q6k's)
    o("v_bfe_u32 v20, v0, 2, 3");          // g
    o("v_and_b32 v21, 1, v0");             // t & 1
    o("v_bfe_u32 v22, v0, 1, 1");          // h
    o("v_mul_u32_u24 v23, 210, v20");
    o("v_lshlrev_b32 v1, 4, v21");         // 16 (t & 1) for now
    o("v_lshl_add_u32 v2, v22, 6, v1");
    o("v_add_nc_u32 v2, v2, v23");
    o("v_lshl_add_u32 v3, v22, 5, v1");
    o("v_add_nc_u32 v3, v3, v23");
    o("v_add_nc_u32 v3, 128, v3");
    o("v_lshl_add_u32 v4, v22, 3, v23");
    o("v_add_nc_u32 v4, 192, v4");
    o("v_lshlrev_b32 v1, 3, v21");         // scale byte shift: 8 (t & 1)
    o("v_and_b32 v5, 1, v20");
    o("v_lshlrev_b32 v5, 1, v5");          // misalignment
    o("v_sub_nc_u32 v2, v2, v5");
    o("v_sub_nc_u32 v3, v3, v5");
    o("v_sub_nc_u32 v4, v4, v5");
    o("v_and_b32 v18, 3, v20");            // g & 3
    for (int k1 = 0; k1 < 2; k1++) {
        if (k1 == 0) o("v_lshlrev_b32 v19, 1, v22");
        else o("v_xor_b32 v19, 2, v19");
        o("v_or_b32 v23, v19, v21");
        o("v_lshl_or_b32 v23, v22, 3, v23");
        o("v_lshl_or_b32 v23, v20, 4, v23");
        o("v_lshlrev_b32 v23, 6, v23");
        for (int c = 0; c < 4; c++) {
            o("v_xor_b32 v15, " + S(c) + ", v18");
            o("v_lshl_add_u32 v" + S(7 + 4 * k1 + c) + ", v15, 4, v23");
        }
    }
    o("v_lshl_or_b32 v6, v20, 4, v21");
    o("v_lshl_or_b32 v6, v22, 3, v6");
    o("v_lshlrev_b32 v6, 2, v6");
    o("v_add_nc_u32 v6, " + S(SUMS) + ", v6");
    // staging: thread j's chunk j (segment j >> 2, chunk j & 3), and + 4096
    o("v_lshlrev_b32 v16, 4, v0");
    o("v_add_nc_u32 v17, 4096, v16");
    o("v_lshrrev_b32 v20, 2, v0");         // s
    o("v_bfe_u32 v21, v20, 3, 1");
    o("v_lshlrev_b32 v21, 1, v21");
    o("v_xor_b32 v22, v20, v21");          // segment slot
    o("v_bfe_u32 v23, v20, 4, 2");
    o("v_and_b32 v15, 3, v0");
    o("v_xor_b32 v15, v15, v23");
    o("v_lshlrev_b32 v15, 4, v15");
    o("v_lshl_add_u32 v15, v22, 6, v15");
    // sums: thread j adds segment j & 127 of token (j >> 7) + 2 r
    o("v_and_b32 v20, 127, v0");
    o("v_bfe_u32 v21, v20, 3, 1");
    o("v_lshlrev_b32 v21, 1, v21");
    o("v_xor_b32 v21, v20, v21");
    o("v_lshlrev_b32 v18, 6, v21");
    o("v_lshrrev_b32 v22, 7, v0");
    o("v_lshl_add_u32 v18, v22, 13, v18");
    o("v_lshlrev_b32 v19, 2, v20");
    o("v_lshl_add_u32 v19, v22, 9, v19");
    o("v_add_nc_u32 v19, " + S(SUMS) + ", v19");
    for (int i = 0; i < K * T; i++) o("v_mov_b32 " + V(ACC + i) + ", 0");
    o("v_lshrrev_b32 v20, 5, v0");
    o("v_readfirstlane_b32 s16, v20");
    o("s_lshl_b32 s17, s10, 3");
    o("s_add_u32 s17, s17, s16");
    o("s_mov_b32 s30, 0x76543210");
    o("s_mov_b32 s31, 0xfedcba98");
    o("s_mov_b32 s50, 0xc2000000");        // -32.0
    o("s_mov_b32 s18, 0");
    label("chunk");
    o("s_barrier");
    o("s_lshl_b32 s11, s18, 13");
    o("s_add_u32 s36, s2, s11");
    o("s_addc_u32 s37, s3, 0");
    for (int t = 0; t < T; t++) {
        if (t) {
            o("s_add_u32 s" + S(36 + 2 * t) + ", s" + S(34 + 2 * t) + ", s7");
            o("s_addc_u32 s" + S(37 + 2 * t) + ", s" + S(35 + 2 * t) + ", 0");
        }
        o("global_load_dwordx4 " + VR(VF + 8 * t, 4) + ", v16, s[" + S(36 + 2 * t) + ":" + S(37 + 2 * t) + "]");
        o("global_load_dwordx4 " + VR(VF + 8 * t + 4, 4) + ", v17, s[" + S(36 + 2 * t) + ":" + S(37 + 2 * t) + "]");
    }
    o("s_waitcnt vmcnt(0)");
    for (int t = 0; t < T; t++) {
        o("ds_write_b128 v15, " + VR(VF + 8 * t, 4) + " offset:" + S(8192 * t));
        o("ds_write_b128 v15, " + VR(VF + 8 * t + 4, 4) + " offset:" + S(8192 * t + 4096));
    }
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    for (int r = 0; 2 * r < T; r++) {
        const bool half = 2 * r + 1 >= T;
        if (half) {
            o("v_cmp_gt_u32 vcc_lo, 128, v0");
            o("s_and_saveexec_b32 s19, vcc_lo");
        }
        for (int c = 0; c < 4; c++) o("ds_read_b128 " + VR(VF + 4 * c, 4) + ", v18 offset:" + S(r * 16384 + 16 * c));
        o("s_waitcnt lgkmcnt(0)");
        o("v_add_f32 v20, " + V(VF) + ", " + V(VF + 1));
        for (int c = 2; c < 16; c++) o("v_add_f32 v20, v20, " + V(VF + c));
        o("ds_write_b32 v19, v20 offset:" + S(r * 1024));
        if (half) o("s_mov_b32 exec_lo, s19");
    }
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    auto loads = [&](int kk, int w) {
        const int sa = 22 + 4 * (kk % 2);
        o("s_mul_i32 s11, s8, " + S(kk));
        o("s_add_u32 s11, s11, s17");
        o("s_cmp_lt_u32 s11, s6");
        o("s_cselect_b32 s11, s11, 0");
        o("s_mul_i32 s" + S(sa) + ", s11, " + S(row_bytes));
        o("s_mul_hi_u32 s" + S(sa + 1) + ", s11, " + S(row_bytes));
        o("s_mul_i32 s12, s18, 1680");
        o("s_add_u32 s" + S(sa) + ", s" + S(sa) + ", s12");
        o("s_addc_u32 s" + S(sa + 1) + ", s" + S(sa + 1) + ", 0");
        o("s_add_u32 s" + S(sa) + ", s" + S(sa) + ", s0");
        o("s_addc_u32 s" + S(sa + 1) + ", s" + S(sa + 1) + ", s1");
        const std::string sp = "s[" + S(sa) + ":" + S(sa + 1) + "]";
        o("global_load_dwordx4 " + VR(w, 4) + ", v2, " + sp);
        o("global_load_dword " + V(w + 4) + ", v2, " + sp + " offset:16");
        o("global_load_dwordx4 " + VR(w + 5, 4) + ", v2, " + sp + " offset:32");
        o("global_load_dword " + V(w + 9) + ", v2, " + sp + " offset:48");
        o("global_load_dwordx4 " + VR(w + 10, 4) + ", v3, " + sp);
        o("global_load_dword " + V(w + 14) + ", v3, " + sp + " offset:16");
        o("global_load_dwordx3 " + VR(w + 15, 3) + ", v4, " + sp);
        o("v_nop");
    };
    // x of item i (token i / 4, quarter i % 4) into set i % 2: 16 values, then the sum
    auto xreads = [&](int i) {
        const int t = i / 4, q = i % 4, xr = XQ + 17 * (i % 2);
        for (int c = 0; c < 4; c++)
            o("ds_read_b128 " + VR(xr + 4 * c, 4) + ", v" + S(7 + 4 * (q & 1) + c) + " offset:" + S(8192 * t + 256 * (q >> 1)));
        o("ds_read_b32 " + V(xr + 16) + ", v6 offset:" + S(512 * t + 8 * q));
    };
    loads(0, WA);
    for (int kk = 0; kk < K; kk++) {
        const int w = kk % 2 ? WB : WA, A = w, B = w + 5, C = w + 10, SC = w + 15;
        if (kk + 1 < K) loads(kk + 1, kk % 2 ? WA : WB);
        o("s_waitcnt vmcnt(" + S(kk + 1 < K ? 7 : 0) + ")");
        o("s_mul_i32 s11, s8, " + S(kk));
        o("s_add_u32 s11, s11, s17");
        o("s_cmp_ge_u32 s11, s6");
        o("s_cbranch_scc1 skip" + S(kk));
        xreads(0);
        for (int r : {A, B, C})
            for (int i = 0; i < 4; i++) o("v_alignbyte_b32 " + V(r + i) + ", " + V(r + i + 1) + ", " + V(r + i) + ", v5");
        o("v_alignbyte_b32 " + V(SC) + ", " + V(SC + 1) + ", " + V(SC) + ", v5");
        o("v_alignbyte_b32 " + V(SC + 1) + ", " + V(SC + 2) + ", " + V(SC + 1) + ", v5");
        o("v_alignbyte_b32 " + V(SC + 2) + ", " + V(SC + 2) + ", " + V(SC + 2) + ", v5");
        o("v_nop");
        o("v_mov_b32_dpp " + ds(4) + ", " + V(SC + 2) + " quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0xf");
        o("v_lshrrev_b32 " + ds(5) + ", v1, " + V(SC));
        o("v_lshrrev_b32 " + ds(6) + ", v1, " + V(SC + 1));
        o("v_cvt_f32_f16 " + ds(4) + ", " + ds(4));
        o("v_bfe_i32 " + ds(0) + ", " + ds(5) + ", 0, 8");
        o("v_bfe_i32 " + ds(1) + ", " + ds(5) + ", 16, 8");
        o("v_bfe_i32 " + ds(2) + ", " + ds(6) + ", 0, 8");
        o("v_bfe_i32 " + ds(3) + ", " + ds(6) + ", 16, 8");
        for (int q = 0; q < 4; q++) {
            o("v_cvt_f32_i32 " + ds(q) + ", " + ds(q));
            o("v_mul_f32 " + ds(q) + ", " + ds(4) + ", " + ds(q));
        }
        // the 64 values as f32 (mv_q6k's unpacking), once for all tokens
        for (int q = 0; q < 4; q++) {
            const int L = (q & 1) ? B : A;
            for (int i = 0; i < 4; i++) {
                if (q == 0) o("v_lshlrev_b32 " + ds(7) + ", 4, " + V(C + i));
                else if (q == 1) o("v_lshlrev_b32 " + ds(7) + ", 2, " + V(C + i));
                else if (q == 3) o("v_lshrrev_b32 " + ds(7) + ", 2, " + V(C + i));
                o("v_and_b32 " + ds(7) + ", 0x30303030, " + (q == 2 ? V(C + i) : ds(7)));
                std::string lo = V(L + i);
                if (q >= 2) {
                    o("v_lshrrev_b32 " + ds(8) + ", 4, " + V(L + i));
                    lo = ds(8);
                }
                o("v_and_or_b32 " + ds(9) + ", " + lo + ", 0x0f0f0f0f, " + ds(7));
                for (int b = 0; b < 4; b++) o("v_cvt_f32_ubyte" + S(b) + " " + V(VF + 16 * q + 4 * i + b) + ", " + ds(9));
            }
        }
        for (int i = 0; i < 4 * T; i++) {
            const int t = i / 4, q = i % 4, xr = XQ + 17 * (i % 2);
            if (i + 1 < 4 * T) {
                xreads(i + 1);
                o("s_waitcnt lgkmcnt(5)");
            } else {
                o("s_waitcnt lgkmcnt(0)");
            }
            for (int j = 0; j < 16; j++) {
                if (j == 0) o("v_mul_f32 " + ds(11) + ", " + V(VF + 16 * q) + ", " + V(xr));
                else o("v_fmac_f32 " + ds(11) + ", " + V(VF + 16 * q + j) + ", " + V(xr + j));
            }
            o("v_fma_f32 " + ds(11) + ", " + V(xr + 16) + ", s50, " + ds(11));
            o("v_fmac_f32 " + V(ACC + T * kk + t) + ", " + ds(q) + ", " + ds(11));
        }
        label("skip" + S(kk));
    }
    o("s_add_u32 s18, s18, 1");
    o("s_cmp_lt_u32 s18, " + S(nit));
    o("s_cbranch_scc1 chunk");
    for (int kk = 0; kk < K; kk++) {
        o("s_mul_i32 s11, s8, " + S(kk));
        o("s_add_u32 s11, s11, s17");
        o("s_cmp_ge_u32 s11, s6");
        o("s_cbranch_scc1 end");
        o("v_nop");
        o("v_nop");
        for (int t = 0; t < T; t++) {
            const std::string a = V(ACC + T * kk + t);
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:1 row_mask:0xf bank_mask:0xf");
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:2 row_mask:0xf bank_mask:0xf");
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:4 row_mask:0xf bank_mask:0xf");
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:8 row_mask:0xf bank_mask:0xf");
            o("v_permlanex16_b32 v20, " + a + ", s30, s31");
            o("v_add_f32 " + a + ", " + a + ", v20");
        }
        o("s_lshl_b32 s11, s11, 2");
        o("s_mov_b32 s29, exec_lo");
        o("s_mov_b32 exec_lo, 1");
        o("s_mov_b32 s12, s4");
        o("s_mov_b32 s13, s5");
        for (int t = 0; t < T; t++) {
            if (t) {
                o("s_add_u32 s12, s12, s9");
                o("s_addc_u32 s13, s13, 0");
            }
            o("v_mov_b32 v20, s11");
            o("global_store_dword v20, " + V(ACC + T * kk + t) + ", s[12:13]");
            o("v_nop");
        }
        o("s_mov_b32 exec_lo, s29");
    }
    label("end");
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
