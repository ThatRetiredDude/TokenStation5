// mvt2_q4k: mvt_q4k (y_t = W x_t for T tokens, bit for bit mv_q4k's) with
// x read from LDS once a chunk and half-step instead of once a row slot.
//
// Round 100 measured mvt_q4k at about 0.55 of a one-token mat-vec for each
// token past the first: per row step a lane reads 2 x 34 floats of x per
// token from LDS, about half the one-token kernel's time already, and
// mvt_q4k did that again for every row slot. Here the items of a chunk go
// half-step first: for half g (blocks 4g..4g+3 of the step) every token's 34
// x floats are read into VGPRs once, then each of the K row slots loads only
// that half's 16 bytes (and the header dword), decodes it and adds into its
// lane sums. A lane sum still gets half 0 then half 1 of a step, steps in
// order, so the numbers are mv_q4k's. Weights load PD items ahead in NS = 4
// sets of 5 VGPRs (two items = one step's bytes, as mv_q4k keeps in flight).
//
// Arguments, rows and passes as mvt_q4k. K even; T up to 4 (x for 4 tokens
// takes 136 VGPRs).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ts5::direct {

inline std::string gen_mvt2_q4k(int ncols, int T, int K = 4, int PD = 2, bool resident = true) {
    const int NS = 4, J = 2 * K;          // weight sets; items a chunk: (half g, slot kk), g major
    if (T < 1 || T > 4 || K < 2 || K % 2 || PD < 1 || PD >= NS || PD > K) return "";
    const int nit = ncols / 2048;          // steps (chunks) per row
    const int row_bytes = ncols / 256 * 144;
    const int SUMS = T * 8192;             // LDS: x chunk of token t at 8192 t, its 128 sums at SUMS + 512 t
    // resident (round 110): where every chunk of x fits in LDS at once (4096
    // columns up to T 3), all of it is staged once at the start, chunk c at
    // c CB, and the passes read it with no staging, barriers or waits for
    // every load (each chunk of each pass otherwise staged x again behind an
    // s_waitcnt vmcnt(0) that also drained the weight loads in flight).
    const int CB = SUMS + T * 512;
    const bool res = resident && nit * CB <= 65536;
    const int lds = res ? std::max(nit * CB, 33 * 1024) : std::max(SUMS + T * 512, 33 * 1024);
    auto S = [](long v) { return std::to_string(v); };
    auto V = [&](int r) { return "v" + S(r); };
    auto VR = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto label = [&](const std::string& l) { k += l + ":\n"; };
    // VGPRs: v1 header dword offset, v2 nibble offset, v3 scale shift, v4-7 x
    // chunk addresses (c ^ p), v8 sums address, v9 staging LDS address, v10/v15
    // staging global offsets, v11 sums read, v12 sums write, v13-14 temps,
    // v16.. the lane sums (row slot k, token t: v16 + T k + t), then NS weight
    // sets of 5 (header dword, the half's 4 nibble dwords), the half's header
    // dwords (4), decode temps (14), the half's nibbles as f32 (32), and x of
    // the half for every token (34 each; the staging and sums use them too).
    const int ACC = 16, WS = ACC + K * T, HD = WS + 5 * NS, D = HD + 4, NF = D + 14, XR = NF + 32, XE = XR + 34 * T;
    if (XE > 256) return "";
    auto W = [&](int s) { return WS + 5 * s; };
    auto d = [&](int i) { return V(D + i); };   // d(0) = mv_q4k's v108 ... d(14) = v122
    k += ".kernel mvt2_q4k\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(lds) + "\n";
    // ---- lane constants (mv_q4k's)
    o("v_bfe_u32 v13, v0, 3, 2");          // grp
    o("v_mul_u32_u24 v14, 144, v13");
    o("v_and_b32 v12, 7, v0");             // th
    o("v_lshl_add_u32 v2, v12, 4, v14");
    o("v_add_nc_u32 v2, 16, v2");          // nibble offset
    o("v_bfe_u32 v3, v0, 1, 1");
    o("v_lshlrev_b32 v3, 4, v3");          // scale byte shift
    o("v_bfe_u32 v11, v0, 1, 2");          // p
    o("v_and_b32 v12, 1, v0");             // th & 1
    o("v_lshlrev_b32 v9, 8, v11");
    o("v_lshl_add_u32 v9, v12, 6, v9");
    o("v_lshl_add_u32 v9, v13, 10, v9");   // x bytes: 256 p + 64 (th & 1) + 1024 grp
    o("v_lshlrev_b32 v8, 4, v11");
    o("v_lshl_add_u32 v8, v12, 2, v8");
    o("v_lshl_add_u32 v8, v13, 6, v8");
    o("v_add_nc_u32 v8, " + S(SUMS) + ", v8");
    for (int c = 0; c < 4; c++) {
        o("v_xor_b32 v14, " + S(c) + ", v11");
        o("v_lshl_add_u32 v" + S(4 + c) + ", v14, 4, v9");
    }
    o("v_cmp_le_u32 vcc_lo, 2, v11");      // p >= 2, kept in vcc
    // header dword: as mv_q4k
    o("v_bfe_u32 v13, v0, 2, 2");
    o("v_and_b32 v14, 1, v13");
    o("v_lshrrev_b32 v13, 1, v13");
    o("v_lshl_add_u32 v14, v13, 2, v14");
    o("v_bfe_u32 v13, v0, 4, 1");
    o("v_lshl_add_u32 v14, v13, 1, v14");
    o("v_and_b32 v13, 3, v0");
    o("v_lshlrev_b32 v13, 2, v13");
    o("v_mad_u32_u24 v1, v14, 144, v13");
    // staging: thread j copies 16-byte pieces j and j + 256 of each token's chunk
    o("v_lshlrev_b32 v10, 4, v0");
    o("v_add_nc_u32 v15, 4096, v10");
    o("v_lshrrev_b32 v9, 4, v10");
    o("v_and_b32 v9, 0x30, v9");
    o("v_xor_b32 v9, v10, v9");            // swizzled as mv_q4k's
    // sums: thread j adds segment j & 127 of token (j >> 7) + 2 r
    o("v_and_b32 v13, 127, v0");
    o("v_lshrrev_b32 v14, 7, v0");
    o("v_lshlrev_b32 v11, 6, v13");
    o("v_lshl_add_u32 v11, v14, 13, v11");
    o("v_lshlrev_b32 v12, 2, v13");
    o("v_lshl_add_u32 v12, v14, 9, v12");
    o("v_add_nc_u32 v12, " + S(SUMS) + ", v12");
    o("v_lshrrev_b32 v13, 5, v0");
    o("v_readfirstlane_b32 s16, v13");      // wave in the workgroup
    o("s_lshl_b32 s17, s10, 3");
    o("s_add_u32 s17, s17, s16");          // W
    o("s_mov_b32 s30, 0x76543210");
    o("s_mov_b32 s31, 0xfedcba98");
    // ---- item j = (half g, slot kk) of a chunk: its weights into set j % NS
    // (J % NS == 0, so an item's set is the same in every chunk). The next
    // chunk's first items (or the next pass's) load during this chunk's last.
    auto loads = [&](int j, bool next) {
        const int g = j / K, kk = j % K, s = j % NS, sa = 22 + 4 * (j % 2);
        o("s_mul_i32 s11, s8, " + S(kk));
        o("s_add_u32 s11, s11, s17");
        o("s_add_u32 s11, s11, s20");
        if (next) {
            o("s_add_u32 s13, s18, 1");
            o("s_cmp_eq_u32 s13, " + S(nit));
            o("s_cselect_b32 s13, 0, s13");
            o("s_cselect_b32 s14, s15, 0");
            o("s_add_u32 s11, s11, s14");
        } else {
            o("s_mov_b32 s13, s18");
        }
        o("s_cmp_lt_u32 s11, s6");
        o("s_cselect_b32 s11, s11, 0");
        o("s_mul_i32 s" + S(sa) + ", s11, " + S(row_bytes));
        o("s_mul_hi_u32 s" + S(sa + 1) + ", s11, " + S(row_bytes));
        o("s_mul_i32 s12, s13, 1152");
        o("s_add_u32 s" + S(sa) + ", s" + S(sa) + ", s12");
        o("s_addc_u32 s" + S(sa + 1) + ", s" + S(sa + 1) + ", 0");
        o("s_add_u32 s" + S(sa) + ", s" + S(sa) + ", s0");
        o("s_addc_u32 s" + S(sa + 1) + ", s" + S(sa + 1) + ", s1");
        const std::string sp = "s[" + S(sa) + ":" + S(sa + 1) + "]";
        o("global_load_dword " + V(W(s)) + ", v1, " + sp);
        o("global_load_dwordx4 " + VR(W(s) + 1, 4) + ", v2, " + sp + (g ? " offset:576" : ""));
        o("v_nop");                        // (VMEM reads of s[sa:sa+1] before SALU writes them)
    };
    // ---- stage a chunk of every token (s36:37 its address) at LDS offset cb
    auto stage = [&](int cb) {
        for (int t = 0; t < T; t++) {
            if (t) {
                o("s_add_u32 s" + S(36 + 2 * t) + ", s" + S(34 + 2 * t) + ", s7");
                o("s_addc_u32 s" + S(37 + 2 * t) + ", s" + S(35 + 2 * t) + ", 0");
            }
            o("global_load_dwordx4 " + VR(XR + 8 * t, 4) + ", v10, s[" + S(36 + 2 * t) + ":" + S(37 + 2 * t) + "]");
            o("global_load_dwordx4 " + VR(XR + 8 * t + 4, 4) + ", v15, s[" + S(36 + 2 * t) + ":" + S(37 + 2 * t) + "]");
        }
        o("s_waitcnt vmcnt(0)");
        for (int t = 0; t < T; t++) {
            o("ds_write_b128 v9, " + VR(XR + 8 * t, 4) + " offset:" + S(cb + 8192 * t));
            o("ds_write_b128 v9, " + VR(XR + 8 * t + 4, 4) + " offset:" + S(cb + 8192 * t + 4096));
        }
        o("s_waitcnt lgkmcnt(0)");
        o("s_barrier");
        for (int r = 0; 2 * r < T; r++) {
            const bool half = 2 * r + 1 >= T;
            if (half) {
                o("s_mov_b32 s21, vcc_lo");            // (vcc holds p >= 2)
                o("v_cmp_gt_u32 vcc_lo, 128, v0");     // only token 2 r
                o("s_and_saveexec_b32 s19, vcc_lo");
            }
            for (int c = 0; c < 4; c++) o("ds_read_b128 " + VR(XR + 4 * c, 4) + ", v11 offset:" + S(cb + r * 16384 + 16 * c));
            o("s_waitcnt lgkmcnt(0)");
            o("v_add_f32 v13, " + V(XR) + ", " + V(XR + 1));
            for (int c = 2; c < 16; c++) o("v_add_f32 v13, v13, " + V(XR + c));
            o("ds_write_b32 v12, v13 offset:" + S(cb + r * 1024));
            if (half) {
                o("s_mov_b32 exec_lo, s19");
                o("s_mov_b32 vcc_lo, s21");
            }
        }
        o("s_waitcnt lgkmcnt(0)");
        o("s_barrier");
    };
    o("s_mul_i32 s15, s8, " + S(K));       // rows a pass of the dispatch takes
    o("s_mov_b32 s20, 0");                 // the pass's row offset
    o("s_mov_b32 s18, 0");
    for (int j = 0; j < PD; j++) loads(j, false);   // in flight while x is staged
    if (res)
        for (int c = 0; c < nit; c++) {
            o("s_add_u32 s36, s2, " + S(8192 * c));
            o("s_addc_u32 s37, s3, 0");
            stage(c * CB);
        }
    label("pass");
    for (int i = 0; i < K * T; i++) o("v_mov_b32 " + V(ACC + i) + ", 0");
    o("s_mov_b32 s18, 0");
    label("chunk");
    if (!res) {
        o("s_barrier");                    // every wave done with the last chunk
        o("s_lshl_b32 s11, s18, 13");
        o("s_add_u32 s36, s2, s11");
        o("s_addc_u32 s37, s3, 0");
        stage(0);
    }
    // ---- the chunk's items: half g of each row slot's step, x of the half
    // read once for all slots (round 101: mvt_q4k read it once a slot, and
    // the LDS reads cost about half a one-token mat-vec a token)
    for (int g = 0; g < 2; g++) {
        for (int t = 0; t < T; t++) {
            const int xr = XR + 34 * t;
            for (int c = 0; c < 8; c++)
                o("ds_read_b128 " + VR(xr + 4 * c, 4) + ", v" + S(4 + (c & 3)) + " offset:" + S(8192 * t + 4096 * g + (c < 4 ? 0 : 128)));
            o("ds_read_b32 " + V(xr + 32) + ", v8 offset:" + S(512 * t + 256 * g));
            o("ds_read_b32 " + V(xr + 33) + ", v8 offset:" + S(512 * t + 256 * g + 8));
        }
        for (int kk = 0; kk < K; kk++) {
            const int j = g * K + kk, w = W(j % NS), jn = j + PD;
            if (jn < J) loads(jn, false);
            else loads(jn - J, true);
            o("s_waitcnt vmcnt(" + S(2 * PD) + ")");
            o("s_mul_i32 s11, s8, " + S(kk));
            o("s_add_u32 s11, s11, s17");
            o("s_add_u32 s11, s11, s20");
            o("s_cmp_ge_u32 s11, s6");
            o("s_cbranch_scc1 skip" + S(j));
            for (int dd = 0; dd < 4; dd++) {
                const unsigned lo = (8u * g + dd) * 0x11111111u, hi = (8u * g + 4 + dd) * 0x11111111u;
                o("s_mov_b32 s48, " + S(lo));
                o("s_mov_b32 s49, " + S(hi));
                o("v_permlane16_b32 " + V(HD + dd) + ", " + V(w) + ", s48, s49");
            }
            {
                const int h = HD, q = w + 1;
                // scales and mins of the lane's two 32-value halves (mv_q4k's block())
                o("v_cvt_f32_f16 " + d(0) + ", " + V(h));
                o("v_cvt_f32_f16_sdwa " + d(1) + ", " + V(h) + " src0_sel:WORD_1");
                o("v_cndmask_b32 " + d(2) + ", " + V(h + 1) + ", " + V(h + 2) + ", vcc_lo");
                o("v_cndmask_b32 " + d(3) + ", " + V(h + 2) + ", " + V(h + 3) + ", vcc_lo");
                o("v_lshrrev_b32 " + d(2) + ", v3, " + d(2));
                o("v_lshrrev_b32 " + d(3) + ", v3, " + d(3));
                o("v_lshrrev_b32 " + d(4) + ", v3, " + V(h + 1));
                o("v_and_b32 " + d(5) + ", 63, " + d(2));
                o("v_and_b32 " + d(6) + ", 63, " + d(3));
                o("v_bfe_u32 " + d(7) + ", " + d(2) + ", 8, 6");
                o("v_bfe_u32 " + d(8) + ", " + d(3) + ", 8, 6");
                o("v_bfe_u32 " + d(9) + ", " + d(4) + ", 6, 2");
                o("v_and_b32 " + d(10) + ", 15, " + d(3));
                o("v_lshl_or_b32 " + d(9) + ", " + d(9) + ", 4, " + d(10));
                o("v_bfe_u32 " + d(10) + ", " + d(2) + ", 6, 2");
                o("v_bfe_u32 " + d(11) + ", " + d(3) + ", 4, 4");
                o("v_lshl_or_b32 " + d(10) + ", " + d(10) + ", 4, " + d(11));
                o("v_bfe_u32 " + d(11) + ", " + d(4) + ", 14, 2");
                o("v_bfe_u32 " + d(12) + ", " + d(3) + ", 8, 4");
                o("v_lshl_or_b32 " + d(11) + ", " + d(11) + ", 4, " + d(12));
                o("v_bfe_u32 " + d(12) + ", " + d(2) + ", 14, 2");
                o("v_bfe_u32 " + d(13) + ", " + d(3) + ", 12, 4");
                o("v_lshl_or_b32 " + d(12) + ", " + d(12) + ", 4, " + d(13));
                o("v_cndmask_b32 " + d(5) + ", " + d(5) + ", " + d(9) + ", vcc_lo");
                o("v_cndmask_b32 " + d(6) + ", " + d(6) + ", " + d(10) + ", vcc_lo");
                o("v_cndmask_b32 " + d(7) + ", " + d(7) + ", " + d(11) + ", vcc_lo");
                o("v_cndmask_b32 " + d(8) + ", " + d(8) + ", " + d(12) + ", vcc_lo");
                for (int r = 5; r <= 8; r++) o("v_cvt_f32_u32 " + d(r) + ", " + d(r));
                o("v_mul_f32 " + d(5) + ", " + d(0) + ", " + d(5));   // d sc, low
                o("v_mul_f32 " + d(6) + ", " + d(1) + ", " + d(6));   // dmin m, low
                o("v_mul_f32 " + d(7) + ", " + d(0) + ", " + d(7));   // high
                o("v_mul_f32 " + d(8) + ", " + d(1) + ", " + d(8));
                // the nibbles as f32, once for all tokens
                for (int c = 0; c < 4; c++) {
                    o("v_and_b32 " + d(11) + ", 0x0f0f0f0f, " + V(q + c));
                    o("v_lshrrev_b32 " + d(12) + ", 4, " + V(q + c));
                    o("v_and_b32 " + d(12) + ", 0x0f0f0f0f, " + d(12));
                    for (int b = 0; b < 4; b++) {
                        o("v_cvt_f32_ubyte" + S(b) + " " + V(NF + 4 * c + b) + ", " + d(11));
                        o("v_cvt_f32_ubyte" + S(b) + " " + V(NF + 16 + 4 * c + b) + ", " + d(12));
                    }
                }
                if (kk == 0) o("s_waitcnt lgkmcnt(0)");   // the half's x
                for (int t = 0; t < T; t++) {
                    const int xr = XR + 34 * t;
                    for (int jj = 0; jj < 16; jj++) {
                        if (jj == 0) {
                            o("v_mul_f32 " + d(9) + ", " + V(NF + jj) + ", " + V(xr + jj));
                            o("v_mul_f32 " + d(10) + ", " + V(NF + 16 + jj) + ", " + V(xr + 16 + jj));
                        } else {
                            o("v_fmac_f32 " + d(9) + ", " + V(NF + jj) + ", " + V(xr + jj));
                            o("v_fmac_f32 " + d(10) + ", " + V(NF + 16 + jj) + ", " + V(xr + 16 + jj));
                        }
                    }
                    const std::string a = V(ACC + T * kk + t);
                    o("v_fmac_f32 " + a + ", " + d(5) + ", " + d(9));
                    o("v_fma_f32 " + a + ", -" + d(6) + ", " + V(xr + 32) + ", " + a);
                    o("v_fmac_f32 " + a + ", " + d(7) + ", " + d(10));
                    o("v_fma_f32 " + a + ", -" + d(8) + ", " + V(xr + 33) + ", " + a);
                }
            }
            label("skip" + S(j));
            if (kk == 0) o("s_waitcnt lgkmcnt(0)");
        }
    }
    if (res)   // the next chunk's x and sums
        for (int r = 4; r <= 8; r++) o("v_add_nc_u32 v" + S(r) + ", " + S(CB) + ", v" + S(r));
    o("s_add_u32 s18, s18, 1");
    o("s_cmp_lt_u32 s18, " + S(nit));
    o("s_cbranch_scc1 chunk");
    if (res)
        for (int r = 4; r <= 8; r++) {
            char h[16];
            std::snprintf(h, sizeof h, "0x%x", (unsigned)(uint32_t)(-(int64_t)nit * CB));
            o("v_add_nc_u32 v" + S(r) + ", " + h + ", v" + S(r));
        }
    // ---- each slot's row: the 32 lanes' sums, lane 0 stores
    for (int kk = 0; kk < K; kk++) {
        o("s_mul_i32 s11, s8, " + S(kk));
        o("s_add_u32 s11, s11, s17");
        o("s_add_u32 s11, s11, s20");
        o("s_cmp_ge_u32 s11, s6");
        o("s_cbranch_scc1 stored");
        o("v_nop");
        o("v_nop");
        for (int t = 0; t < T; t++) {
            const std::string a = V(ACC + T * kk + t);
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:1 row_mask:0xf bank_mask:0xf");
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:2 row_mask:0xf bank_mask:0xf");
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:4 row_mask:0xf bank_mask:0xf");
            o("v_add_f32_dpp " + a + ", " + a + ", " + a + " row_xmask:8 row_mask:0xf bank_mask:0xf");
            o("v_permlanex16_b32 v13, " + a + ", s30, s31");
            o("v_add_f32 " + a + ", " + a + ", v13");
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
            o("v_mov_b32 v13, s11");
            o("global_store_dword v13, " + V(ACC + T * kk + t) + ", s[12:13]");
            o("v_nop");
        }
        o("s_mov_b32 exec_lo, s29");
    }
    label("stored");
    // the next pass while the workgroup's first row is a row
    o("s_add_u32 s20, s20, s15");
    o("s_lshl_b32 s32, s10, 3");
    o("s_add_u32 s32, s32, s20");
    o("s_cmp_lt_u32 s32, s6");
    o("s_cbranch_scc1 pass");
    o("s_waitcnt vmcnt(0)");
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
