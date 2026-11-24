// mv_q4k2: y = W x for one token, W Q4_K (see mv_q4k.h for the block).
//
// Second layout. Round 12 showed mv_q4k's weight loads were its limit
// (about 230 GB/s against 436 for plain reads): 8 lanes shared each block,
// so half of every lane's loads were the same 16-byte header. Here 2 lanes
// share a block, each taking 64 bytes of nibbles (4 loads) and the header
// (1 load), so 4 of every 5 loads carry new bytes; a wave's step is 16
// blocks (2304 bytes, 4096 columns) and each lane decodes its four scales
// and mins once per step with whole-dword byte masks.
//
// A lane's values are 128 contiguous x (half a block): its four 32-value
// sub-blocks are read from LDS one at a time, the next while the current
// one is worked on. x sits in LDS with each 128-byte row's 16-byte chunks
// swizzled by bits 9..11 of the address (chunk c at c ^ ((addr >> 9) & 7)),
// since lane l starts at 512 l and would otherwise hit the same banks as
// every other lane. The sum of every 32 x values sits after x.
//
// Specialized on the column count (a multiple of 4096, at most 12288).
// Arguments as mv_q4k: s[0:1] W, s[2:3] x, s[4:5] y, s6 rows, s8 rows per
// workgroup; s10 workgroup id. diag & 8 writes the wave records as mv_q4k
// does (s7, s9: their address).
#pragma once
#include <algorithm>
#include <string>

namespace ts5::direct {

inline std::string gen_mv_q4k2(int ncols, const std::string& hints = "slc dlc", int diag = 0, int depth = 1) {
    const int nit = ncols / 4096;        // steps per row
    const int row_bytes = ncols / 256 * 144;
    const int nseg = ncols / 32;
    const int sums = ncols * 4;
    const int lds = std::max(sums + nseg * 4, 33 * 1024);
    const std::string H = hints.empty() ? "" : " " + hints;
    auto S = [](long v) { return std::to_string(v); };
    auto V = [&](int r) { return "v" + S(r); };
    auto VR = [&](int a, int n) { return "v[" + S(a) + ":" + S(a + n - 1) + "]"; };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto label = [&](const std::string& l) { k += l + ":\n"; };

    k += ".kernel mv_q4k2\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(lds) + "\n";
    if (diag & 8) {
        o("s_memtime s[38:39]");
        o("s_getreg_b32 s47, hwreg(HW_REG_HW_ID1)");
    }

    // ---- x into LDS, swizzled
    const int nload = ncols / 1024;
    o("v_lshlrev_b32 v7, 4, v0");
    o("s_mov_b32 s36, s2");
    o("s_mov_b32 s37, s3");
    for (int i = 0; i < nload; i++) {
        o("global_load_dwordx4 " + VR(40 + 4 * i, 4) + ", v7, s[36:37]");
        o("v_nop");
        o("s_add_u32 s36, s36, 4096");
        o("s_addc_u32 s37, s37, 0");
    }
    o("v_lshrrev_b32 v8, 5, v7");          // (addr >> 9) << 4
    o("v_and_b32 v8, 0x70, v8");
    o("v_xor_b32 v8, v7, v8");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < nload; i++) o("ds_write_b128 v8, " + VR(40 + 4 * i, 4) + " offset:" + S(4096 * i));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    // ---- sums of 32: thread j takes segment j (its 128-byte row, any chunk order)
    o("v_lshlrev_b32 v7, 7, v0");
    o("v_lshlrev_b32 v9, 2, v0");
    const int rounds = (nseg + 255) / 256;
    for (int r = 0; r < rounds; r++) {
        const int left = nseg - r * 256;
        if (left < 256) {
            o("v_cmp_gt_u32 vcc_lo, " + S(left) + ", v0");
            o("s_and_saveexec_b32 s15, vcc_lo");
        }
        for (int c = 0; c < 8; c++) o("ds_read_b128 " + VR(40 + 4 * c, 4) + ", v7 offset:" + S(r * 32768 + 16 * c));
        o("s_waitcnt lgkmcnt(0)");
        o("v_add_f32 v8, v40, v41");
        for (int c = 42; c < 72; c++) o("v_add_f32 v8, v8, " + V(c));
        o("ds_write_b32 v9, v8 offset:" + S(sums + r * 1024));
        if (left < 256) o("s_mov_b32 exec_lo, s15");
    }
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    if (diag & 8) o("s_memtime s[42:43]");

    // ---- rows: r_begin + wave, step 8
    o("v_lshrrev_b32 v7, 5, v0");
    o("v_readfirstlane_b32 s16, v7");
    o("s_mul_i32 s17, s10, s8");
    o("s_add_u32 s18, s17, s8");
    o("s_min_u32 s18, s18, s6");
    o("s_add_u32 s34, s17, s16");
    o("s_cmp_ge_u32 s34, s18");
    o("s_cbranch_scc1 end");
    o("s_mov_b32 s33, 0");
    o("s_mov_b32 s26, s34");
    o("s_mov_b32 s27, 0");
    o("s_mul_i32 s22, s26, " + S(row_bytes));
    o("s_mul_hi_u32 s23, s26, " + S(row_bytes));
    o("s_add_u32 s22, s22, s0");
    o("s_addc_u32 s23, s23, s1");

    // ---- lane constants
    o("v_and_b32 v14, 31, v0");            // lane
    o("v_lshrrev_b32 v15, 1, v14");        // block of the step
    o("v_mul_u32_u24 v1, 144, v15");       // header offset
    o("v_and_b32 v3, 1, v14");             // h: which half of the block
    o("v_lshl_add_u32 v2, v3, 6, v1");
    o("v_add_nc_u32 v2, 16, v2");          // nibbles: 16 + 64 h
    o("v_lshlrev_b32 v15, 9, v14");        // x bytes: 512 lane
    o("v_and_b32 v16, 7, v14");
    for (int c = 0; c < 8; c++) {          // chunk c's address: 512 lane + 16 (c ^ (lane & 7))
        o("v_xor_b32 v17, " + S(c) + ", v16");
        o("v_lshl_add_u32 " + V(4 + c) + ", v17, 4, v15");
    }
    o("v_lshlrev_b32 v12, 4, v14");
    o("v_add_nc_u32 v12, " + S(sums) + ", v12");   // sums: 16 lane
    o("v_cmp_ne_u32 vcc_lo, 0, v3");       // upper half: the other scale packing
    o("v_mov_b32 v13, 0");
    o("s_mov_b32 s30, 0x76543210");
    o("s_mov_b32 s31, 0xfedcba98");

    const int setb[4] = {16, 36, 56, 184};   // 20 registers each: header 4, nibbles 16
    auto loads = [&](int b) {
        if (diag & 4) return;
        o("global_load_dwordx4 " + VR(b, 4) + ", v1, s[22:23]" + H);
        for (int i = 0; i < 4; i++) o("global_load_dwordx4 " + VR(b + 4 + 4 * i, 4) + ", v2, s[22:23] offset:" + S(16 * i) + H);
        o("v_nop");
    };
    int site = 0;
    auto advance_load = [&]() {
        const std::string n = S(site++);
        o("s_add_u32 s27, s27, 1");
        o("s_cmp_eq_u32 s27, " + S(nit));
        o("s_cbranch_scc1 lnew" + n);
        o("s_add_u32 s22, s22, 2304");
        o("s_addc_u32 s23, s23, 0");
        o("s_branch ldone" + n);
        label("lnew" + n);
        o("s_mov_b32 s27, 0");
        o("s_add_u32 s26, s26, 8");
        o("s_cmp_lt_u32 s26, s18");
        o("s_cselect_b32 s11, s26, 0");
        o("s_mul_i32 s22, s11, " + S(row_bytes));
        o("s_mul_hi_u32 s23, s11, " + S(row_bytes));
        o("s_add_u32 s22, s22, s0");
        o("s_addc_u32 s23, s23, s1");
        label("ldone" + n);
    };
    const int XA = 80, XB = 112, SUM = 144, D = 148, M = 152;   // temps v156..v164, addresses v172..v180
    auto read_x = [&](int sub, int dst) {   // sub-block `sub`'s 32 x values
        if (diag & 2) return;
        for (int c = 0; c < 8; c++) o("ds_read_b128 " + VR(dst + 4 * c, 4) + ", " + V(172 + c) + " offset:" + S(128 * sub));
    };
    auto compute = [&](int b) {
        const std::string n = S(site++);
        o("s_lshl_b32 s11, s33, 14");
        for (int c = 0; c < 8; c++) o("v_add_nc_u32 " + V(172 + c) + ", s11, " + V(4 + c));
        o("s_lshl_b32 s11, s33, 9");
        o("v_add_nc_u32 v180, s11, v12");
        if (!(diag & 2)) o("ds_read_b128 " + VR(SUM, 4) + ", v180");
        read_x(0, XA);
        o("s_waitcnt vmcnt(" + S(5 * depth) + ")");
        if (!(diag & 1)) {
            // Scales and mins of the lane's four sub-blocks, as bytes.
            const int h1 = b + 1, h2 = b + 2, h3 = b + 3;
            o("v_and_b32 v156, 0x3f3f3f3f, " + V(h1));            // low half: sc = q[i] & 63
            o("v_and_b32 v157, 0x3f3f3f3f, " + V(h2));            //           m = q[i+4] & 63
            o("v_and_b32 v158, 0x0f0f0f0f, " + V(h3));            // high half: sc = q[i+8] & 15 | (q[i] >> 6) << 4
            o("v_lshrrev_b32 v159, 2, " + V(h1));
            o("v_and_or_b32 v158, v159, 0x30303030, v158");
            o("v_lshrrev_b32 v159, 4, " + V(h3));                 //            m = q[i+8] >> 4 | (q[i+4] >> 6) << 4
            o("v_and_b32 v159, 0x0f0f0f0f, v159");
            o("v_lshrrev_b32 v160, 2, " + V(h2));
            o("v_and_or_b32 v159, v160, 0x30303030, v159");
            o("v_cndmask_b32 v156, v156, v158, vcc_lo");
            o("v_cndmask_b32 v157, v157, v159, vcc_lo");
            o("v_cvt_f32_f16 v160, " + V(b));                      // d
            o("v_cvt_f32_f16_sdwa v161, " + V(b) + " src0_sel:WORD_1");   // dmin
            for (int i = 0; i < 4; i++) {
                o("v_cvt_f32_ubyte" + S(i) + " " + V(D + i) + ", v156");
                o("v_mul_f32 " + V(D + i) + ", v160, " + V(D + i));
                o("v_cvt_f32_ubyte" + S(i) + " " + V(M + i) + ", v157");
                o("v_mul_f32 " + V(M + i) + ", v161, " + V(M + i));
            }
        }
        for (int i = 0; i < 4; i++) {
            const int xr = (i & 1) ? XB : XA;
            if (i < 3) {
                read_x(i + 1, (i & 1) ? XA : XB);
                if (!(diag & 2)) o("s_waitcnt lgkmcnt(8)");
            } else if (!(diag & 2)) {
                o("s_waitcnt lgkmcnt(0)");
            }
            const int q0 = b + 4 + 8 * (i >> 1);   // chunk i>>1's 8 dwords; low nibbles for even i
            if (diag & 1) {
                o("v_add_f32 v13, " + V(xr) + ", v13");
                o("v_cvt_f32_u32 v162, " + V(q0));
                o("v_add_f32 v13, v162, v13");
                if (i == 0) {
                    o("v_add_f32 v13, " + V(SUM) + ", v13");
                    o("v_cvt_f32_u32 v162, " + V(b));
                    o("v_add_f32 v13, v162, v13");
                }
                continue;
            }
            for (int w = 0; w < 8; w++) {
                if (i & 1) {
                    o("v_lshrrev_b32 v162, 4, " + V(q0 + w));
                    o("v_and_b32 v162, 0x0f0f0f0f, v162");
                } else {
                    o("v_and_b32 v162, 0x0f0f0f0f, " + V(q0 + w));
                }
                for (int bb = 0; bb < 4; bb++) {
                    const int xi = 4 * w + bb;
                    o("v_cvt_f32_ubyte" + S(bb) + " v163, v162");
                    if (xi == 0) o("v_mul_f32 v164, v163, " + V(xr));
                    else o("v_fmac_f32 v164, v163, " + V(xr + xi));
                }
            }
            o("v_fmac_f32 v13, " + V(D + i) + ", v164");
            o("v_fma_f32 v13, -" + V(M + i) + ", " + V(SUM + i) + ", v13");
        }
        o("s_add_u32 s33, s33, 1");
        o("s_cmp_eq_u32 s33, " + S(nit));
        o("s_cbranch_scc0 cnext" + n);
        o("v_nop");
        o("v_nop");
        o("v_add_f32_dpp v13, v13, v13 row_xmask:1 row_mask:0xf bank_mask:0xf");
        o("v_add_f32_dpp v13, v13, v13 row_xmask:2 row_mask:0xf bank_mask:0xf");
        o("v_add_f32_dpp v13, v13, v13 row_xmask:4 row_mask:0xf bank_mask:0xf");
        o("v_add_f32_dpp v13, v13, v13 row_xmask:8 row_mask:0xf bank_mask:0xf");
        o("v_permlanex16_b32 v7, v13, s30, s31");
        o("v_add_f32 v13, v13, v7");
        o("s_lshl_b32 s11, s34, 2");
        o("v_mov_b32 v7, s11");
        o("s_mov_b32 s29, exec_lo");
        o("s_mov_b32 exec_lo, 1");
        o("global_store_dword v7, v13, s[4:5]");
        o("s_mov_b32 exec_lo, s29");
        o("v_mov_b32 v13, 0");
        o("s_mov_b32 s33, 0");
        o("s_add_u32 s34, s34, 8");
        o("s_cmp_ge_u32 s34, s18");
        o("s_cbranch_scc1 end");
        label("cnext" + n);
    };

    for (int i = 0; i < depth; i++) {
        loads(setb[i]);
        advance_load();
    }
    label("loop");
    for (int i = 0; i <= depth; i++) {
        loads(setb[(i + depth) % (depth + 1)]);
        advance_load();
        compute(setb[i]);
    }
    o("s_branch loop");
    label("end");
    if (diag & 8) {
        o("s_memtime s[44:45]");
        o("s_waitcnt vmcnt(0) lgkmcnt(0)");
        o("s_sub_u32 s42, s42, s38");
        o("s_sub_u32 s44, s44, s38");
        o("s_lshl_b32 s46, s10, 3");
        o("s_add_u32 s46, s46, s16");
        o("s_lshl_b32 s46, s46, 4");
        o("s_mov_b32 s40, s7");
        o("s_mov_b32 s41, s9");
        o("v_mov_b32 v7, s46");
        o("v_mov_b32 v8, s38");
        o("v_mov_b32 v9, s47");
        o("v_mov_b32 v10, s42");
        o("v_mov_b32 v11, s44");
        o("s_mov_b32 exec_lo, 1");
        o("global_store_dwordx4 v7, v[8:11], s[40:41]");
    }
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
