// mv_q6k: y = W x for one token, W a Q6_K matrix (blocks of 256 values in
// 210 bytes: 128 bytes of low nibbles ql, 64 bytes of high bit pairs qh,
// 16 int8 scales, f16 d; value = d * scale * (q - 32), q = 6 bits), x and
// y f32.
//
// Same frame as mv_q4k: one workgroup of 256 threads per CU, rows dealt
// round-robin over every wave of the dispatch (wave v of workgroup w takes
// rows 8w + v, then + s8 each time), a wave works a whole row at a time,
// loads run one step ahead. Here 4 lanes share a block and each lane does 64
// values: lane t of the block takes half h = t >> 1 and l = 16 (t & 1) ..
// + 15 of it, which are ql[64h + l], ql[64h + 32 + l], qh[32h + l] (three
// 16-byte loads), and scales 8h + (t & 1) + {0, 2, 4, 6} (one 12-byte
// load at byte 192 + 8h; the h = 1 lanes' load also holds d, which the
// h = 0 lanes get by DPP). 8 blocks (1680 bytes, 2048 columns) per step.
//
// Blocks are only 2-byte aligned: block g of a step starts 2 (g & 1) bytes
// past a dword. Round 21 loaded at those addresses directly and every row
// came out wrong: the GPU drops the low address bits (a model of the kernel
// with every odd block read 2 bytes early reproduces its row 0 exactly,
// -1.14e8 vs -66.1), so loads start at the dword below
// and take one dword more, and v_alignbyte_b32 shifts the bytes into place.
//
// x sits in LDS with the sum of every 16 values next to it, so the "- 32"
// costs one multiply-add per 16 values: sum x (q - 32) = q.x - 32 sum(x).
// LDS layout: 16-value segment s at 64 * (s ^ (((s >> 3) & 1) << 1)), its
// 16-byte chunk c at 16 * (c ^ ((s >> 4) & 3)), so the 16 lanes of a
// half-wave reading the same chunk of their segments hit 16 different
// bank groups.
//
// Arguments: s[0:1] W, s[2:3] x, s[4:5] y, s6 rows, s7 records lo, s8 row
// step (workgroups * 8), s9 records hi; s10 workgroup id. W must be
// readable 2 bytes past its end (the last block's scale load).
#pragma once
#include <algorithm>
#include <string>

#include "mv_stage.h"

namespace ts5::direct {

// diag: 4 drops the weight loads (results wrong); 8 (results right) has
// every wave write s_memtime records as mv_q4k's diag 8 does.
inline std::string gen_mv_q6k(int ncols, int diag = 0, Stage st = Stage::None, float eps = 1e-6f) {
    const int nit = ncols / 2048;          // steps per row
    const int row_bytes = ncols / 256 * 210;
    const int nseg = ncols / 16;
    const int sums = ncols * 4;            // LDS byte offset of the 16-value sums
    // 32 more bytes for the staged norm's sums (mv_stage.h)
    const int lds = std::max(sums + nseg * 4, 33 * 1024) + (st != Stage::None ? 32 : 0);   // >= 33 KiB: one workgroup per CU (see mv_q4k)
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto label = [&](const std::string& l) { k += l + ":\n"; };

    k += ".kernel mv_q6k\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(lds) + "\n";
    if (diag & 8) {
        o("s_memtime s[38:39]");
        o("s_getreg_b32 s47, hwreg(HW_REG_HW_ID1)");
    }
    // ---- x into LDS: thread j copies 16-byte chunks j, j + 256, ...
    const int nload = ncols / 1024;
    o("v_lshlrev_b32 v7, 4, v0");
    o("s_mov_b32 s36, s2");
    o("s_mov_b32 s37, s3");
    for (int i = 0; i < nload; i++) {
        o("global_load_dwordx4 v[" + S(40 + 4 * i) + ":" + S(43 + 4 * i) + "], v7, s[36:37]");
        o("v_nop");
        o("s_add_u32 s36, s36, 4096");
        o("s_addc_u32 s37, s37, 0");
    }
    // chunk j: segment s = j >> 2 (+ 64 i, which leaves bits 3..5 alone), c = j & 3
    o("v_lshrrev_b32 v8, 2, v0");         // s
    o("v_bfe_u32 v9, v8, 3, 1");
    o("v_lshlrev_b32 v9, 1, v9");
    o("v_xor_b32 v10, v8, v9");           // segment slot
    o("v_bfe_u32 v11, v8, 4, 2");
    o("v_and_b32 v12, 3, v0");
    o("v_xor_b32 v12, v12, v11");         // chunk slot
    o("v_lshlrev_b32 v13, 4, v12");
    o("v_lshl_add_u32 v13, v10, 6, v13");
    if (st != Stage::None) k += gen_stage(st, ncols, eps, lds - 32, "s10");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < nload; i++)
        o("ds_write_b128 v13, v[" + S(40 + 4 * i) + ":" + S(43 + 4 * i) + "] offset:" + S(4096 * i));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    // ---- the sum of every 16 values: thread j takes segments j, j + 256, ...
    // (its four chunks in any order; segment j + 256 r sits 16384 r further)
    o("v_bfe_u32 v9, v0, 3, 1");
    o("v_lshlrev_b32 v9, 1, v9");
    o("v_xor_b32 v9, v0, v9");
    o("v_lshlrev_b32 v7, 6, v9");
    o("v_lshlrev_b32 v9, 2, v0");
    const int rounds = (nseg + 255) / 256;
    for (int r = 0; r < rounds; r++) {
        const int left = nseg - r * 256;
        if (left < 256) {
            o("v_cmp_gt_u32 vcc_lo, " + S(left) + ", v0");
            o("s_and_saveexec_b32 s15, vcc_lo");
        }
        for (int c = 0; c < 4; c++)
            o("ds_read_b128 v[" + S(40 + 4 * c) + ":" + S(43 + 4 * c) + "], v7 offset:" + S(r * 16384 + 16 * c));
        o("s_waitcnt lgkmcnt(0)");
        o("v_add_f32 v8, v40, v41");
        for (int c = 42; c < 56; c++) o("v_add_f32 v8, v8, v" + S(c));
        o("ds_write_b32 v9, v8 offset:" + S(sums + r * 1024));
        if (left < 256) o("s_mov_b32 exec_lo, s15");
    }
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    if (diag & 8) o("s_memtime s[42:43]");

    // ---- this wave's rows: 8 w + v, step s8
    o("v_lshrrev_b32 v7, 5, v0");
    o("v_readfirstlane_b32 s16, v7");
    o("s_lshl_b32 s17, s10, 3");
    o("s_mov_b32 s18, s6");
    o("s_mov_b32 s19, s8");
    o("s_add_u32 s34, s17, s16");         // compute cursor: row
    o("s_cmp_ge_u32 s34, s18");
    o("s_cbranch_scc1 end");
    o("s_mov_b32 s33, 0");                // compute cursor: step
    o("s_mov_b32 s26, s34");              // load cursor: row, step, address
    o("s_mov_b32 s27, 0");
    o("s_mul_i32 s22, s26, " + S(row_bytes));
    o("s_mul_hi_u32 s23, s26, " + S(row_bytes));
    o("s_add_u32 s22, s22, s0");
    o("s_addc_u32 s23, s23, s1");

    // ---- lane constants
    o("v_bfe_u32 v20, v0, 2, 3");         // g: which of the step's 8 blocks
    o("v_and_b32 v21, 1, v0");            // t & 1
    o("v_bfe_u32 v22, v0, 1, 1");         // h
    o("v_mul_u32_u24 v23, 210, v20");     // block offset
    o("v_lshlrev_b32 v24, 4, v21");       // 16 (t & 1)
    o("v_lshl_add_u32 v2, v22, 6, v24");
    o("v_add_nc_u32 v2, v2, v23");        // ql: 64h + 16 (t&1)
    o("v_lshl_add_u32 v3, v22, 5, v24");
    o("v_add_nc_u32 v3, v3, v23");
    o("v_add_nc_u32 v3, 128, v3");     // qh: 128 + 32h + 16 (t&1)
    o("v_lshl_add_u32 v4, v22, 3, v23");
    o("v_add_nc_u32 v4, 192, v4");        // scales: 192 + 8h
    o("v_lshlrev_b32 v24, 3, v21");       // scale byte shift: 8 (t&1)
    o("v_and_b32 v29, 1, v20");
    o("v_lshlrev_b32 v29, 1, v29");       // misalignment: 2 (g & 1) bytes
    o("v_sub_nc_u32 v2, v2, v29");
    o("v_sub_nc_u32 v3, v3, v29");
    o("v_sub_nc_u32 v4, v4, v29");
    // x: segment 16 (8 step + g) + 8h + 4 (k >> 1) + 2 (k & 1) + (t & 1); its
    // slot's bit 1 is (k & 1) ^ h, its chunk c sits at c ^ (g & 3). One
    // address per (k & 1, c); k >> 1 is the offset 256, the step 8192.
    o("v_and_b32 v25, 3, v20");           // g & 3
    for (int k1 = 0; k1 < 2; k1++) {
        if (k1 == 0) o("v_lshlrev_b32 v26, 1, v22");          // 2h
        else o("v_xor_b32 v26, 2, v26");                      // 2 (1 ^ h)
        o("v_or_b32 v27, v26, v21");
        o("v_lshl_or_b32 v27, v22, 3, v27");
        o("v_lshl_or_b32 v27, v20, 4, v27");                  // slot
        o("v_lshlrev_b32 v27, 6, v27");
        for (int c = 0; c < 4; c++) {
            o("v_xor_b32 v28, " + S(c) + ", v25");
            o("v_lshl_add_u32 v" + S(40 + 4 * k1 + c) + ", v28, 4, v27");
        }
    }
    o("v_lshl_or_b32 v25, v20, 4, v21");
    o("v_lshl_or_b32 v25, v22, 3, v25");
    o("v_lshlrev_b32 v25, 2, v25");
    o("v_add_nc_u32 v25, " + S(sums) + ", v25");   // sums: 4 (16 g + 8h + (t&1)), + 8k
    o("v_mov_b32 v6, 0");                 // the row's sum
    o("s_mov_b32 s30, 0x76543210");
    o("s_mov_b32 s31, 0xfedcba98");
    o("s_mov_b32 s50, 0xc2000000");       // -32.0

    // One step's loads into v[base .. base + 17], each from the dword at or
    // below its bytes: ql (2 x 20), qh (20), scales (12).
    auto loads = [&](int base) {
        if (diag & 4) return;
        o("global_load_dwordx4 v[" + S(base) + ":" + S(base + 3) + "], v2, s[22:23]");
        o("global_load_dword v" + S(base + 4) + ", v2, s[22:23] offset:16");
        o("global_load_dwordx4 v[" + S(base + 5) + ":" + S(base + 8) + "], v2, s[22:23] offset:32");
        o("global_load_dword v" + S(base + 9) + ", v2, s[22:23] offset:48");
        o("global_load_dwordx4 v[" + S(base + 10) + ":" + S(base + 13) + "], v3, s[22:23]");
        o("global_load_dword v" + S(base + 14) + ", v3, s[22:23] offset:16");
        o("global_load_dwordx3 v[" + S(base + 15) + ":" + S(base + 17) + "], v4, s[22:23]");
        o("v_nop");   // VMEMtoScalarWrite
    };
    int site = 0;
    auto advance_load = [&]() {
        const std::string n = S(site++);
        o("s_add_u32 s27, s27, 1");
        o("s_cmp_eq_u32 s27, " + S(nit));
        o("s_cbranch_scc1 lnew" + n);
        o("s_add_u32 s22, s22, 1680");
        o("s_addc_u32 s23, s23, 0");
        o("s_branch ldone" + n);
        label("lnew" + n);
        o("s_mov_b32 s27, 0");
        o("s_add_u32 s26, s26, s19");
        o("s_cmp_lt_u32 s26, s18");
        o("s_cselect_b32 s11, s26, 0");   // past the end: re-read row 0, never used
        o("s_mul_i32 s22, s11, " + S(row_bytes));
        o("s_mul_hi_u32 s23, s11, " + S(row_bytes));
        o("s_add_u32 s22, s22, s0");
        o("s_addc_u32 s23, s23, s1");
        label("ldone" + n);
    };
    auto V = [&](int r) { return "v" + S(r); };
    auto compute = [&](int base) {
        const std::string n = S(site++);
        const int A = base, B = base + 5, C = base + 10, SC = base + 15;
        o("s_lshl_b32 s11, s33, 13");
        for (int i = 0; i < 8; i++) o("v_add_nc_u32 v" + S(48 + i) + ", s11, v" + S(40 + i));
        o("s_lshl_b32 s11, s33, 9");
        o("v_add_nc_u32 v26, s11, v25");
        // per k: x chunks 0..3 into v[56 + 16k ..], then the sum into v[120 + k]
        for (int q = 0; q < 4; q++) {
            for (int c = 0; c < 4; c++)
                o("ds_read_b128 v[" + S(56 + 16 * q + 4 * c) + ":" + S(59 + 16 * q + 4 * c) + "], v" + S(48 + 4 * (q & 1) + c) +
                  " offset:" + S(256 * (q >> 1)));
            o("ds_read_b32 v" + S(120 + q) + ", v26 offset:" + S(8 * q));
        }
        o("s_waitcnt vmcnt(7)");
        for (int r : {A, B, C})   // bytes into place, in order (each reads the next, not yet shifted)
            for (int i = 0; i < 4; i++) o("v_alignbyte_b32 v" + S(r + i) + ", v" + S(r + i + 1) + ", v" + S(r + i) + ", v29");
        o("v_alignbyte_b32 v" + S(SC) + ", v" + S(SC + 1) + ", v" + S(SC) + ", v29");
        o("v_alignbyte_b32 v" + S(SC + 1) + ", v" + S(SC + 2) + ", v" + S(SC + 1) + ", v29");
        o("v_alignbyte_b32 v" + S(SC + 2) + ", v" + S(SC + 2) + ", v" + S(SC + 2) + ", v29");
        // d: from the block's h = 1 lanes (t = 2, 3), bytes 8..9 of their scale load
        o("v_mov_b32_dpp v164, " + V(SC + 2) + " quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0xf");
        o("v_lshrrev_b32 v165, v24, " + V(SC));
        o("v_lshrrev_b32 v166, v24, " + V(SC + 1));
        o("v_cvt_f32_f16 v164, v164");
        o("v_bfe_i32 v160, v165, 0, 8");
        o("v_bfe_i32 v161, v165, 16, 8");
        o("v_bfe_i32 v162, v166, 0, 8");
        o("v_bfe_i32 v163, v166, 16, 8");
        for (int q = 0; q < 4; q++) {
            o("v_cvt_f32_i32 v" + S(160 + q) + ", v" + S(160 + q));
            o("v_mul_f32 v" + S(160 + q) + ", v164, v" + S(160 + q));   // d * scale
        }
        for (int q = 0; q < 4; q++) {   // quarter q: values 32q + l: ql[(q & 1) ? B : A] nibble (q >> 1), qh bits 2q
            o("s_waitcnt lgkmcnt(" + S(20 - 5 * (q + 1)) + ")");
            const int L = (q & 1) ? B : A;
            for (int i = 0; i < 4; i++) {
                if (q == 0) o("v_lshlrev_b32 v168, 4, " + V(C + i));
                else if (q == 1) o("v_lshlrev_b32 v168, 2, " + V(C + i));
                else if (q == 3) o("v_lshrrev_b32 v168, 2, " + V(C + i));
                o("v_and_b32 v168, 0x30303030, " + (q == 2 ? V(C + i) : std::string("v168")));
                std::string lo = V(L + i);
                if (q >= 2) {
                    o("v_lshrrev_b32 v169, 4, " + V(L + i));
                    lo = "v169";
                }
                o("v_and_or_b32 v170, " + lo + ", 0x0f0f0f0f, v168");
                for (int b = 0; b < 4; b++) {
                    const int xi = 56 + 16 * q + 4 * i + b;
                    o("v_cvt_f32_ubyte" + S(b) + " v171, v170");
                    if (i == 0 && b == 0) o("v_mul_f32 v167, v171, v" + S(xi));
                    else o("v_fmac_f32 v167, v171, v" + S(xi));
                }
            }
            o("v_fma_f32 v167, v" + S(120 + q) + ", s50, v167");   // q.x - 32 sum(x)
            o("v_fmac_f32 v6, v" + S(160 + q) + ", v167");
        }
        o("s_add_u32 s33, s33, 1");
        o("s_cmp_eq_u32 s33, " + S(nit));
        o("s_cbranch_scc0 cnext" + n);
        // Row done: add the 32 lanes, lane 0 stores.
        o("v_nop");
        o("v_nop");
        o("v_add_f32_dpp v6, v6, v6 row_xmask:1 row_mask:0xf bank_mask:0xf");
        o("v_add_f32_dpp v6, v6, v6 row_xmask:2 row_mask:0xf bank_mask:0xf");
        o("v_add_f32_dpp v6, v6, v6 row_xmask:4 row_mask:0xf bank_mask:0xf");
        o("v_add_f32_dpp v6, v6, v6 row_xmask:8 row_mask:0xf bank_mask:0xf");
        o("v_permlanex16_b32 v7, v6, s30, s31");
        o("v_add_f32 v6, v6, v7");
        o("s_lshl_b32 s11, s34, 2");
        o("v_mov_b32 v7, s11");
        o("s_mov_b32 s29, exec_lo");
        o("s_mov_b32 exec_lo, 1");
        o("global_store_dword v7, v6, s[4:5]");
        o("s_mov_b32 exec_lo, s29");
        o("v_mov_b32 v6, 0");
        o("s_mov_b32 s33, 0");
        o("s_add_u32 s34, s34, s19");
        o("s_cmp_ge_u32 s34, s18");
        o("s_cbranch_scc1 end");
        label("cnext" + n);
    };
    // Two register sets in rotation: one step's loads in flight while the other is worked on.
    const int sets[2] = {124, 142};   // 18 registers each
    loads(sets[0]);
    advance_load();
    label("loop");
    for (int i = 0; i < 2; i++) {
        loads(sets[(i + 1) % 2]);
        advance_load();
        compute(sets[i]);
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
