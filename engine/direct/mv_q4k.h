// mv_q4k: y = W x for one token, W a Q4_K matrix (blocks of 256 values in
// 144 bytes: f16 d, f16 dmin, 12 bytes of 6-bit scales and mins, 128 bytes
// of nibbles), x and y f32.
//
// Laid out from the read tests: one workgroup of 256 threads (8 wave32s)
// per CU, each workgroup streaming its own contiguous slice of rows. Inside
// it a wave takes a whole row at a time (rows r, r+8, ...), 8 lanes per
// block, 4 blocks side by side, two of those groups (1152 bytes) per step.
// Loads run one step ahead: while a step is worked on, the next one's four
// 16-byte loads per lane are in flight, across row ends too.
//
// x is copied into LDS once per workgroup, with the sum of every 16 values
// next to it, so a lane's 16 low and 16 high nibbles cost one convert and
// one fused multiply-add each: sum (d sc q - dmin m) x = d sc (q . x) - dmin m sum(x).
//
// Specialized on the column count (a multiple of 2048, at most 14336) so
// the staging and loops are fixed at assembly.
//
// Arguments: s[0:1] W, s[2:3] x, s[4:5] y, s6 rows, s7 unused, s8 rows per
// workgroup, s9 unused; s10 workgroup id.
#pragma once
#include <algorithm>
#include <string>

namespace ts5::direct {

// diag (timing only, results wrong): 1 drops the block math, 2 the LDS
// reads of x and sums, 4 the weight loads; 8 (results still right) has
// every wave write s_memtime at its start, after x is staged and at its
// end: 16 bytes per wave (start lo, HW_ID1, staged - start, end - start)
// at s[7], s[9] + 16 * (workgroup * 8 + wave).
// depth: steps whose loads are in flight while one is worked on (1 to 3).
// hints: none by default. Round 17: with "slc dlc" every 128-byte line two
// blocks share (nibbles start 16 bytes into a 144-byte block) came from
// memory twice, and the kernel ran at about 211 GB/s; with no hints it runs
// at 392-407 GB/s, within 7% of a plain read of the same bytes.
// inter: rows dealt round-robin over every wave of the dispatch (wave v of
// workgroup w takes rows 8w + v, then + s8 each time; the host passes s8 =
// workgroups * 8), so the whole GPU walks the matrix front to back together,
// as Vulkan's one-row workgroups do; otherwise each workgroup owns a slice of
// s8 rows.
inline std::string gen_mv_q4k(int ncols, const std::string& hints = "", int diag = 0, int depth = 1, bool inter = false) {
    const int nit = ncols / 2048;          // steps per row
    const int row_bytes = ncols / 256 * 144;
    const int nseg = ncols / 16;
    const int sums = ncols * 4;            // LDS byte offset of the 16-value sums
    // At least 33 KiB of LDS, so that no two workgroups share a CU (64 KiB
    // each in CU mode): with 17 KiB (4096 columns) the dispatcher packed
    // several onto some CUs and left others idle, and those shapes ran at
    // half the speed of 12288 columns' 52 KiB (round 8).
    const int lds = std::max(sums + nseg * 4, 33 * 1024);
    const std::string H = hints.empty() ? "" : " " + hints;
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto label = [&](const std::string& l) { k += l + ":\n"; };

    k += ".kernel mv_q4k\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(lds) + "\n";

    if (diag & 8) {
        o("s_memtime s[38:39]");
        o("s_getreg_b32 s47, hwreg(HW_REG_HW_ID1)");   // not s39: s_memtime writes s[38:39] later
    }
    // ---- x into LDS: every thread copies ncols/1024 runs of 16 bytes.
    const int nload = ncols / 1024;
    o("v_lshlrev_b32 v7, 4, v0");
    o("s_mov_b32 s36, s2");
    o("s_mov_b32 s37, s3");
    for (int i = 0; i < nload; i++) {
        o("global_load_dwordx4 v[" + S(40 + 4 * i) + ":" + S(43 + 4 * i) + "], v7, s[36:37]");
        o("v_nop");   // VMEM reads s[36:37]; the SALU write below waits for it
        o("s_add_u32 s36, s36, 4096");
        o("s_addc_u32 s37, s37, 0");
    }
    // LDS layout: 16-value segment s at s*64, its four 16-byte chunks
    // swizzled, chunk c at 16 * (c ^ ((s >> 2) & 3)) (address bits 5:4 xor
    // bits 9:8). The 8 lanes of a block read segments 4p + (0|1) (+2 for the
    // high nibbles), p = 0..3, which without the swizzle all sit at the
    // same two bank offsets: a 16-way conflict on every x read (round 6:
    // 41 GB/s).
    o("v_lshrrev_b32 v8, 4, v7");
    o("v_and_b32 v8, 0x30, v8");
    o("v_xor_b32 v8, v7, v8");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < nload; i++)
        o("ds_write_b128 v8, v[" + S(40 + 4 * i) + ":" + S(43 + 4 * i) + "] offset:" + S(4096 * i));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    // ---- the sum of every 16 values: thread j takes j, j+256, ...
    o("v_lshlrev_b32 v7, 6, v0");     // j*64: the segment's first byte
    o("v_lshlrev_b32 v9, 2, v0");     // j*4: its sum's slot
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

    // ---- this wave's rows: r_begin + wave, step 8, below r_end
    o("v_lshrrev_b32 v7, 5, v0");
    o("v_readfirstlane_b32 s16, v7");
    if (inter) {
        o("s_lshl_b32 s17, s10, 3");
        o("s_mov_b32 s18, s6");           // r_end: the matrix's end
        o("s_mov_b32 s19, s8");           // row step
    } else {
        o("s_mul_i32 s17, s10, s8");      // r_begin
        o("s_add_u32 s18, s17, s8");
        o("s_min_u32 s18, s18, s6");      // r_end
        o("s_mov_b32 s19, 8");
    }
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
    o("v_bfe_u32 v108, v0, 3, 2");        // grp: which of the 4 blocks
    o("v_mul_u32_u24 v1, 144, v108");     // header offset
    o("v_and_b32 v109, 7, v0");           // th: which 16 bytes of nibbles
    o("v_lshl_add_u32 v2, v109, 4, v1");
    o("v_add_nc_u32 v2, 16, v2");         // nibble offset
    o("v_bfe_u32 v3, v0, 1, 1");
    o("v_lshlrev_b32 v3, 4, v3");         // scale byte shift: 16 * (p & 1)
    o("v_bfe_u32 v110, v0, 1, 2");        // p = th >> 1: which 64 values
    o("v_and_b32 v111, 1, v0");           // which 16 of the 32
    o("v_lshlrev_b32 v4, 8, v110");
    o("v_lshl_add_u32 v4, v111, 6, v4");
    o("v_lshl_add_u32 v4, v108, 10, v4"); // x bytes: 256 p + 64 (th&1) + 1024 grp
    o("v_lshlrev_b32 v5, 4, v110");
    o("v_lshl_add_u32 v5, v111, 2, v5");
    o("v_lshl_add_u32 v5, v108, 6, v5");
    o("v_add_nc_u32 v5, " + S(sums) + ", v5");  // sums bytes
    for (int c = 0; c < 4; c++) {   // x chunk c of the lane's segments, swizzled: v4 + 16 (c ^ p)
        o("v_xor_b32 v112, " + S(c) + ", v110");
        o("v_lshl_add_u32 v" + S(126 + c) + ", v112, 4, v4");
    }
    // Headers: one dword load per step brings the step's 8 headers (round
    // 13: loads, not math, set the pace, and a 16-byte header loaded by all
    // 8 lanes of its block cost a whole load instruction per block). In
    // each 16-lane row, lane i holds dword (i & 3) of block
    // [2r, 2r+1, 2r+4, 2r+5][(i >> 2) & 3], r the row: exactly the blocks
    // that row's lanes work on, so v_permlane16 hands each lane its block's
    // four dwords.
    o("v_bfe_u32 v112, v0, 2, 2");        // j
    o("v_and_b32 v113, 1, v112");
    o("v_lshrrev_b32 v112, 1, v112");
    o("v_lshl_add_u32 v113, v112, 2, v113");   // (j & 1) + 4 (j >> 1)
    o("v_bfe_u32 v112, v0, 4, 1");        // r
    o("v_lshl_add_u32 v113, v112, 1, v113");   // block
    o("v_and_b32 v112, 3, v0");
    o("v_lshlrev_b32 v112, 2, v112");
    o("v_mad_u32_u24 v135, v113, 144, v112");  // header dword offset
    o("v_cmp_le_u32 vcc_lo, 2, v110");    // the upper scales' packing (p >= 2), kept in vcc
    o("v_mov_b32 v6, 0");                 // the row's sum
    o("s_mov_b32 s30, 0x76543210");
    o("s_mov_b32 s31, 0xfedcba98");

    // One step's loads into v[base..base+8]: the header dword, then each
    // group's 16 bytes of nibbles.
    auto loads = [&](int base) {
        if (diag & 4) return;
        o("global_load_dword v" + S(base) + ", v135, s[22:23]" + H);
        o("global_load_dwordx4 v[" + S(base + 1) + ":" + S(base + 4) + "], v2, s[22:23]" + H);
        o("global_load_dwordx4 v[" + S(base + 5) + ":" + S(base + 8) + "], v2, s[22:23] offset:576" + H);
        o("v_nop");   // VMEMtoScalarWrite: the loads read s[22:23] before it changes
    };
    int site = 0;
    auto advance_load = [&]() {
        const std::string n = S(site++);
        o("s_add_u32 s27, s27, 1");
        o("s_cmp_eq_u32 s27, " + S(nit));
        o("s_cbranch_scc1 lnew" + n);
        o("s_add_u32 s22, s22, 1152");
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
    // One block: header h[0:3], nibbles q[0:3], x low at xr (16), high at xr+16, sums at sr, sr+1.
    auto block = [&](int h, int q, int xr, int sr) {
        auto V = [&](int r) { return "v" + S(r); };
        if (diag & 1) {   // keep every input alive, nothing else
            o("v_add3_u32 v119, " + V(h) + ", " + V(q) + ", " + V(q + 3));
            o("v_add_f32 v6, " + V(xr) + ", v6");
            o("v_add_f32 v6, " + V(sr) + ", v6");
            o("v_cvt_f32_u32 v119, v119");
            o("v_add_f32 v6, v119, v6");
            return;
        }
        o("v_cvt_f32_f16 v108, " + V(h));
        o("v_cvt_f32_f16_sdwa v109, " + V(h) + " src0_sel:WORD_1");
        o("v_cndmask_b32 v110, " + V(h + 1) + ", " + V(h + 2) + ", vcc_lo");   // A
        o("v_cndmask_b32 v111, " + V(h + 2) + ", " + V(h + 3) + ", vcc_lo");   // B
        o("v_lshrrev_b32 v110, v3, v110");
        o("v_lshrrev_b32 v111, v3, v111");
        o("v_lshrrev_b32 v112, v3, " + V(h + 1));                               // C
        o("v_and_b32 v113, 63, v110");
        o("v_and_b32 v114, 63, v111");
        o("v_bfe_u32 v115, v110, 8, 6");
        o("v_bfe_u32 v116, v111, 8, 6");
        o("v_bfe_u32 v117, v112, 6, 2");
        o("v_and_b32 v118, 15, v111");
        o("v_lshl_or_b32 v117, v117, 4, v118");
        o("v_bfe_u32 v118, v110, 6, 2");
        o("v_bfe_u32 v119, v111, 4, 4");
        o("v_lshl_or_b32 v118, v118, 4, v119");
        o("v_bfe_u32 v119, v112, 14, 2");
        o("v_bfe_u32 v120, v111, 8, 4");
        o("v_lshl_or_b32 v119, v119, 4, v120");
        o("v_bfe_u32 v120, v110, 14, 2");
        o("v_bfe_u32 v121, v111, 12, 4");
        o("v_lshl_or_b32 v120, v120, 4, v121");
        o("v_cndmask_b32 v113, v113, v117, vcc_lo");
        o("v_cndmask_b32 v114, v114, v118, vcc_lo");
        o("v_cndmask_b32 v115, v115, v119, vcc_lo");
        o("v_cndmask_b32 v116, v116, v120, vcc_lo");
        for (int r = 113; r <= 116; r++) o("v_cvt_f32_u32 v" + S(r) + ", v" + S(r));
        o("v_mul_f32 v113, v108, v113");   // d sc, low
        o("v_mul_f32 v114, v109, v114");   // dmin m, low
        o("v_mul_f32 v115, v108, v115");   // high
        o("v_mul_f32 v116, v109, v116");
        for (int c = 0; c < 4; c++) {
            o("v_and_b32 v119, 0x0f0f0f0f, " + V(q + c));
            o("v_lshrrev_b32 v120, 4, " + V(q + c));
            o("v_and_b32 v120, 0x0f0f0f0f, v120");
            for (int b = 0; b < 4; b++) {
                const int i = 4 * c + b;
                o("v_cvt_f32_ubyte" + S(b) + " v121, v119");
                o("v_cvt_f32_ubyte" + S(b) + " v122, v120");
                if (i == 0) {
                    o("v_mul_f32 v117, v121, " + V(xr + i));
                    o("v_mul_f32 v118, v122, " + V(xr + 16 + i));
                } else {
                    o("v_fmac_f32 v117, v121, " + V(xr + i));
                    o("v_fmac_f32 v118, v122, " + V(xr + 16 + i));
                }
            }
        }
        o("v_fmac_f32 v6, v113, v117");
        o("v_fma_f32 v6, -v114, " + V(sr) + ", v6");
        o("v_fmac_f32 v6, v115, v118");
        o("v_fma_f32 v6, -v116, " + V(sr + 1) + ", v6");
    };
    auto compute = [&](int base) {   // the step whose loads sit in v[base..base+15]
        const std::string n = S(site++);
        o("s_lshl_b32 s11, s33, 13");
        for (int c = 0; c < 4; c++) o("v_add_nc_u32 v" + S(130 + c) + ", s11, v" + S(126 + c));
        o("s_lshl_b32 s11, s33, 9");
        o("v_add_nc_u32 v125, s11, v5");
        for (int blk = 0; blk < 2 && !(diag & 2); blk++) {
            const int xr = blk ? 74 : 40, sr = blk ? 106 : 72, xo = blk * 4096, so = blk * 256;
            for (int c = 0; c < 8; c++)
                o("ds_read_b128 v[" + S(xr + 4 * c) + ":" + S(xr + 4 * c + 3) + "], v" + S(130 + (c & 3)) + " offset:" +
                  S(xo + (c < 4 ? 0 : 128)));
            o("ds_read_b32 v" + S(sr) + ", v125 offset:" + S(so));
            o("ds_read_b32 v" + S(sr + 1) + ", v125 offset:" + S(so + 8));
        }
        o("s_waitcnt vmcnt(" + S(3 * depth) + ")");
        for (int g = 0; g < 2; g++)
            for (int d = 0; d < 4; d++) {   // group g's header dword d from lane 4 (2g + (i >= 8)) + d of the row
                const unsigned lo = (8u * g + d) * 0x11111111u, hi = (8u * g + 4 + d) * 0x11111111u;
                o("s_mov_b32 s48, " + S(lo));
                o("s_mov_b32 s49, " + S(hi));
                o("v_permlane16_b32 v" + S(184 + 4 * g + d) + ", v" + S(base) + ", s48, s49");
            }
        o("s_waitcnt lgkmcnt(10)");
        block(184, base + 1, 40, 72);
        o("s_waitcnt lgkmcnt(0)");
        block(188, base + 5, 74, 106);
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

    // depth+1 register sets in rotation; each compute waits only for the
    // oldest set while the other `depth` sets' loads stay in flight.
    const int sets[4] = {8, 17, 26, 150};   // 9 registers each
    for (int i = 0; i < depth; i++) {
        loads(sets[i]);
        advance_load();
    }
    label("loop");
    for (int i = 0; i <= depth; i++) {
        loads(sets[(i + depth) % (depth + 1)]);
        advance_load();
        compute(sets[i]);
    }
    o("s_branch loop");
    label("end");
    if (diag & 8) {
        o("s_memtime s[44:45]");
        o("s_waitcnt vmcnt(0) lgkmcnt(0)");   // prefetched loads into v[8:11] must land before they're reused
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
