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
#include <string>

namespace ts5::direct {

// diag (timing only, results wrong): 1 drops the block math, 2 the LDS
// reads of x and sums, 4 the weight loads.
inline std::string gen_mv_q4k(int ncols, const std::string& hints = "slc dlc", int diag = 0) {
    const int nit = ncols / 2048;          // steps per row
    const int row_bytes = ncols / 256 * 144;
    const int nseg = ncols / 16;
    const int sums = ncols * 4;            // LDS byte offset of the 16-value sums
    const int lds = sums + nseg * 4;
    const std::string H = hints.empty() ? "" : " " + hints;
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    auto label = [&](const std::string& l) { k += l + ":\n"; };

    k += ".kernel mv_q4k\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(lds) + "\n";

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

    // ---- this wave's rows: r_begin + wave, step 8, below r_end
    o("v_lshrrev_b32 v7, 5, v0");
    o("v_readfirstlane_b32 s16, v7");
    o("s_mul_i32 s17, s10, s8");          // r_begin
    o("s_add_u32 s18, s17, s8");
    o("s_min_u32 s18, s18, s6");          // r_end
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
    o("v_cmp_le_u32 vcc_lo, 2, v110");    // the upper scales' packing (p >= 2), kept in vcc
    o("v_mov_b32 v6, 0");                 // the row's sum
    o("s_mov_b32 s30, 0x76543210");
    o("s_mov_b32 s31, 0xfedcba98");

    auto loads = [&](int base) {   // one step's four loads into v[base..base+15]
        if (diag & 4) return;
        o("global_load_dwordx4 v[" + S(base) + ":" + S(base + 3) + "], v1, s[22:23]" + H);
        o("global_load_dwordx4 v[" + S(base + 4) + ":" + S(base + 7) + "], v2, s[22:23]" + H);
        o("global_load_dwordx4 v[" + S(base + 8) + ":" + S(base + 11) + "], v1, s[22:23] offset:576" + H);
        o("global_load_dwordx4 v[" + S(base + 12) + ":" + S(base + 15) + "], v2, s[22:23] offset:576" + H);
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
        o("s_add_u32 s26, s26, 8");
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
        o("s_waitcnt vmcnt(4)");
        o("s_waitcnt lgkmcnt(10)");
        block(base, base + 4, 40, 72);
        o("s_waitcnt lgkmcnt(0)");
        block(base + 8, base + 12, 74, 106);
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
        o("s_add_u32 s34, s34, 8");
        o("s_cmp_ge_u32 s34, s18");
        o("s_cbranch_scc1 end");
        label("cnext" + n);
    };

    loads(8);
    advance_load();
    label("loop");
    loads(24);
    advance_load();
    compute(8);
    loads(8);
    advance_load();
    compute(24);
    o("s_branch loop");
    label("end");
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
