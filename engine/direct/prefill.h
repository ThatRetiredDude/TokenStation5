// Kernels for reading a prompt T tokens at a time (the batched mat-vecs are
// mm_q4k.h and mm_q6k.h; the per-token steps reuse ops.h and attn.h).
#pragma once
#include <string>

namespace ts5::direct {

// to_xT: T tokens' rows of K values (Y, T x K, f32) to the prompt kernels'
// input: xT (K x T, a column's T values together), xs32 (K/32 x T, the sums
// of each 32 columns, for Q4_K) and xs16 (K/16 x T, for Q6_K).
// Arguments: s[0:1] Y, s[2:3] xT, s[4:5] xs32, s[6:7] xs16; s8 workgroup id.
// Grid: K / 256 workgroups of 256. T is 8 or 16.
inline std::string gen_to_xT(int K, int T) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int LT = T == 16 ? 4 : 3;
    k += ".kernel to_xT\n.wave32\n.user_sgprs 8\n.tgid_x\n.lds " + S(T * 1024) + "\n";
    o("s_lshl_b32 s9, s8, 8");
    o("v_add_nc_u32 v1, s9, v0");          // the column
    o("v_lshlrev_b32 v2, 2, v1");
    for (int t = 0; t < T; t++) {
        o("s_add_u32 s10, s0, " + S((long)t * K * 4));
        o("s_addc_u32 s11, s1, 0");
        o("global_load_dword v" + S(10 + t) + ", v2, s[10:11]");
    }
    o("v_lshlrev_b32 v3, 2, v0");
    o("v_lshlrev_b32 v2, " + S(LT + 2) + ", v1");   // the column's T dwords in xT
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < T; i += 4) o("global_store_dwordx4 v2, v[" + S(10 + i) + ":" + S(13 + i) + "], s[2:3] offset:" + S(4 * i));
    for (int t = 0; t < T; t++) o("ds_write_b32 v3, v" + S(10 + t) + " offset:" + S(t * 1024));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    // Thread j: token t = j % T, group g = j / T; its sums' place is the
    // workgroup's first + j in both xs32 and xs16.
    o("v_and_b32 v4, " + S(T - 1) + ", v0");          // t
    o("v_lshrrev_b32 v5, " + S(LT) + ", v0");         // g
    o("v_lshlrev_b32 v4, 10, v4");                    // t 1024
    o("s_lshl_b32 s9, s8, " + S(LT + 4 + 2));         // the workgroup's 16 T sums, bytes
    o("v_add_nc_u32 v7, s9, v3");
    // xs16: threads j < 16 T, 16 columns from 16 g
    o("v_lshl_add_u32 v6, v5, 6, v4");
    o("v_cmp_gt_u32 vcc_lo, " + S(16 * T) + ", v0");
    o("s_and_saveexec_b32 s12, vcc_lo");
    o("s_cbranch_execz sums32");
    for (int i = 0; i < 16; i++) o("ds_read_b32 v" + S(40 + i) + ", v6 offset:" + S(4 * i));
    o("s_waitcnt lgkmcnt(0)");
    for (int i = 1; i < 16; i++) o("v_add_f32 v40, v40, v" + S(40 + i));
    o("global_store_dword v7, v40, s[6:7]");
    k += "sums32:\n";
    o("s_mov_b32 exec_lo, s12");
    // xs32: threads j < 8 T, 32 columns from 32 g
    o("s_lshr_b32 s9, s9, 1");
    o("v_add_nc_u32 v7, s9, v3");
    o("v_lshl_add_u32 v6, v5, 7, v4");
    o("v_cmp_gt_u32 vcc_lo, " + S(8 * T) + ", v0");
    o("s_and_saveexec_b32 s12, vcc_lo");
    o("s_cbranch_execz done");
    for (int i = 0; i < 32; i++) o("ds_read_b32 v" + S(40 + i) + ", v6 offset:" + S(4 * i));
    o("s_waitcnt lgkmcnt(0)");
    for (int i = 1; i < 32; i++) o("v_add_f32 v40, v40, v" + S(40 + i));
    o("global_store_dword v7, v40, s[4:5]");
    k += "done:\n";
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
