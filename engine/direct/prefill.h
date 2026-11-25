// Kernels for reading a prompt T tokens at a time (the batched mat-vecs are
// mm_q4k.h and mm_q6k.h; the per-token steps reuse ops.h and attn.h).
#pragma once
#include <string>

namespace ts5::direct {

// to_xT: T tokens' rows of K values (Y, T x K, f32) to the prompt kernels'
// input: xT (K x T, a column's T values together), xs32 (K/32 x T, the sums
// of each 32 columns, for Q4_K) and xs16 (K/16 x T, for Q6_K).
// A workgroup of 256 takes C = 1024 / T columns of every token: each thread
// 4 columns of one token (rows read coalesced), the sums by DPP across the
// threads of a token, then through LDS to xT, 4 tokens a thread (one
// contiguous 4 KiB a workgroup). Round 43: the first version (256 columns
// a workgroup, 16-48 workgroups, sums by 16-32 serial LDS reads) took 24%
// of a prompt's time.
// Arguments: s[0:1] Y, s[2:3] xT, s[4:5] xs32, s[6:7] xs16; s8 workgroup id.
// Grid: K / C workgroups of 256. T is 8 or 16.
inline std::string gen_to_xT(int K, int T) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int LT = T == 16 ? 4 : 3, C = 1024 / T, TPT = 256 / T;   // columns a workgroup, threads a token
    const int LTPT = 8 - LT;
    k += ".kernel to_xT\n.wave32\n.user_sgprs 8\n.tgid_x\n.lds 4096\n";
    o("v_lshrrev_b32 v1, " + S(LTPT) + ", v0");          // t
    o("v_and_b32 v2, " + S(TPT - 1) + ", v0");           // q: columns 4q .. 4q + 3 of the workgroup's
    o("v_mul_u32_u24 v3, " + S(K * 4) + ", v1");
    o("v_lshl_add_u32 v3, v2, 4, v3");
    o("s_mul_i32 s9, s8, " + S(C * 4));
    o("s_add_u32 s10, s0, s9");
    o("s_addc_u32 s11, s1, 0");
    o("global_load_dwordx4 v[8:11], v3, s[10:11]");
    o("v_mul_u32_u24 v4, " + S(C * 4) + ", v1");         // LDS [t][C]
    o("v_lshl_add_u32 v4, v2, 4, v4");
    o("s_waitcnt vmcnt(0)");
    o("ds_write_b128 v4, v[8:11]");
    // sums: 4 columns, then 4 threads (16 columns), then 8 (32)
    o("v_add_f32 v12, v8, v9");
    o("v_add_f32 v13, v10, v11");
    o("v_add_f32 v12, v12, v13");
    o("v_nop");
    o("v_nop");
    o("v_add_f32_dpp v12, v12, v12 row_xmask:1 row_mask:0xf bank_mask:0xf");
    o("v_nop");
    o("v_nop");
    o("v_add_f32_dpp v12, v12, v12 row_xmask:2 row_mask:0xf bank_mask:0xf");
    o("v_nop");
    o("v_nop");
    o("v_add_f32_dpp v13, v12, v12 row_xmask:4 row_mask:0xf bank_mask:0xf");
    // xs16: lanes q % 4 == 0, at (wg C/16 + q/4) T + t
    o("s_mul_i32 s9, s8, " + S(C / 16 * T * 4));
    o("v_lshrrev_b32 v5, 2, v2");
    o("v_mul_u32_u24 v5, " + S(T * 4) + ", v5");
    o("v_lshl_add_u32 v5, v1, 2, v5");
    o("v_add_nc_u32 v5, s9, v5");
    o("v_and_b32 v6, 3, v2");
    o("v_cmp_eq_u32 vcc_lo, 0, v6");
    o("s_and_saveexec_b32 s12, vcc_lo");
    o("global_store_dword v5, v12, s[6:7]");
    o("s_mov_b32 exec_lo, s12");
    // xs32: lanes q % 8 == 0, at (wg C/32 + q/8) T + t
    o("s_mul_i32 s9, s8, " + S(C / 32 * T * 4));
    o("v_lshrrev_b32 v5, 3, v2");
    o("v_mul_u32_u24 v5, " + S(T * 4) + ", v5");
    o("v_lshl_add_u32 v5, v1, 2, v5");
    o("v_add_nc_u32 v5, s9, v5");
    o("v_and_b32 v6, 7, v2");
    o("v_cmp_eq_u32 vcc_lo, 0, v6");
    o("s_and_saveexec_b32 s12, vcc_lo");
    o("global_store_dword v5, v13, s[4:5]");
    o("s_mov_b32 exec_lo, s12");
    // xT: thread j takes column c = j / (T/4), tokens t0 = 4 (j % (T/4)) .. + 3
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_lshrrev_b32 v1, " + S(LT - 2) + ", v0");        // c
    o("v_and_b32 v2, " + S(T / 4 - 1) + ", v0");
    o("v_mul_u32_u24 v2, " + S(4 * C * 4) + ", v2");     // t0 C 4
    o("v_lshl_add_u32 v2, v1, 2, v2");
    for (int i = 0; i < 4; i++) o("ds_read_b32 v" + S(8 + i) + ", v2 offset:" + S(i * C * 4));
    o("s_lshl_b32 s9, s8, 12");
    o("s_add_u32 s10, s2, s9");
    o("s_addc_u32 s11, s3, 0");
    o("v_lshlrev_b32 v3, 4, v0");
    o("s_waitcnt lgkmcnt(0)");
    o("global_store_dwordx4 v3, v[8:11], s[10:11]");
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
