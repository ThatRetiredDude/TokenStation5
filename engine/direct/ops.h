// The small kernels between a token's mat-vecs (Qwen3): RMS norm, the
// per-head Q/K norm + rotation + KV cache store, and SiLU(gate) * up. All f32
// activations; the KV cache is f16 [position][KV head][128].
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace ts5::direct {

inline std::string f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    char b[16];
    std::snprintf(b, sizeof b, "0x%08x", u);
    return b;
}

// y = x / sqrt(mean(x^2) + eps) * w, n values (a multiple of 1024), one
// workgroup of 256 threads, each n/256 consecutive values. add: first
// x += t (written back), the residual add after a layer's attention or FFN.
// Arguments: s[0:1] x, s[2:3] w, s[4:5] y, s[6:7] t (add only).
inline std::string gen_rmsnorm(int n, float eps, bool add = false) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int per = n / 256, nl = per / 4;   // values and 16-byte loads per thread
    k += std::string(".kernel ") + (add ? "add_rmsnorm" : "rmsnorm") + "\n.wave32\n.user_sgprs 8\n.lds 64\n";
    o("v_mul_u32_u24 v1, " + S(per * 4) + ", v0");
    for (int i = 0; i < nl; i++) o("global_load_dwordx4 v[" + S(8 + 4 * i) + ":" + S(11 + 4 * i) + "], v1, s[0:1] offset:" + S(16 * i));
    if (add)
        for (int i = 0; i < nl; i++)
            o("global_load_dwordx4 v[" + S(80 + 4 * i) + ":" + S(83 + 4 * i) + "], v1, s[6:7] offset:" + S(16 * i));
    for (int i = 0; i < nl; i++) o("global_load_dwordx4 v[" + S(8 + per + 4 * i) + ":" + S(11 + per + 4 * i) + "], v1, s[2:3] offset:" + S(16 * i));
    o("s_waitcnt vmcnt(" + S(nl) + ")");
    if (add) {
        for (int i = 0; i < per; i++) o("v_add_f32 v" + S(8 + i) + ", v" + S(8 + i) + ", v" + S(80 + i));
        for (int i = 0; i < nl; i++) o("global_store_dwordx4 v1, v[" + S(8 + 4 * i) + ":" + S(11 + 4 * i) + "], s[0:1] offset:" + S(16 * i));
    }
    o("v_mul_f32 v2, v8, v8");
    for (int i = 1; i < per; i++) o("v_fmac_f32 v2, v" + S(8 + i) + ", v" + S(8 + i));
    o("v_nop");
    o("v_nop");
    for (int x : {1, 2, 4, 8}) o("v_add_f32_dpp v2, v2, v2 row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
    o("s_mov_b32 s8, 0x76543210");
    o("s_mov_b32 s9, 0xfedcba98");
    o("v_permlanex16_b32 v3, v2, s8, s9");
    o("v_add_f32 v2, v2, v3");
    o("v_lshrrev_b32 v3, 5, v0");
    o("v_lshlrev_b32 v3, 2, v3");
    o("ds_write_b32 v3, v2");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_mov_b32 v3, 0");
    o("ds_read_b128 v[64:67], v3");
    o("ds_read_b128 v[68:71], v3 offset:16");
    o("s_waitcnt lgkmcnt(0)");
    o("v_add_f32 v2, v64, v65");
    for (int i = 66; i < 72; i++) o("v_add_f32 v2, v2, v" + S(i));
    o("v_mul_f32 v2, " + f32_bits(1.0f / n) + ", v2");
    o("v_add_f32 v2, " + f32_bits(eps) + ", v2");
    o("v_rsq_f32 v2, v2");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < per; i++) {
        o("v_mul_f32 v" + S(8 + i) + ", v2, v" + S(8 + i));
        o("v_mul_f32 v" + S(8 + i) + ", v" + S(8 + per + i) + ", v" + S(8 + i));
    }
    for (int i = 0; i < nl; i++) o("global_store_dwordx4 v1, v[" + S(8 + 4 * i) + ":" + S(11 + 4 * i) + "], s[4:5] offset:" + S(16 * i));
    o("s_endpgm");
    return k;
}

// Q and K heads: RMS norm over the head (128) with q_norm / k_norm, then the
// NEOX rotation (pairs i, i + 64) with cos/sin from a table the host writes
// per position; Q goes back in place (f32), K and V into the cache (f16).
// Workgroup h (64 threads, two waves, lane i takes i and i + 64): h < nq a
// Q head, else KV head h - nq.
// Arguments: s[0:1] q, s[2:3] k, s[4:5] v, s[6:7] q_norm, s[8:9] k_norm,
// s[10:11] cos[64] then sin[64], s[12:13] K cache row for this position,
// s[14:15] V cache row; s16 workgroup id.
inline std::string gen_qk_post(int nq, float eps) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    k += ".kernel qk_post\n.wave32\n.user_sgprs 16\n.tgid_x\n.lds 64\n";
    o("s_cmp_lt_u32 s16, " + S(nq));
    o("s_cselect_b32 s20, s0, s2");                 // the head's vector
    o("s_cselect_b32 s21, s1, s3");
    o("s_cselect_b32 s22, s6, s8");                 // its norm weights
    o("s_cselect_b32 s23, s7, s9");
    o("s_sub_u32 s25, s16, " + S(nq));
    o("s_cselect_b32 s25, s16, s25");               // head index within Q or K
    o("s_lshl_b32 s25, s25, 9");                    // * 512 bytes
    o("s_add_u32 s20, s20, s25");
    o("s_addc_u32 s21, s21, 0");
    o("v_lshlrev_b32 v1, 2, v0");                   // 4 i
    o("global_load_dword v2, v1, s[20:21]");
    o("global_load_dword v3, v1, s[20:21] offset:256");
    o("global_load_dword v4, v1, s[22:23]");
    o("global_load_dword v5, v1, s[22:23] offset:256");
    o("global_load_dword v6, v1, s[10:11]");        // cos
    o("global_load_dword v7, v1, s[10:11] offset:256");   // sin
    o("s_waitcnt vmcnt(4)");
    o("v_mul_f32 v8, v2, v2");
    o("v_fmac_f32 v8, v3, v3");
    o("v_nop");
    o("v_nop");
    for (int x : {1, 2, 4, 8}) o("v_add_f32_dpp v8, v8, v8 row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
    o("s_mov_b32 s26, 0x76543210");
    o("s_mov_b32 s27, 0xfedcba98");
    o("v_permlanex16_b32 v9, v8, s26, s27");
    o("v_add_f32 v8, v8, v9");
    o("v_lshrrev_b32 v9, 5, v0");
    o("v_lshlrev_b32 v9, 2, v9");
    o("ds_write_b32 v9, v8");
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("v_mov_b32 v9, 0");
    o("ds_read_b64 v[10:11], v9");
    o("s_waitcnt lgkmcnt(0)");
    o("v_add_f32 v8, v10, v11");
    o("v_mul_f32 v8, " + f32_bits(1.0f / 128) + ", v8");
    o("v_add_f32 v8, " + f32_bits(eps) + ", v8");
    o("v_rsq_f32 v8, v8");
    o("s_waitcnt vmcnt(0)");
    o("v_mul_f32 v2, v8, v2");
    o("v_mul_f32 v3, v8, v3");
    o("v_mul_f32 v2, v4, v2");                      // a, b: normed
    o("v_mul_f32 v3, v5, v3");
    o("v_mul_f32 v10, v3, v7");
    o("v_fma_f32 v10, v2, v6, -v10");               // a c - b s
    o("v_mul_f32 v11, v2, v7");
    o("v_fmac_f32 v11, v3, v6");                    // a s + b c
    o("s_cmp_lt_u32 s16, " + S(nq));
    o("s_cbranch_scc0 kv");
    o("global_store_dword v1, v10, s[20:21]");
    o("global_store_dword v1, v11, s[20:21] offset:256");
    o("s_endpgm");
    k += "kv:\n";
    // K into the cache, f16: head (s16 - nq) at + 256 bytes each
    o("s_sub_u32 s25, s16, " + S(nq));
    o("s_lshl_b32 s28, s25, 8");
    o("s_add_u32 s12, s12, s28");
    o("s_addc_u32 s13, s13, 0");
    o("s_add_u32 s14, s14, s28");
    o("s_addc_u32 s15, s15, 0");
    o("v_lshlrev_b32 v12, 1, v0");                  // 2 i
    o("v_cvt_f16_f32 v10, v10");
    o("v_cvt_f16_f32 v11, v11");
    o("global_store_short v12, v10, s[12:13]");
    o("global_store_short v12, v11, s[12:13] offset:128");
    // V: the same head of v, as it is
    o("s_lshl_b32 s25, s25, 9");
    o("s_add_u32 s4, s4, s25");
    o("s_addc_u32 s5, s5, 0");
    o("global_load_dword v2, v1, s[4:5]");
    o("global_load_dword v3, v1, s[4:5] offset:256");
    o("s_waitcnt vmcnt(0)");
    o("v_cvt_f16_f32 v2, v2");
    o("v_cvt_f16_f32 v3, v3");
    o("global_store_short v12, v2, s[14:15]");
    o("global_store_short v12, v3, s[14:15] offset:128");
    o("s_endpgm");
    return k;
}

// out = silu(g) * u, n values (a multiple of 1024): workgroups of 256
// threads, 4 values each. Arguments: s[0:1] g, s[2:3] u, s[4:5] out; s6
// workgroup id.
inline std::string gen_silu_mul() {
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    k += ".kernel silu_mul\n.wave32\n.user_sgprs 6\n.tgid_x\n";
    o("s_lshl_b32 s7, s6, 12");
    o("v_lshl_add_u32 v1, v0, 4, s7");
    o("global_load_dwordx4 v[2:5], v1, s[0:1]");
    o("global_load_dwordx4 v[6:9], v1, s[2:3]");
    o("s_waitcnt vmcnt(1)");
    for (int i = 0; i < 4; i++) {
        const std::string g = "v" + std::to_string(2 + i);
        o("v_mul_f32 v10, " + f32_bits(-1.4426950408889634f) + ", " + g);   // -g log2 e
        o("v_exp_f32 v10, v10");                    // e^-g
        o("v_add_f32 v10, 1.0, v10");
        o("v_rcp_f32 v10, v10");
        o("v_mul_f32 " + g + ", " + g + ", v10");
    }
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 4; i++) o("v_mul_f32 v" + std::to_string(2 + i) + ", v" + std::to_string(2 + i) + ", v" + std::to_string(6 + i));
    o("global_store_dwordx4 v1, v[2:5], s[4:5]");
    o("s_endpgm");
    return k;
}

// The index of the largest of n logits (n a multiple of 4), the lowest on a
// tie as the CPU's argmax, into a 64-bit slot by atomic max: high half the
// value as an ordered integer, low half ~index. The slot must be 0 before
// (embed resets it after reading). Each thread takes 4 runs of 4 values,
// runs past the end clamped onto the last (a repeat changes nothing), so
// the host launches ceil(n / 4096) workgroups of 256.
// Arguments: s[0:1] logits, s[2:3] slot, s4 n / 4 - 1, s5 threads launched; s6 workgroup id.
inline std::string gen_argmax() {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    k += ".kernel argmax\n.wave32\n.user_sgprs 6\n.tgid_x\n";
    o("s_lshl_b32 s7, s6, 8");
    o("v_add_nc_u32 v1, s7, v0");                   // g: the first run
    for (int i = 0; i < 4; i++) {
        o("v_min_u32 v" + S(2 + i) + ", s4, v1");   // run, clamped
        if (i < 3) o("v_add_nc_u32 v1, s5, v1");
    }
    for (int i = 0; i < 4; i++) {
        o("v_lshlrev_b32 v" + S(6 + i) + ", 4, v" + S(2 + i));
        o("global_load_dwordx4 v[" + S(12 + 4 * i) + ":" + S(15 + 4 * i) + "], v" + S(6 + i) + ", s[0:1]");
    }
    o("v_mov_b32 v40, 0");                          // best key (below every value)
    o("v_mov_b32 v41, -1");                         // its index
    for (int i = 0; i < 4; i++) {
        o("s_waitcnt vmcnt(" + S(3 - i) + ")");
        o("v_lshlrev_b32 v10, 2, v" + S(2 + i));    // the run's first index
        for (int c = 0; c < 4; c++) {
            const std::string v = "v" + S(12 + 4 * i + c);
            o("v_ashrrev_i32 v11, 31, " + v);
            o("v_or_b32 v11, 0x80000000, v11");
            o("v_xor_b32 " + v + ", " + v + ", v11");   // ordered: ~b below zero, b | 2^31 above
            o("v_cmp_gt_u32 vcc_lo, " + v + ", v40");
            o("v_cndmask_b32 v40, v40, " + v + ", vcc_lo");
            o("v_add_nc_u32 v11, " + S(c) + ", v10");
            o("v_cndmask_b32 v41, v41, v11, vcc_lo");
        }
    }
    // the wave's largest key, then the lowest index holding it
    o("v_mov_b32 v42, v40");
    o("v_nop");
    o("v_nop");
    for (int x : {1, 2, 4, 8}) o("v_max_u32_dpp v42, v42, v42 row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
    o("s_mov_b32 s8, 0x76543210");
    o("s_mov_b32 s9, 0xfedcba98");
    o("v_permlanex16_b32 v43, v42, s8, s9");
    o("v_max_u32 v42, v42, v43");
    o("v_cmp_eq_u32 vcc_lo, v42, v40");
    o("v_mov_b32 v44, -1");
    o("v_cndmask_b32 v44, v44, v41, vcc_lo");
    o("v_nop");
    o("v_nop");
    for (int x : {1, 2, 4, 8}) o("v_min_u32_dpp v44, v44, v44 row_xmask:" + S(x) + " row_mask:0xf bank_mask:0xf");
    o("v_permlanex16_b32 v43, v44, s8, s9");
    o("v_min_u32 v44, v44, v43");
    o("v_not_b32 v46, v44");
    o("v_mov_b32 v47, v42");
    o("v_mov_b32 v45, 0");
    o("s_mov_b32 exec_lo, 1");
    o("global_atomic_umax_x2 v45, v[46:47], s[2:3]");
    o("s_endpgm");
    return k;
}

// x = row `id` of a Q4_K table (ncols values, 16 per thread: ncols / 16
// threads in one workgroup, at most 1024). id: s6, or with s6 = -1 the
// argmax slot's (~ its low half), which is then reset to 0. The id is
// written to s[8:9] (the token list), so the host can read what was made.
// Arguments: s[0:1] table, s[2:3] x, s[4:5] slot, s6 id or -1, s[8:9] id out.
inline std::string gen_embed_q4k(int ncols) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int rb = ncols / 256 * 144;
    k += ".kernel embed_q4k\n.wave32\n.user_sgprs 10\n";
    o("s_load_dwordx2 s[12:13], s[4:5], 0x0");
    o("s_waitcnt lgkmcnt(0)");
    o("s_not_b32 s14, s12");
    o("s_cmp_eq_u32 s6, -1");
    o("s_cselect_b32 s14, s14, s6");               // the id
    o("s_mul_i32 s16, s14, " + S(rb));
    o("s_mul_hi_u32 s17, s14, " + S(rb));
    o("s_add_u32 s16, s16, s0");
    o("s_addc_u32 s17, s17, s1");                  // its row
    o("v_lshrrev_b32 v2, 4, v0");                   // block
    o("v_and_b32 v3, 15, v0");                      // sub: 16 values of it
    o("v_mul_u32_u24 v1, 144, v2");
    o("global_load_dword v4, v1, s[16:17]");        // d, dmin
    o("v_lshrrev_b32 v5, 1, v3");                   // is: the 32-value group
    o("v_add_nc_u32 v6, v1, v5");
    o("global_load_ubyte v7, v6, s[16:17] offset:4");    // sc[is]
    o("global_load_ubyte v8, v6, s[16:17] offset:8");    // sc[is + 4]
    o("v_mov_b32 v9, 4");
    o("v_cmp_le_u32 vcc_lo, 4, v5");
    o("v_cndmask_b32 v9, 0, v9, vcc_lo");
    o("v_sub_nc_u32 v9, v6, v9");
    o("global_load_ubyte v10, v9, s[16:17] offset:4");   // sc[is - 4] (sc[is] below 4)
    o("v_lshrrev_b32 v11, 2, v3");
    o("v_lshlrev_b32 v11, 5, v11");
    o("v_and_b32 v12, 1, v3");
    o("v_lshlrev_b32 v12, 4, v12");
    o("v_add3_u32 v11, v1, v11, v12");
    o("global_load_dwordx4 v[16:19], v11, s[16:17] offset:16");   // the 16 bytes of nibbles
    o("v_and_b32 v13, 1, v5");
    o("v_lshlrev_b32 v13, 2, v13");                 // nibble shift: 4 (is & 1)
    // the slot is read by every wave before one resets it
    o("s_barrier");
    o("v_mov_b32 v20, 0");
    o("v_mov_b32 v30, 0");
    o("v_mov_b32 v31, 0");
    o("v_mov_b32 v21, s14");
    o("v_cmp_eq_u32 vcc_lo, 0, v0");
    o("s_and_saveexec_b32 s22, vcc_lo");
    o("global_store_dwordx2 v20, v[30:31], s[4:5]");   // slot = 0
    o("global_store_dword v20, v21, s[8:9]");
    o("s_mov_b32 exec_lo, s22");
    o("s_waitcnt vmcnt(0)");
    // scale and min: below 4 sc[is] & 63, sc[is+4] & 63; else
    // (sc[is+4] & 15) | (sc[is-4] >> 6) << 4, (sc[is+4] >> 4) | (sc[is] >> 6) << 4
    o("v_and_b32 v22, 63, v7");
    o("v_and_b32 v23, 63, v8");
    o("v_and_b32 v24, 15, v8");
    o("v_lshrrev_b32 v25, 6, v10");
    o("v_lshl_or_b32 v24, v25, 4, v24");
    o("v_lshrrev_b32 v25, 4, v8");
    o("v_lshrrev_b32 v26, 6, v7");
    o("v_lshl_or_b32 v25, v26, 4, v25");
    o("v_cmp_le_u32 vcc_lo, 4, v5");
    o("v_cndmask_b32 v22, v22, v24, vcc_lo");
    o("v_cndmask_b32 v23, v23, v25, vcc_lo");
    o("v_cvt_f32_u32 v22, v22");
    o("v_cvt_f32_u32 v23, v23");
    o("v_cvt_f32_f16 v24, v4");
    o("v_lshrrev_b32 v25, 16, v4");
    o("v_cvt_f32_f16 v25, v25");
    o("v_mul_f32 v22, v24, v22");                   // d sc
    o("v_mul_f32 v23, v25, v23");                   // dmin m
    for (int i = 0; i < 4; i++) o("v_lshrrev_b32 v" + S(16 + i) + ", v13, v" + S(16 + i));
    for (int j = 0; j < 16; j++) {
        const std::string r = "v" + S(28 + j);
        o("v_bfe_u32 " + r + ", v" + S(16 + j / 4) + ", " + S(8 * (j % 4)) + ", 4");
        o("v_cvt_f32_u32 " + r + ", " + r);
        o("v_mul_f32 " + r + ", v22, " + r);
        o("v_sub_f32 " + r + ", " + r + ", v23");
    }
    o("v_lshlrev_b32 v1, 6, v0");
    for (int i = 0; i < 4; i++) o("global_store_dwordx4 v1, v[" + S(28 + 4 * i) + ":" + S(31 + 4 * i) + "], s[2:3] offset:" + S(16 * i));
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
