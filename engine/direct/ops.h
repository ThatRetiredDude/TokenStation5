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

}  // namespace ts5::direct
