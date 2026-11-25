// mm_q6k_h: mm_q6k_sb with packed f16 multiply-adds and x16 input, as
// mm_q4k_h.h (round 52: 1.5-2x faster than f32 on Q4_K). Each group's 16
// products are summed in f16, then into f32 with its scale and the -32
// offset (xs16, f32). x16 is scaled by 2^-E per 32 columns and token so
// that |x| < 16 (round 53: unscaled, Qwen3's activations overflowed f16's
// 65504 and gave NaN); E is xs16's low byte (to_xT), undone exactly here. The layout below is mm_q6k_sb's.
//
// mm_q6k_sb: the Q6_K prompt kernel with each block split across the
// workgroup's 4 waves, as mm_q4k_sb.h (round 49: sharing the x lines made
// the Q4_K kernel 10-38% faster and halved its VGPRs).
//
// A Q6_K block (210 bytes): half n (values 128 n ..) has ql at 64 n (64
// bytes), qh at 128 + 32 n (32), int8 scales at 192 (16, one per 16
// values), d (f16) at 208. Value l + 32 qq of a half (l < 32) is the low
// (qq 0, 1) or high (qq 2, 3) nibble of ql[l + 32 (qq & 1)] with bits
// 2 qq of qh[l] above it, minus 32. Wave w takes half n = w / 2 and
// l = 16 i .. 16 i + 15 (i = w % 2): 48 bytes (ql[l], ql[l + 32], qh[l])
// for 64 values in four groups of 16 (qq), each with its own scale
// (scale byte 8 n + i + 2 qq).
//
// Odd blocks start 2 bytes into a dword (and the GPU rounds an unaligned
// load's address down, round 21), so each piece is loaded as 5 dwords from
// the aligned start and shifted down 2 bytes when the block is odd.
//
// Arguments as mm_q6k: s[0:1] W, s[2:3] xT (K x T), s[4:5] Y (T x R), s6 R,
// s[8:9] xs (K/16 x T); s10 workgroup id. Grid: R / 32 workgroups of 128.
#pragma once
#include <string>

namespace ts5::direct {

inline std::string gen_mm_q6k_h(int K, int T) {
    auto S = [](long v) { return std::to_string(v); };
    std::string k;
    auto o = [&](const std::string& line) { k += "    " + line + "\n"; };
    const int NB = K / 256, RB = NB * 210, P = T / 2;
    const int A = 8, B = 28;                     // 4 pieces of 5 dwords: ql[l], ql[l + 32], qh[l], scales + d
    const int ACC = 48, TMP = 48 + T, U = TMP + P, DSC = U + 8, M32 = U + 12;
    const int SX = 20, SXS = 84;
    k += ".kernel mm_q6k_h\n.wave32\n.user_sgprs 10\n.tgid_x\n.lds " + S(4 * T * 128) + "\n";
    o("v_lshrrev_b32 v2, 5, v0");
    o("v_readfirstlane_b32 s19, v2");            // w
    o("v_and_b32 v0, 31, v0");
    o("s_lshl_b32 s11, s10, 5");
    o("v_add_nc_u32 v3, s11, v0");               // the row
    o("v_mul_u32_u24 v1, " + S(RB) + ", v3");    // its block's aligned start (block 0: aligned, RB a multiple of 4)
    o("s_lshr_b32 s17, s19, 1");                 // n
    o("s_and_b32 s18, s19, 1");                  // i
    o("s_lshl_b32 s11, s17, 6");
    o("s_lshl_b32 s20, s18, 4");
    o("s_add_u32 s11, s11, s20");                // 64 n + 16 i
    o("v_add_nc_u32 v2, s11, v1");               // ql[l]
    o("s_lshl_b32 s11, s17, 5");
    o("s_add_u32 s11, s11, s20");
    o("s_add_u32 s11, s11, 128");                // 128 + 32 n + 16 i
    o("v_add_nc_u32 v4, s11, v1");               // qh[l]
    o("v_lshlrev_b32 v3, 2, v3");
    o("v_mov_b32 v" + S(M32) + ", 0xc2000000");  // -32.0
    for (int t = 0; t < T; t++) o("v_mov_b32 v" + S(ACC + t) + ", 0");
    auto load = [&](int base) {
        const char* addr[4] = {"v2", "v2", "v4", "v1"};
        const int off[4] = {0, 32, 0, 192};
        for (int p = 0; p < 4; p++) {
            o("global_load_dwordx4 v[" + S(base + 5 * p) + ":" + S(base + 5 * p + 3) + "], " + addr[p] + ", s[0:1] offset:" + S(off[p]));
            o("global_load_dword v" + S(base + 5 * p + 4) + ", " + addr[p] + ", s[0:1] offset:" + S(off[p] + 16));
        }
    };
    load(A);
    // x of the wave's first columns (128 n + 16 i) and their sums' place (8 n + i)
    o("s_lshl_b32 s11, s17, 7");
    o("s_add_u32 s11, s11, s20");
    o("s_mul_i32 s11, s11, " + S(P * 4));
    o("s_add_u32 s12, s2, s11");
    o("s_addc_u32 s13, s3, 0");
    o("s_lshl_b32 s11, s17, 3");
    o("s_add_u32 s11, s11, s18");
    o("s_mul_i32 s11, s11, " + S(T * 4));
    o("s_add_u32 s14, s8, s11");
    o("s_addc_u32 s15, s9, 0");
    o("s_mov_b32 s16, " + S(NB));
    o("s_mov_b32 s18, 0");                       // this block odd
    auto sload = [&](int dst, int n, const std::string& base, int off) {
        for (int i = 0; i < n; i += (n - i >= 16 ? 16 : 8)) {
            const int w = n - i >= 16 ? 16 : 8;
            o("s_load_dwordx" + S(w) + " s[" + S(dst + i) + ":" + S(dst + i + w - 1) + "], " + base + ", " + S(off + 4 * i));
        }
    };
    sload(SX, 4 * P, "s[12:13]", 0);
    k += "block:\n";
    // the next block's aligned start: + 208 from an even block, + 212 from an odd one
    o("s_lshl_b32 s17, s18, 2");
    o("s_add_u32 s17, s17, 208");
    o("v_add_nc_u32 v1, s17, v1");
    o("v_add_nc_u32 v2, s17, v2");
    o("v_add_nc_u32 v4, s17, v4");
    load(B);
    o("s_waitcnt vmcnt(8)");
    o("s_cmp_eq_u32 s18, 0");
    o("s_cbranch_scc1 aligned");
    for (int p = 0; p < 4; p++)
        for (int j = 0; j < 5; j++) {
            const int a = A + 5 * p + j, hi = j < 4 ? a + 1 : a;
            o("v_alignbyte_b32 v" + S(a) + ", v" + S(hi) + ", v" + S(a) + ", 2");
        }
    k += "aligned:\n";
    // d sc of the wave's four groups: scale byte 8 n + i + 2 qq (w uniform: a branch per w)
    o("v_cvt_f32_f16 v" + S(U) + ", v" + S(A + 19));   // d: byte 16 of the scales piece
    for (int w = 0; w < 4; w++) {
        k += "sc" + S(w) + ":\n";
        if (w < 3) {
            o("s_cmp_lg_u32 s19, " + S(w));
            o("s_cbranch_scc1 sc" + S(w + 1));
        }
        const int n = w / 2, i = w % 2;
        for (int qq = 0; qq < 4; qq++) {
            const int byte = 8 * n + i + 2 * qq;
            o("v_bfe_i32 v" + S(DSC + qq) + ", v" + S(A + 15 + byte / 4) + ", " + S(8 * (byte % 4)) + ", 8");
        }
        o("s_branch scdone");
    }
    k += "scdone:\n";
    for (int qq = 0; qq < 4; qq++) {
        o("v_cvt_f32_i32 v" + S(DSC + qq) + ", v" + S(DSC + qq));
        o("v_mul_f32 v" + S(DSC + qq) + ", v" + S(U) + ", v" + S(DSC + qq));
    }
    // 16 steps of 4 columns: group qq, dword kk (bytes 4 kk .. of l).
    auto xoff = [&](int st) { const int qq = st / 4, kk = st % 4; return (32 * qq + 4 * kk) * P * 4; };
    auto buf = [&](int st) { return SX + (st & 1) * 4 * P; };
    for (int st = 0; st < 16; st++) {
        const int qq = st / 4, kk = st % 4;
        o("s_waitcnt lgkmcnt(0)");
        if (st + 1 < 16) sload(buf(st + 1), 4 * P, "s[12:13]", xoff(st + 1));
        else sload(buf(0), 4 * P, "s[12:13]", 256 * P * 4);   // the next block's first step
        if (kk == 2) sload(SXS, T, "s[14:15]", 2 * qq * T * 4);
        const std::string ql = "v" + S(A + 5 * (qq & 1) + kk), qh = "v" + S(A + 10 + kk);
        const std::string u = "v" + S(U), u1 = "v" + S(U + 1);
        if (qq < 2) o("v_and_b32 " + u + ", 0x0f0f0f0f, " + ql);
        else {
            o("v_lshrrev_b32 " + u + ", 4, " + ql);
            o("v_and_b32 " + u + ", 0x0f0f0f0f, " + u);
        }
        if (qq) {
            o("v_lshrrev_b32 " + u1 + ", " + S(2 * qq) + ", " + qh);
            o("v_and_b32 " + u1 + ", 0x03030303, " + u1);
        } else {
            o("v_and_b32 " + u1 + ", 0x03030303, " + qh);
        }
        o("v_lshl_or_b32 " + u + ", " + u1 + ", 4, " + u);
        for (int b = 0; b < 4; b++) o("v_cvt_f32_ubyte" + S(b) + " v" + S(U + 1 + b) + ", " + u);
        o("v_cvt_pkrtz_f16_f32 v" + S(U + 5) + ", v" + S(U + 1) + ", v" + S(U + 2));
        o("v_cvt_pkrtz_f16_f32 v" + S(U + 6) + ", v" + S(U + 3) + ", v" + S(U + 4));
        for (int c = 0; c < 4; c++) {
            const std::string w = "v" + S(U + 5 + c / 2);
            const bool hi = c & 1;
            for (int p = 0; p < P; p++) {
                const std::string x = "s" + S(buf(st) + c * P + p), d = "v" + S(TMP + p);
                if (kk == 0 && c == 0) o("v_pk_mul_f16 " + d + ", " + x + ", " + w + (hi ? " op_sel:[0,1] op_sel_hi:[1,1]" : " op_sel_hi:[1,0]"));
                else o("v_pk_fma_f16 " + d + ", " + x + ", " + w + ", " + d + (hi ? " op_sel:[0,1,0] op_sel_hi:[1,1,1]" : " op_sel_hi:[1,0,1]"));
            }
        }
        if (kk == 3)
            // acc[t] += d sc (tmp[t] - 32 xs[t])
            for (int p = 0; p < P; p++)
                for (int h = 0; h < 2; h++) {
                    const int t = 2 * p + h;
                    const std::string f = "v" + S(U + 7);
                    if (h) {
                        o("v_lshrrev_b32 " + f + ", 16, v" + S(TMP + p));
                        o("v_cvt_f32_f16 " + f + ", " + f);
                    } else {
                        o("v_cvt_f32_f16 " + f + ", v" + S(TMP + p));
                    }
                    o("s_sext_i32_i8 s17, s" + S(SXS + t));   // x16's scale exponent (to_xT)
                    o("v_ldexp_f32 " + f + ", " + f + ", s17");
                    o("v_fmac_f32 " + f + ", s" + S(SXS + t) + ", v" + S(M32));
                    o("v_fmac_f32 v" + S(ACC + t) + ", v" + S(DSC + qq) + ", " + f);
                }
    }
    o("s_add_u32 s12, s12, " + S(256 * P * 4));
    o("s_addc_u32 s13, s13, 0");
    o("s_add_u32 s14, s14, " + S(16 * T * 4));
    o("s_addc_u32 s15, s15, 0");
    o("s_xor_b32 s18, s18, 1");
    o("s_waitcnt vmcnt(0)");
    for (int i = 0; i < 20; i++) o("v_mov_b32 v" + S(A + i) + ", v" + S(B + i));
    o("s_sub_u32 s16, s16, 1");
    o("s_cmp_lg_u32 s16, 0");
    o("s_cbranch_scc1 block");
    o("v_lshlrev_b32 v5, 2, v0");
    o("s_mul_i32 s11, s19, " + S(T * 128));
    o("v_add_nc_u32 v5, s11, v5");
    for (int t = 0; t < T; t++) o("ds_write_b32 v5, v" + S(ACC + t) + " offset:" + S(t * 128));
    o("s_waitcnt lgkmcnt(0)");
    o("s_barrier");
    o("s_cmp_lg_u32 s19, 0");
    o("s_cbranch_scc1 done");
    o("v_lshlrev_b32 v5, 2, v0");
    for (int w = 1; w < 4; w++) {
        for (int t = 0; t < T; t++) o("ds_read_b32 v" + S(TMP + t) + ", v5 offset:" + S((w * T + t) * 128));
        o("s_waitcnt lgkmcnt(0)");
        for (int t = 0; t < T; t++) o("v_add_f32 v" + S(ACC + t) + ", v" + S(ACC + t) + ", v" + S(TMP + t));
    }
    for (int t = 0; t < T; t++) {
        o("s_mul_i32 s17, s6, " + S(4 * t));
        o("s_add_u32 s100, s4, s17");
        o("s_addc_u32 s101, s5, 0");
        o("global_store_dword v3, v" + S(ACC + t) + ", s[100:101]");
    }
    k += "done:\n";
    o("s_endpgm");
    return k;
}

}  // namespace ts5::direct
