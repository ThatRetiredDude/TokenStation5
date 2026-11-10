// The arithmetic of a Q4_K (with Q5K, Q5_K) block that every one-token
// (mv16_q4_k) and verify-batch (mvt_q4_k, mvtx_q4_k, mvt_glu_q4_k) kernel
// shares, so that each sums a block into its row with the same
// floating-point operations in the same order. Speculative decoding keeps
// its promise of the same output only while a verify batch reproduces the
// one-token forward bit for bit (see mvt_q4_k.glsl).
//
// Thread th of the 8 that share a block holds 32 of its values: 16 low
// nibbles of sub-block 2 (th / 2) and 16 high ones of the next. A half adds
//     d s * (w . x) - dmin m * sum(x)
// to the row: the 16 weights dotted with x as a chain of fused
// multiply-adds, then two more for the scale and the min. d s and dmin m
// are exact in f32 (an f16 times a 6-bit integer: 17 significant bits), so
// a kernel may form them once per row and block, ahead of its token loop.

// Sub-blocks 2p and 2p + 1's 6-bit scales (bytes 0 and 1 of the result)
// and mins (bytes 2 and 3), from the block's header h. Sub-block j < 4 has
// its scale in the low 6 bits of byte j of the scales and its min in those
// of byte j + 4; j >= 4 has their low 4 bits in byte j + 4 (scale low
// nibble, min high) and their top 2 bits in the top bits of bytes j - 4
// (scale) and j (min).
uint scales_mins(uvec4 h, uint p) {
    const uint sh = 16u * (p & 1u);
    const uint a = (h.y >> sh) & 0xFFFFu, b = (h.z >> sh) & 0xFFFFu, c = (h.w >> sh) & 0xFFFFu;
    // Both, then a select: a branch on p (which differs between threads)
    // split the caller's row loop into blocks, and the compiler then waited
    // for each row's loads before issuing the next row's.
    const uint lo = (a & 0x3F3Fu) | ((b & 0x3F3Fu) << 16);
    const uint hi = ((c & 0x0F0Fu) | ((a >> 2) & 0x3030u)) | ((((c >> 4) & 0x0F0Fu) | ((b >> 2) & 0x3030u)) << 16);
    return p < 2 ? lo : hi;
}
// The 4 bytes of a word as floats (one conversion each on AMD hardware).
vec4 bytes4(uint w) { return vec4(float(w & 255u), float((w >> 8) & 255u), float((w >> 16) & 255u), float(w >> 24)); }

// Thread th's (d s_lo, d s_hi, dmin m_lo, dmin m_hi), from header h.
vec4 block_coefs(uvec4 h, uint th) {
    const vec4 sm = bytes4(scales_mins(h, th >> 1));
    const vec2 dm = unpackHalf2x16(h.x);
    return vec4(dm.x * sm.x, dm.x * sm.y, dm.y * sm.z, dm.y * sm.w);
}

// The thread's 16 low (hi == 0) or high nibbles of word q as floats; with
// Q5K, qh is its word of high bits (bytes 16 (th % 2) .. + 15: bit j of
// byte l is value l's fifth bit in sub-block j).
#ifdef Q5K
void nibbles(uvec4 q, uvec4 qh, uint th, uint hi, out vec4 w[4]) {
    const uint j = (th & ~1u) + hi, sh = 4 * hi;
    [[unroll]] for (uint c = 0; c < 4; c++)
        w[c] = bytes4(((q[c] >> sh) & 0x0F0F0F0Fu) | (((qh[c] >> j) & 0x01010101u) << 4));
}
#else
void nibbles(uvec4 q, uint th, uint hi, out vec4 w[4]) {
    const uint sh = 4 * hi;
    [[unroll]] for (uint c = 0; c < 4; c++) w[c] = bytes4((q[c] >> sh) & 0x0F0F0F0Fu);
}
#endif

// The sum of a half's 16 x values.
float half_sum(vec4 x[4]) {
    float s = 0.0;
    [[unroll]] for (uint c = 0; c < 4; c++) s += x[c].x + x[c].y + x[c].z + x[c].w;
    return s;
}

// acc plus a half: k d s (or its high half's), km dmin m, xs half_sum(x).
float half_acc(float acc, vec4 w[4], vec4 x[4], float k, float km, float xs) {
    float s = w[0].x * x[0].x;
    s = fma(w[0].y, x[0].y, s);
    s = fma(w[0].z, x[0].z, s);
    s = fma(w[0].w, x[0].w, s);
    [[unroll]] for (uint c = 1; c < 4; c++) {
        s = fma(w[c].x, x[c].x, s);
        s = fma(w[c].y, x[c].y, s);
        s = fma(w[c].z, x[c].z, s);
        s = fma(w[c].w, x[c].w, s);
    }
    return fma(-km, xs, fma(k, s, acc));
}
