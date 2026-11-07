// A Q5_0 block (22 bytes: d as f16, qh, the 32 values' fifth bits, then 16
// bytes of nibbles, byte j's low one value j and its high one value j + 16;
// a value is (q - 16) d) and its dot with 32 values of x (x[c]: values
// 4c .. 4c + 3), shared by mv16_q5_0 and mvt_q5_0 so their sums agree bit
// for bit. Blocks are only 2-byte aligned: a block is read as six 4-byte
// words from its byte o's word, w[0] .. w[5], and sh = 16 where o is 2 past
// a word (then w[0]'s low half is the block before's), 0 where it starts
// one (then w[5]'s high half is past the block).
struct Q5Blk { float d; uint qh; uint q[4]; };

Q5Blk q5_0_unpack(uint w[6], uint sh) {
    Q5Blk b;
    b.d = unpackHalf2x16(w[0] >> sh).x;
    b.qh = sh == 0 ? (w[0] >> 16) | (w[1] << 16) : w[1];
    [[unroll]] for (uint k = 0; k < 4; k++) b.q[k] = sh == 0 ? (w[k + 1] >> 16) | (w[k + 2] << 16) : w[k + 2];
    return b;
}

float q5_0_dot(Q5Blk b, vec4 x[8]) {
    float s = 0.0;
    [[unroll]] for (uint k = 0; k < 4; k++) {
        // Bytes 4k .. 4k + 3: values 4k + i (low nibbles, fifth bits 4k + i)
        // and 16 + 4k + i (high nibbles, fifth bits 16 + 4k + i).
        const uvec4 q = uvec4(b.q[k]) >> uvec4(0, 8, 16, 24);
        const uvec4 hl = uvec4(b.qh >> (4 * k)) >> uvec4(0, 1, 2, 3);
        const uvec4 hh = uvec4(b.qh >> (4 * k + 16)) >> uvec4(0, 1, 2, 3);
        const vec4 lo = vec4((q & 15u) | ((hl & 1u) << 4)) - 16.0;
        const vec4 hi = vec4(((q >> 4) & 15u) | ((hh & 1u) << 4)) - 16.0;
        s += dot(lo, x[k]);
        s += dot(hi, x[k + 4]);
    }
    return b.d * s;
}
