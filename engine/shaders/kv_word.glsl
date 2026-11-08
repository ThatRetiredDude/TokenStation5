// A word of 4 cache values as floats, before their block's scale: 8-bit
// values (KV_Q8), or 4-bit ones (KV_Q4 as well; value i in bits 4i .. of the
// word's low 16), both two's complement. kv4x4: 16 values, as 4 words of
// 32 bits (8-bit) or 2 (4-bit) hold them; part j of 4.
#ifdef KV_Q4
vec4 kv4(uint w) {
    const int i = int(w);
    return vec4(bitfieldExtract(i, 0, 4), bitfieldExtract(i, 4, 4), bitfieldExtract(i, 8, 4), bitfieldExtract(i, 12, 4));
}
#define KV16 uvec2
#define kv4x4(w, j) kv4((j) % 2 == 0 ? (w)[(j) / 2] : (w)[(j) / 2] >> 16)
#else
vec4 kv4(uint w) {
    const int i = int(w);
    return vec4(bitfieldExtract(i, 0, 8), bitfieldExtract(i, 8, 8), bitfieldExtract(i, 16, 8), bitfieldExtract(i, 24, 8));
}
#define KV16 uvec4
#define kv4x4(w, j) kv4((w)[j])
#endif
