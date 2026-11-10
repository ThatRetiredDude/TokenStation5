// A word of 4 cache values as floats, before their block's scale: 8-bit
// values (KV_Q8), or 4-bit ones (value i in bits 4i .. of the word's low
// 16), both two's complement. KV_Q4 makes K and V 4-bit (the _q4 builds),
// KV_V4 only V (the _k8v4 builds: K 8-bit, V 4-bit). kk4 and vv4 read a K
// and a V word, KW_T and VW_T are their types. kv4x4: 16 values of K, as 4
// words of 32 bits (8-bit) or 2 (4-bit) hold them; part j of 4.
#ifdef KV_Q4
#define KV_K4
#define KV_V4
#endif
vec4 kv4_4(uint w) {
    const int i = int(w);
    return vec4(bitfieldExtract(i, 0, 4), bitfieldExtract(i, 4, 4), bitfieldExtract(i, 8, 4), bitfieldExtract(i, 12, 4));
}
vec4 kv4_8(uint w) {
    const int i = int(w);
    return vec4(bitfieldExtract(i, 0, 8), bitfieldExtract(i, 8, 8), bitfieldExtract(i, 16, 8), bitfieldExtract(i, 24, 8));
}
#ifdef KV_K4
#define kk4 kv4_4
#define KW_T uint16_t
#define KV16 uvec2
#define kv4x4(w, j) kv4_4((j) % 2 == 0 ? (w)[(j) / 2] : (w)[(j) / 2] >> 16)
#else
#define kk4 kv4_8
#define KW_T uint
#define KV16 uvec4
#define kv4x4(w, j) kv4_8((w)[j])
#endif
#ifdef KV_V4
#define vv4 kv4_4
#define VW_T uint16_t
#else
#define vv4 kv4_8
#define VW_T uint
#endif
