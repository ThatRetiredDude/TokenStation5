// Included by peak_fma32.comp and peak_fma16.comp (#version and the
// include extension come from them).
// ts5 peak: arithmetic rate. 8 independent chains of fused multiply-adds
// on 4-wide vectors (f32, or packed f16 with PEAK_F16), 16 steps a loop.
#ifdef PEAK_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define T f16vec4
#define S float16_t
#else
#define T vec4
#define S float
#endif
layout(local_size_x = 256) in;
layout(std430, binding = 0) writeonly buffer Obuf { float o[]; };
layout(push_constant) uniform PC { uint n, iters, chunk; } pc;
void main() {
    const S f = S(gl_LocalInvocationID.x & 7u) * S(0.001);
    T a0 = T(f), a1 = T(f + S(0.01)), a2 = T(f + S(0.02)), a3 = T(f + S(0.03));
    T a4 = T(f + S(0.04)), a5 = T(f + S(0.05)), a6 = T(f + S(0.06)), a7 = T(f + S(0.07));
    const T m = T(S(0.999)), c = T(S(0.0001));
    for (uint i = 0; i < pc.iters; i++) {
        for (uint k = 0; k < 16; k++) {
            a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
            a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
        }
    }
    const T s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (float(s.x + s.y + s.z + s.w) == -1.0) o[gl_WorkGroupID.x] = 1.0;
}
