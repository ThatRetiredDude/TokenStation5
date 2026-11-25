// ts5-direct mm-q4k|mm-q6k [T] [split]: the prompt kernels (mm_q4k.h,
// mm_q6k.h) on Qwen3 8B's shapes, against the CPU, and timed: one pass over
// the weights for T tokens.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/gguf.h"
#include "../src/quants.h"
#include "gfx10_asm.h"
#include "kfd.h"
#include "mm_q4k.h"
#include "mm_q4k_sb.h"
#include "mm_q6k_sb.h"
#include "mm_q6k.h"
#include "prefill.h"
#include "pm4.h"

namespace ts5::direct {

int cmd_mm_q4k(int argc, char** argv) {
    const bool q6 = std::string(argv[1]) == "mm-q6k";
    const int T = argc > 2 ? std::atoi(argv[2]) : 16;
    const int SPLIT = argc > 3 ? std::atoi(argv[3]) : 4;
    const int EXP = argc > 4 ? std::atoi(argv[4]) : 0;   // gen_mm_q4k's timing experiments (results wrong)
    if (T != 8 && T != 16) { std::fprintf(stderr, "mm-q4k: T is 8 or 16\n"); return 1; }
    if (SPLIT != 1 && SPLIT != 2 && SPLIT != 4 && SPLIT != 8) { std::fprintf(stderr, "mm-q4k: split is 1, 2, 4 or 8\n"); return 1; }
    const int BB = q6 ? 210 : 144, SUB = q6 ? 16 : 32;   // bytes a block, values a sub-block (x sums)
    kfd::Device dev;
    if (!dev.open() || !dev.create_queue(1 << 20)) { std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str()); return 1; }
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    uint64_t seq = 0;
    sig[0] = 0;
    kfd::Buffer ts;
    if (!dev.alloc(4096, kfd::Mem::Gtt, ts)) return 1;
    auto run = [&](pm4::Stream& body, double* gpu_s) -> bool {
        pm4::Stream w;
        w.acquire_all();
        w.release(ts.gpu, 0, true);
        w.w.insert(w.w.end(), body.w.begin(), body.w.end());
        w.release(ts.gpu + 8, 0, true);
        w.release(dev.signals.gpu, ++seq);
        if (!dev.submit(w.w) || !dev.wait((volatile uint64_t*)sig, seq, 20000)) return false;
        const uint64_t* tt = (const uint64_t*)ts.cpu;
        *gpu_s = (tt[1] - tt[0]) / 100e6;
        return true;
    };
    struct Shape { const char* name; int R, K; };
    const Shape shapes[] = {{"attn_q/output 4096x4096", 4096, 4096}, {"attn_k 1024x4096", 1024, 4096},
                            {"ffn_gate/up 12288x4096", 12288, 4096}, {"ffn_down 4096x12288", 4096, 12288}};
    uint32_t seed = 777;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    bool all_ok = true;
    std::printf("%s, T = %d tokens a pass, %d waves a workgroup%s\n", q6 ? "mm_q6k" : "mm_q4k", T, SPLIT,
                EXP == 1 ? ", TIMING ONLY: no x loads" : EXP == 2 ? ", TIMING ONLY: no unpacking" : EXP == 3 ? ", TIMING ONLY: neither"
                : EXP == 4 ? ", TIMING ONLY: x from LDS" : EXP == 5 ? ", TIMING ONLY: half the x bytes"
                : EXP == 6 ? ", TIMING ONLY: x always the same 128 bytes" : EXP == 10 ? ", blocks split across the waves (sb)" : EXP == 11 ? ", blocks split across the waves, 2 rows a lane (sb)" : "");
    for (const Shape& sh : shapes) {
        const int NB = sh.K / 256;
        if (NB % SPLIT || (q6 && (NB / SPLIT) % 2)) continue;
        const size_t rb = (size_t)NB * BB, wbytes = rb * sh.R;
        gfx10::Kernel k;
        std::string err;
        if (!gfx10::assemble(q6 ? (EXP == 10 ? gen_mm_q6k_sb(sh.K, T) : gen_mm_q6k(sh.K, T, SPLIT)) : EXP == 10 ? gen_mm_q4k_sb(sh.K, T) : EXP == 11 ? gen_mm_q4k_sb(sh.K, T, 2) : gen_mm_q4k(sh.K, T, SPLIT, EXP), k, &err)) { std::fprintf(stderr, "mm-q4k: %s\n", err.c_str()); return 1; }
        kfd::Buffer code, W, xT, xs, Y;
        const size_t cb = (k.code.size() * 4 + 256 + 4095) & ~(size_t)4095;
        if (!dev.alloc(cb, kfd::Mem::Gtt, code) || !dev.alloc(wbytes + 4096, kfd::Mem::Gtt, W) ||
            !dev.alloc((size_t)sh.K * T * 4 + 4096, kfd::Mem::Gtt, xT) || !dev.alloc((size_t)sh.K / SUB * T * 4, kfd::Mem::Gtt, xs) ||
            !dev.alloc((size_t)sh.R * T * 4, kfd::Mem::Gtt, Y)) {
            std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str());
            return 1;
        }
        for (size_t i = 0; i < cb / 4; i++) ((uint32_t*)code.cpu)[i] = 0xbf9f0000;
        std::memcpy(code.cpu, k.code.data(), k.code.size() * 4);
        uint8_t* w = (uint8_t*)W.cpu;
        for (size_t b = 0; b < (size_t)NB * sh.R && q6; b++) {
            uint8_t* blk = w + b * 210;   // ql, qh, int8 scales, then f16 d
            for (int i = 0; i < 208; i++) blk[i] = (uint8_t)(rnd() >> 24);
            const uint16_t d = fp32_to_fp16(0.0001f + (rnd() >> 8) * (0.0008f / 16777216.0f));
            std::memcpy(blk + 208, &d, 2);
        }
        for (size_t b = 0; b < (size_t)NB * sh.R && !q6; b++) {
            uint8_t* blk = w + b * 144;
            const uint16_t d = fp32_to_fp16(0.0005f + (rnd() >> 8) * (0.004f / 16777216.0f));
            const uint16_t dm = fp32_to_fp16(0.0005f + (rnd() >> 8) * (0.004f / 16777216.0f));
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dm, 2);
            for (int i = 4; i < 144; i++) blk[i] = (uint8_t)(rnd() >> 24);
        }
        std::vector<float> X((size_t)T * sh.K);
        for (float& v : X) v = (float)((int)(rnd() >> 16) - 32768) / 32768.0f;
        float* xt = (float*)xT.cpu;
        float* xsum = (float*)xs.cpu;
        for (int c = 0; c < sh.K; c++)
            for (int t = 0; t < T; t++) xt[(size_t)c * T + t] = X[(size_t)t * sh.K + c];
        for (int s = 0; s < sh.K / SUB; s++)
            for (int t = 0; t < T; t++) {
                float a = 0;
                for (int c = 0; c < SUB; c++) a += X[(size_t)t * sh.K + SUB * s + c];
                xsum[(size_t)s * T + t] = a;
            }
        // CPU reference, and each row's sum of |terms| to judge errors by.
        std::vector<double> ref((size_t)T * sh.R), mag((size_t)T * sh.R);
        std::vector<float> row(sh.K);
        for (int r = 0; r < sh.R; r++) {
            dequantize_row(q6 ? GGML_Q6_K : GGML_Q4_K, w + r * rb, row.data(), sh.K);
            for (int t = 0; t < T; t++) {
                double a = 0, m = 0;
                for (int c = 0; c < sh.K; c++) {
                    const double v = (double)row[c] * X[(size_t)t * sh.K + c];
                    a += v;
                    m += std::fabs(v);
                }
                ref[(size_t)t * sh.R + r] = a;
                mag[(size_t)t * sh.R + r] = m;
            }
        }
        const uint32_t args[10] = {(uint32_t)W.gpu, (uint32_t)(W.gpu >> 32), (uint32_t)xT.gpu, (uint32_t)(xT.gpu >> 32), (uint32_t)Y.gpu,
                                   (uint32_t)(Y.gpu >> 32), (uint32_t)sh.R, 0, (uint32_t)xs.gpu, (uint32_t)(xs.gpu >> 32)};
        const uint32_t g[3] = {(uint32_t)(sh.R / (EXP == 11 ? 64 : 32)), 1, 1}, th[3] = {(uint32_t)(32 * SPLIT), 1, 1};
        std::memset(Y.cpu, 0xff, (size_t)sh.R * T * 4);
        double gs;
        {
            pm4::Stream s;
            s.dispatch(k, code.gpu, args, 10, g, th);
            if (!run(s, &gs)) { std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str()); return 1; }
        }
        const float* y = (const float*)Y.cpu;
        double worst = 0;
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < (size_t)T * sh.R; i++) {
            double e = std::fabs(y[i] - ref[i]) / (mag[i] > 0 ? mag[i] : 1);
            if (!(e == e)) e = 1e9;
            if (e > 1e-5 && bad++ == 0) first = i;
            worst = std::max(worst, e);
        }
        if (bad && (!EXP || EXP == 10 || EXP == 11)) {
            all_ok = false;
            std::printf("  %s: WRONG, %zu of %zu values, worst %.3g, first token %zu row %zu: gpu %g cpu %g\n", sh.name, bad,
                        (size_t)T * sh.R, worst, first / sh.R, first % sh.R, y[first], ref[first]);
        }
        // Timed: batches of 20 passes, each waiting for the last, after
        // 100 ms of warm-up (round 39's first timings ran on a cold clock).
        for (double warm = 0; warm < 0.1;) {
            pm4::Stream s;
            for (int i = 0; i < 20; i++) {
                if (i) s.wait_kernels();
                s.dispatch(k, code.gpu, args, 10, g, th);
            }
            double tot;
            if (!run(s, &tot)) { std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str()); return 1; }
            warm += tot;
        }
        std::vector<double> st;
        for (int rep = 0; rep < 12; rep++) {
            pm4::Stream s;
            for (int i = 0; i < 20; i++) {
                if (i) s.wait_kernels();
                s.dispatch(k, code.gpu, args, 10, g, th);
            }
            double tot;
            if (!run(s, &tot)) { std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str()); return 1; }
            if (rep >= 2) st.push_back(tot / 20);
        }
        std::sort(st.begin(), st.end());
        const double us = st[st.size() / 2] * 1e6;
        std::printf("%-26s %7.1f MB  err %.2e  %8.1f us a pass  %6.1f us a token  %6.1f GB/s of weights  %5.0f GFLOP/s  (%u words, %d vgprs, %d sgprs)\n",
                    sh.name, wbytes / 1e6, worst, us, us / T, wbytes / us / 1e3, 2.0 * sh.R * sh.K * T / us / 1e3, (unsigned)k.code.size(),
                    k.vgprs, k.sgprs);
        std::fflush(stdout);
    }
    std::printf("%s: %s\n", q6 ? "mm-q6k" : "mm-q4k", EXP && EXP != 10 && EXP != 11 ? "timing only, values not checked" : all_ok ? "all values match the CPU" : "SOME VALUES WRONG");
    return all_ok ? 0 : 1;
}

// ts5-direct xt-bench: to_xT (prefill.h) against the CPU, then timed with
// its buffers in GTT and in VRAM (round 44: 1.2 ms a prompt token in the
// profile, which the kernel's own work can't explain).
int cmd_xt_bench(int, char**) {
    const int T = 16;
    kfd::Device dev;
    if (!dev.open() || !dev.create_queue(1 << 20)) { std::fprintf(stderr, "xt-bench: %s\n", dev.err().c_str()); return 1; }
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    uint64_t seq = 0;
    sig[0] = 0;
    kfd::Buffer ts;
    if (!dev.alloc(4096, kfd::Mem::Gtt, ts)) return 1;
    auto run = [&](pm4::Stream& body, double* gpu_s) -> bool {
        pm4::Stream w;
        w.acquire_all();
        w.release(ts.gpu, 0, true);
        w.w.insert(w.w.end(), body.w.begin(), body.w.end());
        w.release(ts.gpu + 8, 0, true);
        w.release(dev.signals.gpu, ++seq);
        if (!dev.submit(w.w) || !dev.wait((volatile uint64_t*)sig, seq, 20000)) return false;
        const uint64_t* tt = (const uint64_t*)ts.cpu;
        *gpu_s = (tt[1] - tt[0]) / 100e6;
        return true;
    };
    bool ok = true;
    uint32_t seed = 99;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    for (int K : {4096, 12288}) {
        gfx10::Kernel k;
        std::string err;
        if (!gfx10::assemble(gen_to_xT(K, T), k, &err)) { std::fprintf(stderr, "xt-bench: %s\n", err.c_str()); return 1; }
        kfd::Buffer code;
        if (!dev.alloc(65536, kfd::Mem::Gtt, code)) return 1;
        std::memcpy(code.cpu, k.code.data(), k.code.size() * 4);
        for (kfd::Mem mem : {kfd::Mem::Gtt, kfd::Mem::Vram}) {
            kfd::Buffer Y, xT, x32, x16;
            if (!dev.alloc((size_t)T * K * 4, mem, Y) || !dev.alloc((size_t)T * K * 4, mem, xT) || !dev.alloc((size_t)K / 32 * T * 4, mem, x32) ||
                !dev.alloc((size_t)K / 16 * T * 4, mem, x16)) {
                std::fprintf(stderr, "xt-bench: %s\n", dev.err().c_str());
                return 1;
            }
            const bool gtt = mem == kfd::Mem::Gtt;
            if (gtt)
                for (size_t i = 0; i < (size_t)T * K; i++) ((float*)Y.cpu)[i] = (float)((int)(rnd() >> 16) - 32768) / 32768.0f;
            const uint32_t args[8] = {(uint32_t)Y.gpu, (uint32_t)(Y.gpu >> 32), (uint32_t)xT.gpu, (uint32_t)(xT.gpu >> 32),
                                      (uint32_t)x32.gpu, (uint32_t)(x32.gpu >> 32), (uint32_t)x16.gpu, (uint32_t)(x16.gpu >> 32)};
            const uint32_t g[3] = {(uint32_t)(K / (1024 / T)), 1, 1}, th[3] = {256, 1, 1};
            double gs;
            {
                pm4::Stream s;
                s.dispatch(k, code.gpu, args, 8, g, th);
                if (!run(s, &gs)) { std::fprintf(stderr, "xt-bench: %s\n", dev.err().c_str()); return 1; }
            }
            if (gtt) {
                const float* y = (const float*)Y.cpu;
                const float* xt = (const float*)xT.cpu;
                double worst = 0;
                for (int c = 0; c < K; c++)
                    for (int t = 0; t < T; t++) worst = std::max(worst, (double)std::fabs(xt[(size_t)c * T + t] - y[(size_t)t * K + c]));
                for (int sub : {16, 32}) {
                    const float* xs = (const float*)(sub == 16 ? x16.cpu : x32.cpu);
                    for (int gi = 0; gi < K / sub; gi++)
                        for (int t = 0; t < T; t++) {
                            double a = 0;
                            for (int c = 0; c < sub; c++) a += y[(size_t)t * K + gi * sub + c];
                            worst = std::max(worst, std::fabs(xs[(size_t)gi * T + t] - a));
                        }
                }
                std::printf("to_xT K %d: worst error %.3g %s\n", K, worst, worst < 1e-4 ? "ok" : "WRONG");
                if (!(worst < 1e-4)) ok = false;
            }
            for (int chain = 0; chain < 2; chain++) {
                std::vector<double> st;
                for (int rep = 0; rep < 12; rep++) {
                    pm4::Stream s;
                    for (int i = 0; i < 20; i++) {
                        if (i) {
                            s.wait_kernels();
                            if (chain) s.acquire_light();
                        }
                        s.dispatch(k, code.gpu, args, 8, g, th);
                    }
                    double tot;
                    if (!run(s, &tot)) { std::fprintf(stderr, "xt-bench: %s\n", dev.err().c_str()); return 1; }
                    if (rep >= 2) st.push_back(tot / 20);
                }
                std::sort(st.begin(), st.end());
                std::printf("to_xT K %5d, %s, wait%s: %7.1f us a launch (%.1f GB/s moved)\n", K, gtt ? "GTT " : "VRAM", chain ? " + invalidate" : "",
                            st[st.size() / 2] * 1e6, 2.0 * T * K * 4 / st[st.size() / 2] / 1e9);
            }
            std::fflush(stdout);
        }
    }
    std::printf("xt-bench: %s\n", ok ? "to_xT matches the CPU" : "to_xT WRONG");
    return ok ? 0 : 1;
}

}  // namespace ts5::direct
