// ts5-direct mm-q4k|mm-q6k [T] [split]: the prompt kernels (mm_q4k.h,
// mm_q6k.h) on Qwen3 8B's shapes, against the CPU, and timed: one pass over
// the weights for T tokens.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/gguf.h"
#include "../src/quants.h"
#include "gfx10_asm.h"
#include "kfd.h"
#include "mm_q4k.h"
#include "mm_q4k_sb.h"
#include "mm_q4k_h.h"
#include "mm_q4k_w.h"
#include "mm_q6k_sb.h"
#include "mm_q6k_h.h"
#include "mm_q6k.h"
#include "prefill.h"
#include "pm4.h"

namespace ts5::direct {

int cmd_mm_q4k(int argc, char** argv) {
    const bool q6 = std::string(argv[1]) == "mm-q6k";
    const int T = argc > 2 ? std::atoi(argv[2]) : 16;
    const int SPLIT = argc > 3 ? std::atoi(argv[3]) : 4;
    const int EXP = argc > 4 ? std::atoi(argv[4]) : 0;   // gen_mm_q4k's timing experiments (results wrong)
    if (T != 8 && T != 16 && !(EXP == 20 && (T == 32 || T == 64)) && !((EXP == 12 || EXP == 22) && !q6 && T == 32)) { std::fprintf(stderr, "mm-q4k: T is 8 or 16 (32, 64 for test 20; 32 for Q4_K test 12 / 22)\n"); return 1; }
    if (SPLIT != 1 && SPLIT != 2 && SPLIT != 4 && SPLIT != 8 && !(EXP >= 12 && (SPLIT == 16 || SPLIT == 32))) { std::fprintf(stderr, "mm-q4k: split is 1, 2, 4 or 8\n"); return 1; }
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
    std::vector<Shape> shapes = {{"attn_q/output 4096x4096", 4096, 4096}, {"attn_k 1024x4096", 1024, 4096},
                                 {"ffn_gate/up 12288x4096", 12288, 4096}, {"ffn_down 4096x12288", 4096, 12288}};
    // EXP 16: mm_q4k_h over rows 1024 .. 16384 (K 4096) and a few K 12288,
    // to see whether the time a row jumps once the workgroups no longer all
    // fit on the GPU at once (round 55: gate/up 12288 rows, 384 workgroups,
    // ran slower without its x loads).
    static std::vector<std::string> names;
    if (EXP == 16) {
        shapes.clear();
        std::vector<std::pair<int, int>> rk;
        for (int R : {1024, 2048, 3072, 4096, 6144, 7168, 8192, 8960, 9216, 10240, 11264, 12288, 14336, 16384, 24576}) rk.push_back({R, 4096});
        for (int R : {2048, 4096, 8192}) rk.push_back({R, 12288});
        for (auto [R, K] : rk) names.push_back("sweep " + std::to_string(R) + "x" + std::to_string(K));
        for (size_t i = 0; i < rk.size(); i++) shapes.push_back({names[i].c_str(), rk[i].first, rk[i].second});
    }
    uint32_t seed = 777;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    bool all_ok = true;
    std::printf("%s, T = %d tokens a pass, %d waves a workgroup%s\n", q6 ? "mm_q6k" : "mm_q4k", T, SPLIT,
                EXP == 1 ? ", TIMING ONLY: no x loads" : EXP == 2 ? ", TIMING ONLY: no unpacking" : EXP == 3 ? ", TIMING ONLY: neither"
                : EXP == 4 ? ", TIMING ONLY: x from LDS" : EXP == 5 ? ", TIMING ONLY: half the x bytes"
                : EXP == 6 ? ", TIMING ONLY: x always the same 128 bytes" : EXP == 10 ? ", blocks split across the waves (sb)" : EXP == 11 ? ", blocks split across the waves, 2 rows a lane (sb)"
                : EXP == 12 ? ", packed f16 multiply-adds (mm_q4k_h; errors judged at 3e-3)"
                : EXP == 16 ? ", f16 (mm_q4k_h), rows sweep; pair = two launches back to back as gate + up"
                : EXP == 25 ? ", f16 (mm_q4k_h), weights interleaved for the GPU (every load 512 contiguous bytes)" : EXP == 23 ? ", f16, TIMING ONLY: no weight loads" : EXP == 24 ? ", f16, TIMING ONLY: multiply-adds only (no weight or x loads, unpacking or epilogue)" : EXP == 22 ? ", f16 (mm_q4k_h), the old register layout (sums in every VGPR bank)" : EXP == 21 ? ", f16 (mm_q4k_h), wave64: 64 rows a wave" : EXP == 17 ? ", f16 (mm_q4k_h), 2 rows a lane" : EXP == 20 ? ", wide f16 (mm_q4k_w: weights dequantized to f16, a column a step)" : EXP == 18 ? ", f16, TIMING ONLY: no f32 epilogue" : EXP == 19 ? ", f16, TIMING ONLY: no x loads, unpacking or epilogue" : EXP == 13 ? ", f16, TIMING ONLY: no x loads" : EXP == 14 ? ", f16, TIMING ONLY: no unpacking" : EXP == 15 ? ", f16, TIMING ONLY: neither" : "");
    if (std::getenv("TS5_MM_WVRAM") && std::atoi(std::getenv("TS5_MM_WVRAM"))) std::printf("weights in VRAM (TS5_MM_WVRAM)\n");
    for (const Shape& sh : shapes) {
        const int NB = sh.K / 256;
        if (EXP >= 12 ? (NB % (SPLIT / 4)) : (NB % SPLIT || (q6 && (NB / SPLIT) % 2))) continue;
        const size_t rb = (size_t)NB * BB, wbytes = rb * sh.R;
        gfx10::Kernel k;
        std::string err;
        if (!gfx10::assemble(q6 ? (EXP >= 12 ? gen_mm_q6k_h(sh.K, T, SPLIT / 4, EXP != 22) : EXP == 10 ? gen_mm_q6k_sb(sh.K, T) : gen_mm_q6k(sh.K, T, SPLIT)) : EXP == 10 ? gen_mm_q4k_sb(sh.K, T) : EXP == 11 ? gen_mm_q4k_sb(sh.K, T, 2) : EXP == 20 ? gen_mm_q4k_w(sh.K, T, SPLIT / 4) : EXP >= 12 ? gen_mm_q4k_h(sh.K, T, EXP == 18 ? 4 : EXP == 19 ? 7 : EXP == 23 ? 8 : EXP == 24 ? 15 : EXP >= 16 ? 0 : EXP - 12, q6 ? 1 : SPLIT / 4, EXP == 17 ? 2 : 1, EXP == 21 ? 64 : 32, EXP != 22, EXP == 25) : gen_mm_q4k(sh.K, T, SPLIT, EXP), k, &err)) { std::fprintf(stderr, "mm-q4k: %s\n", err.c_str()); return 1; }
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
        if (EXP >= 12)   // a few tokens far outside f16's range once summed (round 53: Qwen3's activations)
            for (int t = 3; t < T; t += 4)
                for (int c = 0; c < sh.K; c++) X[(size_t)t * sh.K + c] *= 4000.0f;
        // to_xT's x16 scale exponent of each 32 columns and token
        auto scale_e = [&](int, int c0) {   // the chunk's max over all the tokens (round 62)
            float m = 0;
            for (int t = 0; t < T; t++)
                for (int c = c0 & ~31; c < (c0 & ~31) + 32; c++) m = std::max(m, std::fabs(X[(size_t)t * sh.K + c]));
            uint32_t b;
            std::memcpy(&b, &m, 4);
            return std::max((int)(b >> 23) - 130, -100);
        };
        float* xt = (float*)xT.cpu;
        float* xsum = (float*)xs.cpu;
        for (int c = 0; c < sh.K; c++)
            for (int t = 0; t < T; t++) xt[(size_t)c * T + t] = X[(size_t)t * sh.K + c];
        if (EXP >= 12) {   // the f16 kernel reads x16: a column's tokens in pairs, f16
            uint16_t* x16 = (uint16_t*)xT.cpu;
            for (int c = 0; c < sh.K; c++)
                for (int t = 0; t < T; t++) x16[(size_t)c * T + t] = fp32_to_fp16(std::ldexp(X[(size_t)t * sh.K + c], -scale_e(t, c)));
        }
        for (int s = 0; s < sh.K / SUB; s++)
            for (int t = 0; t < T; t++) {
                float a = 0;
                for (int c = 0; c < SUB; c++) a += X[(size_t)t * sh.K + SUB * s + c];
                xsum[(size_t)s * T + t] = a;
                if (EXP >= 12) {   // the exponent in the sum's low byte
                    uint32_t b;
                    std::memcpy(&b, &a, 4);
                    b = (b & ~0xffu) | ((uint32_t)scale_e(t, SUB * s) & 0xff);
                    std::memcpy(&xsum[(size_t)s * T + t], &b, 4);
                }
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
        // TS5_MM_WVRAM=1: the weights copied into VRAM (the GPU's carve-out,
        // large pages) to compare with GTT (system memory, 4 KiB pages; each
        // lane reads its own row, so a wave touches 32 rows' pages at once).
        static const bool wvram = std::getenv("TS5_MM_WVRAM") && std::atoi(std::getenv("TS5_MM_WVRAM"));
        kfd::Buffer Wv;
        if (wvram) {
            static gfx10::Kernel ck;
            static kfd::Buffer cc;
            if (!cc.gpu) {
                const char* src = ".kernel copy16\n.wave32\n.user_sgprs 4\n.tgid_x\n"
                                  "    v_lshlrev_b32 v1, 4, v0\n    s_lshl_b32 s5, s4, 12\n    v_add_nc_u32 v1, s5, v1\n"
                                  "    global_load_dwordx4 v[4:7], v1, s[0:1]\n    s_waitcnt vmcnt(0)\n"
                                  "    global_store_dwordx4 v1, v[4:7], s[2:3]\n    s_endpgm\n";
                if (!gfx10::assemble(src, ck, &err) || !dev.alloc(65536, kfd::Mem::Gtt, cc)) { std::fprintf(stderr, "mm-q4k: copy: %s\n", err.c_str()); return 1; }
                for (size_t i = 0; i < 65536 / 4; i++) ((uint32_t*)cc.cpu)[i] = 0xbf9f0000;
                std::memcpy(cc.cpu, ck.code.data(), ck.code.size() * 4);
            }
            const size_t vb = (wbytes + 4096 + 4095) & ~(size_t)4095;
            if (!dev.alloc(vb, kfd::Mem::Vram, Wv)) { std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str()); return 1; }
            const uint32_t ca[4] = {(uint32_t)W.gpu, (uint32_t)(W.gpu >> 32), (uint32_t)Wv.gpu, (uint32_t)(Wv.gpu >> 32)};
            const uint32_t cg[3] = {(uint32_t)(vb / 4096), 1, 1}, ct[3] = {256, 1, 1};
            pm4::Stream s;
            s.dispatch(ck, cc.gpu, ca, 4, cg, ct);
            double cs;
            if (!run(s, &cs)) { std::fprintf(stderr, "mm-q4k: copy: %s\n", dev.err().c_str()); return 1; }
        }
        // test 25: the weights interleaved (mm_q4k_h.h, IL), built here from W
        kfd::Buffer Wi;
        if (EXP == 25) {
            if (!dev.alloc(wbytes + 4096, kfd::Mem::Gtt, Wi)) { std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str()); return 1; }
            const uint8_t* src = (const uint8_t*)W.cpu;
            uint8_t* dst = (uint8_t*)Wi.cpu;
            for (int G = 0; G < sh.R / 32; G++)
                for (int b = 0; b < NB; b++) {
                    uint8_t* o = dst + ((size_t)G * NB + b) * 4608;
                    for (int l = 0; l < 32; l++) {
                        const uint8_t* blk = src + ((size_t)(32 * G + l) * NB + b) * 144;
                        std::memcpy(o + 16 * l, blk, 16);
                        for (int w = 0; w < 4; w++)
                            for (int h = 0; h < 2; h++) std::memcpy(o + 512 + 1024 * w + 512 * h + 16 * l, blk + 16 + 32 * w + 16 * h, 16);
                    }
                }
        }
        const uint64_t wg = EXP == 25 ? Wi.gpu : wvram ? Wv.gpu : W.gpu;
        const uint32_t args[10] = {(uint32_t)wg, (uint32_t)(wg >> 32), (uint32_t)xT.gpu, (uint32_t)(xT.gpu >> 32), (uint32_t)Y.gpu,
                                   (uint32_t)(Y.gpu >> 32), (uint32_t)sh.R, 0, (uint32_t)xs.gpu, (uint32_t)(xs.gpu >> 32)};
        const uint32_t g[3] = {(uint32_t)(sh.R / (EXP == 11 || EXP == 17 || EXP == 21 ? 64 : 32)), 1, 1}, th[3] = {(uint32_t)((EXP == 21 ? 64 : 32) * SPLIT), 1, 1};
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
            if (e > (EXP >= 12 ? 3e-3 : 1e-5) && bad++ == 0) first = i;   // f16: x rounded, sub-block sums in f16
            worst = std::max(worst, e);
        }
        if (bad && (!EXP || (EXP >= 10 && EXP <= 12) || EXP == 16 || EXP == 17 || EXP == 20 || EXP == 21 || EXP == 22 || EXP == 25)) {
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
        if (EXP == 16) {   // two launches back to back (no wait between them), as gate + up
            std::vector<double> sp;
            for (int rep = 0; rep < 12; rep++) {
                pm4::Stream s;
                for (int i = 0; i < 10; i++) {
                    if (i) s.wait_kernels();
                    s.dispatch(k, code.gpu, args, 10, g, th);
                    s.dispatch(k, code.gpu, args, 10, g, th);
                }
                double tot;
                if (!run(s, &tot)) { std::fprintf(stderr, "mm-q4k: %s\n", dev.err().c_str()); return 1; }
                if (rep >= 2) sp.push_back(tot / 10);
            }
            std::sort(sp.begin(), sp.end());
            std::printf("%-26s rows %5d, %4d workgroups: %8.1f us a pass, %7.2f ns a row; pair %8.1f us (%.2fx one)\n", sh.name, sh.R, sh.R / 32, us,
                        us * 1e3 / sh.R, sp[sp.size() / 2] * 1e6, sp[sp.size() / 2] * 1e6 / us);
        }
        std::printf("%-26s %7.1f MB  err %.2e  %8.1f us a pass  %6.1f us a token  %6.1f GB/s of weights  %5.0f GFLOP/s  (%u words, %d vgprs, %d sgprs)\n",
                    sh.name, wbytes / 1e6, worst, us, us / T, wbytes / us / 1e3, 2.0 * sh.R * sh.K * T / us / 1e3, (unsigned)k.code.size(),
                    k.vgprs, k.sgprs);
        std::fflush(stdout);
    }
    std::printf("%s: %s\n", q6 ? "mm-q6k" : "mm-q4k", EXP && (EXP < 10 || (EXP > 12 && EXP < 16) || EXP == 18 || EXP == 19 || EXP == 23 || EXP == 24) ? "timing only, values not checked" : all_ok ? "all values match the CPU" : "SOME VALUES WRONG");
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
      for (int NT : {256, 1}) {   // 1: to_xT2 (no LDS)
        if (!gfx10::assemble(NT == 1 ? gen_to_xT2(K, T) : gen_to_xT(K, T, NT), k, &err)) { std::fprintf(stderr, "xt-bench: %s\n", err.c_str()); return 1; }
        kfd::Buffer code;
        if (!dev.alloc(65536, kfd::Mem::Gtt, code)) return 1;
        std::memcpy(code.cpu, k.code.data(), k.code.size() * 4);
        for (kfd::Mem mem : {kfd::Mem::Gtt, kfd::Mem::Vram}) {
            kfd::Buffer Y, xT, x32, x16, xh;
            if (!dev.alloc((size_t)T * K * 4, mem, Y) || !dev.alloc((size_t)T * K * 4, mem, xT) || !dev.alloc((size_t)K / 32 * T * 4, mem, x32) ||
                !dev.alloc((size_t)K / 16 * T * 4, mem, x16) || !dev.alloc((size_t)T * K * 2, mem, xh)) {
                std::fprintf(stderr, "xt-bench: %s\n", dev.err().c_str());
                return 1;
            }
            const bool gtt = mem == kfd::Mem::Gtt;
            if (gtt)
                for (size_t i = 0; i < (size_t)T * K; i++) ((float*)Y.cpu)[i] = (float)((int)(rnd() >> 16) - 32768) / 32768.0f;
            const uint32_t args[10] = {(uint32_t)Y.gpu,   (uint32_t)(Y.gpu >> 32),   (uint32_t)xT.gpu, (uint32_t)(xT.gpu >> 32),
                                       (uint32_t)x32.gpu, (uint32_t)(x32.gpu >> 32), (uint32_t)x16.gpu, (uint32_t)(x16.gpu >> 32),
                                       (uint32_t)xh.gpu,  (uint32_t)(xh.gpu >> 32)};
            const uint32_t g[3] = {(uint32_t)(NT == 1 ? K / 256 : K / (4 * NT / T)), 1, 1}, th[3] = {(uint32_t)(NT == 1 ? 256 : NT), 1, 1};
            double gs;
            {
                pm4::Stream s;
                s.dispatch(k, code.gpu, args, 10, g, th);
                if (!run(s, &gs)) { std::fprintf(stderr, "xt-bench: %s\n", dev.err().c_str()); return 1; }
            }
            if (gtt) {
                const float* y = (const float*)Y.cpu;
                const float* xt = (const float*)xT.cpu;
                double worst = 0;
                std::vector<int> e32((size_t)K / 32 * T);
                for (int c = 0; c < K; c++)
                    for (int t = 0; t < T; t++) worst = std::max(worst, (double)std::fabs(xt[(size_t)c * T + t] - y[(size_t)t * K + c]));
                for (int sub : {16, 32}) {
                    const float* xs = (const float*)(sub == 16 ? x16.cpu : x32.cpu);
                    for (int gi = 0; gi < K / sub; gi++)
                        for (int t = 0; t < T; t++) {
                            double a = 0;
                            for (int c = 0; c < sub; c++) a += y[(size_t)t * K + gi * sub + c];
                            float g = xs[(size_t)gi * T + t];
                            uint32_t b;
                            std::memcpy(&b, &g, 4);
                            const int e = (int8_t)(b & 0xff);
                            b &= ~0xffu;
                            std::memcpy(&g, &b, 4);
                            worst = std::max(worst, std::fabs(g - a) - std::fabs(a) * 0x1p-15);
                            // the exponent: the 32 columns' max |x| over 2^e in [8, 16)
                            float m = 0;
                            for (int u = 0; u < T; u++)   // over all the tokens (round 62)
                                for (int c = 0; c < 32; c++) m = std::max(m, std::fabs(y[(size_t)u * K + (gi * sub & ~31) + c]));
                            if (m > 0 && !(std::ldexp(m, -e) >= 8 && std::ldexp(m, -e) < 16)) worst = 1;
                            if (sub == 32) e32[(size_t)gi * T + t] = e;
                        }
                }
                const uint16_t* h = (const uint16_t*)xh.cpu;   // x 2^-e (the 32 columns' e)
                for (int c = 0; c < K; c++)
                    for (int t = 0; t < T; t++) {
                        const float v = y[(size_t)t * K + c];
                        const float g = std::ldexp(fp16_to_fp32(h[(size_t)c * T + t]), e32[(size_t)(c / 32) * T + t]);
                        worst = std::max(worst, (double)std::fabs(g - v) / (std::fabs(v) + 1e-3) * 1e-2);
                    }
                std::printf("to_xT K %d, %d threads: worst error %.3g %s\n", K, NT, worst, worst < 1e-4 ? "ok" : "WRONG");
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
                        s.dispatch(k, code.gpu, args, 10, g, th);
                    }
                    double tot;
                    if (!run(s, &tot)) { std::fprintf(stderr, "xt-bench: %s\n", dev.err().c_str()); return 1; }
                    if (rep >= 2) st.push_back(tot / 20);
                }
                std::sort(st.begin(), st.end());
                std::printf("to_xT K %5d, %d threads, %s, wait%s: %7.1f us a launch (%.1f GB/s moved)\n", K, NT, gtt ? "GTT " : "VRAM", chain ? " + invalidate" : "",
                            st[st.size() / 2] * 1e6, 2.0 * T * K * 4 / st[st.size() / 2] / 1e9);
            }
            std::fflush(stdout);
        }
    }
      }
    std::printf("xt-bench: %s\n", ok ? "to_xT matches the CPU" : "to_xT WRONG");
    return ok ? 0 : 1;
}

}  // namespace ts5::direct

namespace ts5::direct {

// dot-test i8|u8|f16: does gfx1013 run a dot-product instruction? LLVM
// (AMD's compiler) has v_dot4_i32_i8, v_dot4_u32_u8 and v_dot2_f32_f16 on
// gfx1011/gfx1012 but not gfx1013 (the PS5) and refuses them there; this
// runs one anyway. Our assembler writes v_pk_fma_f16 (the same VOP3P form),
// then its opcode is patched to the dot one. A lane's result: 170 (i8, u8:
// 1 5 + 2 6 + 3 7 + 4 8 + 100) or 170.0 (f16: 1.5 2 + 2 3.5 + 160 ... see
// below). If the GPU doesn't decode it the wave faults or halts and the
// wait times out (run it last).
int cmd_dot_test(int argc, char** argv) {
    const std::string kind = argc > 2 ? argv[2] : "i8";
    const int op = kind == "u8" ? 0x17 : kind == "f16" ? 0x13 : 0x16;
    kfd::Device dev;
    if (!dev.open() || !dev.create_queue(1 << 20)) { std::fprintf(stderr, "dot-test: %s\n", dev.err().c_str()); return 1; }
    const bool f16 = kind == "f16";
    // f16: (1.5, 2.0) . (2.0, 3.5) + 160.0 = 3 + 7 + 160 = 170
    const uint32_t a = f16 ? 0x40003e00u : 0x04030201u, b = f16 ? 0x43004000u : 0x08070605u, c = f16 ? 0x43200000u : 100u;
    std::string src = ".kernel dot_test\n.wave32\n.user_sgprs 2\n";
    src += "    v_mov_b32 v2, " + std::to_string(a) + "\n    v_mov_b32 v3, " + std::to_string(b) + "\n    v_mov_b32 v4, " + std::to_string(c) + "\n";
    src += "    v_pk_fma_f16 v1, v2, v3, v4\n";
    src += "    v_lshlrev_b32 v5, 2, v0\n    global_store_dword v5, v1, s[0:1]\n    s_endpgm\n";
    gfx10::Kernel k;
    std::string err;
    if (!gfx10::assemble(src, k, &err)) { std::fprintf(stderr, "dot-test: %s\n", err.c_str()); return 1; }
    size_t at = k.code.size();
    for (size_t i = 0; i + 1 < k.code.size(); i++)
        if ((k.code[i] & 0xff7f0000u) == 0xcc0e0000u && (k.code[i] & 0xff) == 1) { at = i; break; }   // v_pk_fma_f16 v1
    if (at == k.code.size()) { std::fprintf(stderr, "dot-test: v_pk_fma_f16 not found\n"); return 1; }
    k.code[at] = (k.code[at] & ~(0x7fu << 16)) | ((uint32_t)op << 16);
    std::printf("dot-test %s: instruction %08x %08x\n", kind.c_str(), k.code[at], k.code[at + 1]);
    std::fflush(stdout);
    kfd::Buffer code, out, ts;
    if (!dev.alloc(65536, kfd::Mem::Gtt, code) || !dev.alloc(4096, kfd::Mem::Gtt, out)) { std::fprintf(stderr, "dot-test: %s\n", dev.err().c_str()); return 1; }
    for (size_t i = 0; i < 65536 / 4; i++) ((uint32_t*)code.cpu)[i] = 0xbf9f0000;
    std::memcpy(code.cpu, k.code.data(), k.code.size() * 4);
    std::memset(out.cpu, 0xee, 4096);
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    sig[0] = 0;
    const uint32_t args[2] = {(uint32_t)out.gpu, (uint32_t)(out.gpu >> 32)};
    const uint32_t g[3] = {1, 1, 1}, th[3] = {32, 1, 1};
    pm4::Stream s;
    s.acquire_all();
    s.dispatch(k, code.gpu, args, 2, g, th);
    s.release(dev.signals.gpu, 1);
    if (!dev.submit(s.w) || !dev.wait((volatile uint64_t*)sig, 1, 5000)) {
        std::printf("dot-test %s: the GPU did not finish (5 s): it does not run this instruction\n", kind.c_str());
        return 2;
    }
    const uint32_t r = ((const uint32_t*)out.cpu)[0];
    float rf;
    std::memcpy(&rf, &r, 4);
    bool same = true;
    for (int i = 1; i < 32; i++) same = same && ((const uint32_t*)out.cpu)[i] == r;
    const bool ok = f16 ? rf == 170.0f : r == 170u;
    std::printf("dot-test %s: lane 0 = %08x (%s %g), all lanes %s: %s\n", kind.c_str(), r, f16 ? "f32" : "int", f16 ? (double)rf : (double)(int32_t)r,
                same ? "the same" : "DIFFER", ok ? "RUNS, right answer" : "ran, WRONG answer (not a dot product)");
    return ok ? 0 : 3;
}

// ts5-direct fma-peak: how fast this GPU runs back-to-back multiply-adds with
// no memory traffic (round 63: the prompt kernels' bare multiply-adds reached
// only ~66% of the assumed f16 peak; is the peak or the kernel the limit?).
// Variants: pk-s (v_pk_fma_f16 with an SGPR operand, as the prompt kernels),
// pk-v (VGPR operands only), f32 (v_fma_f32), mix (v_fma_mix_f32).
int cmd_fma_peak(int, char**) {
    kfd::Device dev;
    if (!dev.open() || !dev.create_queue(1 << 20)) { std::fprintf(stderr, "fma-peak: %s\n", dev.err().c_str()); return 1; }
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    uint64_t seq = 0;
    sig[0] = 0;
    kfd::Buffer ts, code, out;
    if (!dev.alloc(4096, kfd::Mem::Gtt, ts) || !dev.alloc(65536, kfd::Mem::Gtt, code) || !dev.alloc(1 << 22, kfd::Mem::Gtt, out)) return 1;
    const char* names[4] = {"pk-s", "pk-v", "f32", "mix"};
    const int iters = 2000;
    for (int var = 0; var < 4; var++) {
        std::string src = ".kernel fma_peak\n.wave32\n.user_sgprs 3\n.tgid_x\n";
        for (int i = 0; i < 24; i++) src += "    v_cvt_f32_u32 v" + std::to_string(8 + i) + ", v0\n";
        src += "    s_mov_b32 s20, 0x3c003c00\n    s_mov_b32 s21, 0x3c003c01\n    s_mov_b32 s4, s2\n";
        src += "loop:\n";
        for (int i = 0; i < 128; i++) {
            const std::string d = "v" + std::to_string(8 + i % 16), w = "v" + std::to_string(24 + (i / 16) % 8);
            if (var == 0) src += "    v_pk_fma_f16 " + d + ", s" + std::to_string(20 + (i & 1)) + ", " + w + ", " + d + "\n";
            else if (var == 1) src += "    v_pk_fma_f16 " + d + ", v" + std::to_string(24 + (i / 16 + 3) % 8) + ", " + w + ", " + d + "\n";
            else if (var == 2) src += "    v_fma_f32 " + d + ", s" + std::to_string(20 + (i & 1)) + ", " + w + ", " + d + "\n";
            else src += "    v_fma_mix_f32 " + d + ", " + w + ", " + d + ", " + d + " op_sel_hi:[1,1,0]\n";
        }
        src += "    s_sub_u32 s4, s4, 1\n    s_cmp_lg_u32 s4, 0\n    s_cbranch_scc1 loop\n";
        src += "    s_lshl_b32 s5, s3, 10\n    v_lshlrev_b32 v1, 2, v0\n    v_add_nc_u32 v1, s5, v1\n";
        src += "    v_add_f32 v8, v8, v9\n    global_store_dword v1, v8, s[0:1]\n    s_endpgm\n";
        gfx10::Kernel k;
        std::string err;
        if (!gfx10::assemble(src, k, &err)) { std::fprintf(stderr, "fma-peak: %s\n", err.c_str()); return 1; }
        std::memcpy(code.cpu, k.code.data(), k.code.size() * 4);
        for (int wg : {320, 1280, 4096}) {
            const uint32_t args[3] = {(uint32_t)out.gpu, (uint32_t)(out.gpu >> 32), (uint32_t)iters};
            const uint32_t g[3] = {(uint32_t)wg, 1, 1}, th[3] = {256, 1, 1};
            std::vector<double> st;
            for (int rep = 0; rep < 5; rep++) {
                pm4::Stream w;
                w.acquire_all();
                w.release(ts.gpu, 0, true);
                w.dispatch(k, code.gpu, args, 3, g, th);
                w.wait_kernels();
                w.release(ts.gpu + 8, 0, true);
                w.release(dev.signals.gpu, ++seq);
                if (!dev.submit(w.w) || !dev.wait((volatile uint64_t*)sig, seq, 20000)) { std::fprintf(stderr, "fma-peak: %s\n", dev.err().c_str()); return 1; }
                const uint64_t* tt = (const uint64_t*)ts.cpu;
                if (rep) st.push_back((tt[1] - tt[0]) / 100e6);
            }
            std::sort(st.begin(), st.end());
            const double sec = st[st.size() / 2], fl = (double)wg * 256 * iters * 128 * (var < 2 ? 4 : 2);
            std::printf("fma-peak %-4s %4d workgroups of 256: %8.3f ms, %6.2f TFLOP/s, %.0f instructions a SIMD a us (%d VGPRs)\n", names[var], wg, sec * 1e3,
                        fl / sec / 1e12, (double)wg * 8 * iters * 128 / sec / 1e6 / 80, k.vgprs);
            std::fflush(stdout);
        }
    }
    return 0;
}

// ts5-direct bw-peak: read bandwidth with no arithmetic to speak of, in GTT
// (where the model's weights live) and VRAM (round 72: the prompt kernel's
// weight stream ran at ~223 GB/s; is that the memory or the kernel?).
// Patterns: lin (each wave 512 contiguous bytes a load), row (each lane its
// own 2304-byte row, 16 bytes a load, as the prompt kernels read Q4_K).
int cmd_bw_peak(int, char**) {
    kfd::Device dev;
    if (!dev.open() || !dev.create_queue(1 << 20)) { std::fprintf(stderr, "bw-peak: %s\n", dev.err().c_str()); return 1; }
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    uint64_t seq = 0;
    sig[0] = 0;
    kfd::Buffer ts, code, out;
    if (!dev.alloc(4096, kfd::Mem::Gtt, ts) || !dev.alloc(65536, kfd::Mem::Gtt, code) || !dev.alloc(1 << 22, kfd::Mem::Gtt, out)) return 1;
    const size_t bytes = (size_t)128 << 20;   // 128 MiB (VRAM is a 512 MiB carve-out)
    for (kfd::Mem mem : {kfd::Mem::Gtt, kfd::Mem::Vram}) {
        kfd::Buffer B;
        if (!dev.alloc(bytes, mem, B)) { std::fprintf(stderr, "bw-peak: %s\n", dev.err().c_str()); return 1; }
        if (mem == kfd::Mem::Gtt) std::memset(B.cpu, 1, bytes);
        for (int pat = 0; pat < 2; pat++)
            for (int per : {pat ? 32 : 8, pat ? 144 : 32}) {   // loads a lane (row: 144 = the whole 2304-byte row)
                // lin: workgroup g, wave w, load i: 512 bytes at ((g 8 + w) per + i) 512 + 16 lane
                // row: lane's row = (g 8 + w) 32 + lane, load i at row 2304 + 16 i (per <= 144)
                std::string src = ".kernel bw_peak\n.wave32\n.user_sgprs 4\n.tgid_x\n";
                src += "    v_lshrrev_b32 v2, 5, v0\n    v_and_b32 v0, 31, v0\n    s_lshl_b32 s5, s4, 3\n    v_add_nc_u32 v2, s5, v2\n";   // v2 = g 8 + w
                if (pat == 0) src += "    v_mul_u32_u24 v1, " + std::to_string(per * 512) + ", v2\n    v_lshl_add_u32 v1, v0, 4, v1\n";
                else src += "    v_lshl_add_u32 v3, v2, 5, v0\n    v_mul_u32_u24 v1, 2304, v3\n";
                for (int i = 0; i < 8; i++) src += "    v_mov_b32 v" + std::to_string(8 + i) + ", 0\n";
                for (int i = 0; i < per; i++) {
                    const int r = 16 + 4 * (i % 8);
                    const int off = pat == 0 ? i * 512 : i * 16;
                    if (off < 2048) src += "    global_load_dwordx4 v[" + std::to_string(r) + ":" + std::to_string(r + 3) + "], v1, s[0:1] offset:" + std::to_string(off) + "\n";
                    else src += "    v_add_nc_u32 v4, " + std::to_string(off) + ", v1\n    global_load_dwordx4 v[" + std::to_string(r) + ":" + std::to_string(r + 3) + "], v4, s[0:1]\n";
                    if (i % 8 == 7) {
                        src += "    s_waitcnt vmcnt(0)\n";
                        for (int j = 0; j < 8; j++) src += "    v_add_nc_u32 v" + std::to_string(8 + j) + ", v" + std::to_string(8 + j) + ", v" + std::to_string(16 + 4 * j) + "\n";
                    }
                }
                src += "    v_add_nc_u32 v8, v8, v9\n    v_lshl_add_u32 v1, v2, 5, v0\n    v_lshlrev_b32 v1, 2, v1\n    global_store_dword v1, v8, s[2:3]\n    s_endpgm\n";
                gfx10::Kernel k;
                std::string err;
                if (!gfx10::assemble(src, k, &err)) { std::fprintf(stderr, "bw-peak: %s\n", err.c_str()); return 1; }
                std::memcpy(code.cpu, k.code.data(), k.code.size() * 4);
                const size_t per_wave = pat == 0 ? (size_t)per * 512 : (size_t)32 * 2304;
                const uint32_t wg = (uint32_t)std::min<size_t>(bytes / per_wave / 8, (1 << 22) / 1024);
                const double moved = (double)wg * 8 * 32 * 16 * per;
                const uint32_t args[4] = {(uint32_t)B.gpu, (uint32_t)(B.gpu >> 32), (uint32_t)out.gpu, (uint32_t)(out.gpu >> 32)};
                const uint32_t g[3] = {wg, 1, 1}, th[3] = {256, 1, 1};
                std::vector<double> st;
                // the clock ramps from idle over ~50-100 ms (rounds 12, 20, 74):
                // 150 launches, the last 50 timed
                for (int rep = 0; rep < 150; rep++) {
                    pm4::Stream w;
                    w.acquire_all();
                    w.release(ts.gpu, 0, true);
                    w.dispatch(k, code.gpu, args, 4, g, th);
                    w.wait_kernels();
                    w.release(ts.gpu + 8, 0, true);
                    w.release(dev.signals.gpu, ++seq);
                    if (!dev.submit(w.w) || !dev.wait((volatile uint64_t*)sig, seq, 20000)) { std::fprintf(stderr, "bw-peak: %s\n", dev.err().c_str()); return 1; }
                    const uint64_t* tt = (const uint64_t*)ts.cpu;
                    if (rep >= 100) st.push_back((tt[1] - tt[0]) / 100e6);
                }
                std::sort(st.begin(), st.end());
                std::printf("bw-peak %s %s, %2d loads a lane, %6u workgroups: %8.1f us, %6.1f MB read, %6.1f GB/s\n", mem == kfd::Mem::Gtt ? "GTT " : "VRAM",
                            pat ? "row" : "lin", per, wg, st[st.size() / 2] * 1e6, moved / 1e6, moved / st[st.size() / 2] / 1e9);
                std::fflush(stdout);
            }
    }
    return 0;
}

}  // namespace ts5::direct
