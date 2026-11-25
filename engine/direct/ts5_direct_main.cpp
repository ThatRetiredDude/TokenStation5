// ts5-direct: tries the direct engine on the GPU.
//   ts5-direct info              what the kernel reports about the GPU
//   ts5-direct fill              smoke test: a kernel fills a buffer, the CPU checks it
//   ts5-direct read-bw [MiB]     read bandwidth on the direct path (default 1024 MiB)
//   ts5-direct bw-sweep [MiB]    read bandwidth over memory kinds, loads in flight, cache hints
//                                and workgroup counts, 5 runs each
//   ts5-direct bw-last [MiB] [runs]     grid vs chunked, workgroup placement per run
//   ts5-direct bw-repeat [N] [wave64]   the best setup N times in a row (hang hunt)
//   ts5-direct mv-q4k [runs] [wgs/CU] [v2]   Q4_K mat-vec on Qwen3 8B's shapes: checked against the CPU, timed
// Kernels are assembled at start from the text built into the program.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gfx10_asm.h"
#include "kernels.h"   // generated: kernel sources as strings
#include "kfd.h"
#include "mv_q4k.h"
#include "mv_q4k2.h"
#include "mv_q6k.h"
#include "attn.h"
#include "ops.h"
#include "mm_q4k.h"
#include "mm_q4k_sb.h"
#include "mm_q4k_h.h"
#include "mm_q4k_w.h"
#include "mm_q6k_sb.h"
#include "mm_q6k_h.h"
#include "mm_q6k.h"
#include "prefill.h"
#include "pm4.h"

using namespace ts5;

namespace ts5::direct {
int cmd_token(int argc, char** argv);   // token.cpp (its own device, runner.h)
int cmd_mm_q4k(int argc, char** argv);  // mm_test.cpp
int cmd_xt_bench(int argc, char** argv);  // mm_test.cpp
int cmd_dot_test(int argc, char** argv);  // mm_test.cpp
int cmd_fma_peak(int argc, char** argv);  // mm_test.cpp
}

namespace {

int die(const std::string& why) {
    std::fprintf(stderr, "ts5-direct: %s\n", why.c_str());
    return 1;
}

bool load_kernel(kfd::Device& dev, const char* src, gfx10::Kernel& k, kfd::Buffer& code) {
    std::string err;
    if (!gfx10::assemble(src, k, &err)) { std::fprintf(stderr, "ts5-direct: assembler: %s\n", err.c_str()); return false; }
    // The instruction prefetcher reads past the end: pad with s_code_end.
    size_t bytes = k.code.size() * 4 + 256;
    if (!dev.alloc(bytes, kfd::Mem::Gtt, code)) { std::fprintf(stderr, "ts5-direct: %s\n", dev.err().c_str()); return false; }
    uint32_t* p = (uint32_t*)code.cpu;
    std::memcpy(p, k.code.data(), k.code.size() * 4);
    for (size_t i = k.code.size(); i < code.size / 4; i++) p[i] = 0xbf9f0000;   // s_code_end
    return true;
}


// A read kernel with N loads in flight per loop, optional cache hints on the
// loads ("glc slc dlc" streams past the caches), wave32 or wave64. Same
// arguments as kernels/read_bw.s, which is the N=4, no-hint, wave32 case.
// chunked: each workgroup streams its own contiguous slice (s3 is then the
// workgroup's step, tpw*16) instead of the whole grid striding together.
// Every lane also stores HW_ID1 (where its wave ran) 4 MiB into the output.
std::string gen_read_kernel(int n, const std::string& hints, bool wave64, int tpw = 256, bool chunked = false) {
    int shift = 0;
    while ((1 << shift) < tpw) shift++;
    std::string k = ".kernel read_gen\n";
    k += wave64 ? ".wave64\n" : ".wave32\n";
    k += ".user_sgprs 6\n.tgid_x\n";
    k += "    s_lshl_b32 s7, s6, " + std::to_string(shift) + "\n    v_add_nc_u32 v1, s7, v0\n";
    k += chunked ? "    v_lshlrev_b32 v2, 4, v0\n" : "    v_lshlrev_b32 v2, 4, v1\n";
    const int a = 4 + 4 * n;   // the four sums sit right after the load registers
    for (int c = 0; c < 4; c++) k += "    v_mov_b32 v" + std::to_string(a + c) + ", 0\n";
    k += "    s_mov_b32 s8, s0\n    s_mov_b32 s9, s1\n";
    if (chunked) {   // slice start = wg * loops * n * step
        k += "    s_mul_i32 s10, s2, s3\n    s_mul_i32 s10, s10, " + std::to_string(n) + "\n    s_mul_i32 s10, s10, s6\n";
        k += "    s_add_u32 s8, s8, s10\n    s_addc_u32 s9, s9, 0\n";
    }
    k += "loop:\n";
    for (int i = 0; i < n; i++) {
        int r = 4 + 4 * i;
        k += "    global_load_dwordx4 v[" + std::to_string(r) + ":" + std::to_string(r + 3) + "], v2, s[8:9]" +
             (hints.empty() ? "" : " " + hints) + "\n";
        k += "    s_add_u32 s8, s8, s3\n    s_addc_u32 s9, s9, 0\n";
    }
    k += "    s_sub_u32 s2, s2, 1\n";
    for (int i = 0; i < n; i++) {
        int r = 4 + 4 * i;
        k += "    s_waitcnt vmcnt(" + std::to_string(n - 1 - i) + ")\n";
        for (int c = 0; c < 4; c++)
            k += "    v_add_nc_u32 v" + std::to_string(a + c) + ", v" + std::to_string(a + c) + ", v" + std::to_string(r + c) + "\n";
    }
    k += "    s_cmp_lg_u32 s2, 0\n    s_cbranch_scc1 loop\n";
    const std::string A = std::to_string(a), A1 = std::to_string(a + 1), A2 = std::to_string(a + 2), A3 = std::to_string(a + 3);
    k += "    v_add3_u32 v" + A + ", v" + A + ", v" + A1 + ", v" + A2 + "\n    v_add_nc_u32 v" + A + ", v" + A + ", v" + A3 + "\n";
    k += "    v_lshlrev_b32 v1, 2, v1\n    global_store_dword v1, v" + A + ", s[4:5]\n";
    k += "    s_getreg_b32 s10, hwreg(HW_REG_HW_ID1)\n    v_add_nc_u32 v3, 0x400000, v1\n    v_mov_b32 v" + A1 + ", s10\n";
    k += "    global_store_dword v3, v" + A1 + ", s[4:5]\n    s_endpgm\n";
    return k;
}

const char* mem_name(kfd::Mem m) {
    return m == kfd::Mem::Gtt ? "gtt" : m == kfd::Mem::GttUncached ? "gtt-uc" : "vram";
}

struct ReadCfg {
    int loads;
    std::string hints;
    bool wave64;
    int per_cu;   // workgroups per CU
    int tpw = 256;   // threads per workgroup
    bool chunked = false;
};

std::string cfg_name(const ReadCfg& c) {
    return std::string("wave") + (c.wave64 ? "64" : "32") + " loads " + std::to_string(c.loads) + " hints " +
           (c.hints.empty() ? "-" : c.hints) + " wgs/CU " + std::to_string(c.per_cu) + " threads/wg " + std::to_string(c.tpw) +
           (c.chunked ? " chunked" : " grid");
}

// Read bandwidth over a list of setups. Everything is allocated and every
// kernel assembled into one code buffer before the first dispatch: no
// memory is allocated or freed while the queue runs. (Rounds 1 and 2 on the
// PS5 hung after 29 and 48 rows, each row allocating and freeing its code;
// a new allocation can make the kernel stop and restore the queue.)
// `repeat` runs every setup that many times and reports the spread.
int cmd_bw_sweep(kfd::Device& dev, size_t mib, const std::vector<kfd::Mem>& mems, const std::vector<ReadCfg>& cfgs, int repeat) {
    if (!dev.create_queue()) return die(dev.err());
    gfx10::Kernel fk;
    kfd::Buffer fcode;
    if (!load_kernel(dev, kKernel_fill, fk, fcode)) return 1;
    // All the kernels, 256-byte aligned, in one buffer.
    std::vector<gfx10::Kernel> ks(cfgs.size());
    std::vector<size_t> at(cfgs.size());
    size_t total = 0;
    for (size_t i = 0; i < cfgs.size(); i++) {
        std::string err;
        if (!gfx10::assemble(gen_read_kernel(cfgs[i].loads, cfgs[i].hints, cfgs[i].wave64, cfgs[i].tpw, cfgs[i].chunked), ks[i], &err))
            return die("assembler: " + err);
        at[i] = total;
        total += (ks[i].code.size() * 4 + 256 + 255) & ~(size_t)255;
    }
    kfd::Buffer code;
    if (!dev.alloc(total, kfd::Mem::Gtt, code)) return die(dev.err());
    for (size_t i = 0; i < total / 4; i++) ((uint32_t*)code.cpu)[i] = 0xbf9f0000;   // s_code_end
    for (size_t i = 0; i < cfgs.size(); i++) std::memcpy((char*)code.cpu + at[i], ks[i].code.data(), ks[i].code.size() * 4);

    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    auto run = [&](pm4::Stream& s, double* gpu_s) -> bool {
        std::memset(sig, 0, 64);
        pm4::Stream w;
        w.acquire_all();
        w.release(dev.signals.gpu + 8, 0, true);
        w.w.insert(w.w.end(), s.w.begin(), s.w.end());
        w.release(dev.signals.gpu + 16, 0, true);
        w.release(dev.signals.gpu, 1);
        if (!dev.submit(w.w) || !dev.wait((volatile uint64_t*)sig, 1, 20000)) return false;
        *gpu_s = (sig[2] - sig[1]) / 100e6;
        return true;
    };
    const uint32_t C = 3;
    const size_t bytes = mib << 20;
    const uint32_t max_threads = dev.cu_count() * 16 * 1024;
    double best_all = 0;
    std::string best_cfg;
    long dispatches = 0;
    std::printf("mem     wave  loads hints        wgs    thr   GB/s (best / median / worst of %d)  check\n", repeat);
    for (kfd::Mem m : mems) {
        kfd::Buffer src, out;
        if (!dev.alloc(bytes, m, src)) {
            std::printf("%-7s (can't allocate: %s)\n", mem_name(m), dev.err().c_str());
            continue;
        }
        if (!dev.alloc(8u << 20, kfd::Mem::Gtt, out)) return die(dev.err());   // sums, then HW_ID1 at 4 MiB
        {   // fill the source with C on the GPU
            pm4::Stream s;
            uint32_t n = (uint32_t)(bytes / 4);
            uint32_t args[4] = {(uint32_t)src.gpu, (uint32_t)(src.gpu >> 32), C, n};
            uint32_t g[3] = {(n + 255) / 256, 1, 1}, t[3] = {256, 1, 1};
            s.dispatch(fk, fcode.gpu, args, 4, g, t);
            double gs;
            if (!run(s, &gs)) return die(dev.err() + " (filling)");
            std::printf("%-7s fill %.1f MiB: %.1f GB/s write\n", mem_name(m), bytes / 1048576.0, bytes / gs / 1e9);
        }
        for (size_t i = 0; i < cfgs.size(); i++) {
            const ReadCfg& c = cfgs[i];
            uint32_t wgs = dev.cu_count() * c.per_cu;
            uint32_t threads = wgs * c.tpw;
            if (threads > max_threads) continue;
            size_t step = (size_t)threads * 16;
            size_t loops = bytes / (step * c.loads);
            if (loops < 1) continue;
            size_t used = loops * step * c.loads;
            if (c.chunked) step = (size_t)c.tpw * 16;
            uint32_t args[6] = {(uint32_t)src.gpu, (uint32_t)(src.gpu >> 32), (uint32_t)loops, (uint32_t)step,
                                (uint32_t)out.gpu, (uint32_t)(out.gpu >> 32)};
            uint32_t g[3] = {wgs, 1, 1}, t[3] = {(uint32_t)c.tpw, 1, 1};
            // r: GB/s per run; placement: busiest WGP's workgroup count and WGPs used, per run
            std::vector<double> r;
            std::vector<std::pair<double, std::pair<int, int>>> placed;
            for (int rep = 0; rep < repeat; rep++) {
                pm4::Stream s;
                s.dispatch(ks[i], code.gpu + at[i], args, 6, g, t);
                double gs;
                if (!run(s, &gs)) {
                    std::printf("HANG at dispatch %ld (%s %s, rep %d): %s\n", dispatches, mem_name(m), cfg_name(c).c_str(), rep,
                                dev.err().c_str());
                    return 1;
                }
                dispatches++;
                r.push_back(used / gs / 1e9);
                {   // HW_ID1 of each workgroup's first lane: SE 20:18, SA 16, WGP 13:10
                    const uint32_t* hw = (const uint32_t*)((const char*)out.cpu + (4u << 20));
                    std::map<uint32_t, int> per;
                    for (uint32_t w = 0; w < wgs; w++) per[(hw[w * c.tpw] >> 10) & 0x7ff & ~0x30u]++;
                    int mx = 0;
                    for (auto& p : per) mx = std::max(mx, p.second);
                    placed.push_back({r.back(), {mx, (int)per.size()}});
                }
            }
            std::sort(placed.begin(), placed.end());
            std::sort(r.begin(), r.end());
            uint32_t want = (uint32_t)(loops * c.loads * 4 * C);
            uint32_t* o = (uint32_t*)out.cpu;
            bool ok = o[0] == want && o[threads - 1] == want;
            std::printf("%-7s %-5s %-5d %-12s %-6u %-5d %6.1f / %6.1f / %6.1f  %s\n", mem_name(m), c.wave64 ? "64" : "32", c.loads,
                        c.hints.empty() ? "-" : c.hints.c_str(), wgs, c.tpw, r.back(), r[r.size() / 2], r.front(), ok ? "ok" : "WRONG");
            std::printf("        %s; busiest WGP/WGPs used: slowest run %d/%d, fastest %d/%d; runs under 300 GB/s: %d of %zu\n",
                        c.chunked ? "chunked" : "grid", placed.front().second.first, placed.front().second.second,
                        placed.back().second.first, placed.back().second.second,
                        (int)std::count_if(r.begin(), r.end(), [](double x) { return x < 300; }), r.size());
            std::fflush(stdout);
            if (ok && r[r.size() / 2] > best_all) {
                best_all = r[r.size() / 2];
                best_cfg = std::string(mem_name(m)) + " " + cfg_name(c);
            }
        }
    }
    std::printf("bw-sweep: best median %.1f GB/s (%s), %ld dispatches, no hang\n", best_all, best_cfg.c_str(), dispatches);
    return 0;
}

int cmd_info(kfd::Device& dev) {
    std::printf("GPU %s, id %u, gfx target %u, KFD %u.%u\n", dev.name.c_str(), dev.gpu_id, dev.gfx_target, dev.kfd_major, dev.kfd_minor);
    std::printf("%u CUs (%u SIMDs, %u per CU), %u waves per SIMD, render node %u\n", dev.cu_count(), dev.simd_count, dev.simd_per_cu,
                dev.max_waves_per_simd, dev.drm_minor);
    for (auto& p : dev.props) std::printf("  %s %" PRIu64 "\n", p.first.c_str(), p.second);
    if (!dev.create_queue()) return die(dev.err());
    std::printf("compute queue: created\n");
    return 0;
}

int cmd_fill(kfd::Device& dev) {
    if (!dev.create_queue()) return die(dev.err());
    gfx10::Kernel k;
    kfd::Buffer code, buf;
    if (!load_kernel(dev, kKernel_fill, k, code)) return 1;
    const uint32_t n = 1 << 20;
    if (!dev.alloc(n * 4, kfd::Mem::Gtt, buf)) return die(dev.err());
    std::memset(buf.cpu, 0, n * 4);
    volatile uint64_t* fence = (volatile uint64_t*)dev.signals.cpu;
    *fence = 0;
    pm4::Stream s;
    s.acquire_all();
    uint32_t args[4] = {(uint32_t)buf.gpu, (uint32_t)(buf.gpu >> 32), 0x7e57c0de, n - 5};
    uint32_t groups[3] = {(n + 255) / 256, 1, 1}, threads[3] = {256, 1, 1};
    s.dispatch(k, code.gpu, args, 4, groups, threads);
    s.release(dev.signals.gpu, 1);
    auto t0 = std::chrono::steady_clock::now();
    if (!dev.submit(s.w)) return die(dev.err());
    if (!dev.wait(fence, 1, 5000)) return die(dev.err());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    uint32_t* p = (uint32_t*)buf.cpu;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t want = i < n - 5 ? 0x7e57c0de : 0;
        if (p[i] != want && bad++ < 5) std::printf("  dword %u: 0x%08x, want 0x%08x\n", i, p[i], want);
    }
    std::printf("fill: %s, %u of %u dwords wrong, %.3f ms from submit to fence\n", bad ? "FAIL" : "ok", bad, n, ms);
    return bad ? 1 : 0;
}

int cmd_read_bw(kfd::Device& dev, size_t mib) {
    if (!dev.create_queue()) return die(dev.err());
    gfx10::Kernel k;
    kfd::Buffer code, src, out;
    if (!load_kernel(dev, kKernel_read_bw, k, code)) return 1;
    const uint32_t wgs = dev.cu_count() * 32;          // enough workgroups to fill every CU many times over
    const uint32_t threads = wgs * 256;
    const size_t step = (size_t)threads * 16;            // bytes all threads read in one load
    size_t loops = (mib << 20) / (step * 4);
    if (loops < 1) loops = 1;
    size_t bytes = loops * step * 4;
    if (!dev.alloc(bytes, kfd::Mem::Gtt, src) || !dev.alloc((size_t)threads * 4, kfd::Mem::Gtt, out)) return die(dev.err());
    // Known contents, so the sum can be checked.
    uint32_t* p = (uint32_t*)src.cpu;
    for (size_t i = 0; i < bytes / 4; i++) p[i] = (uint32_t)i * 2654435761u;
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    double best = 0;
    for (int rep = 0; rep < 5; rep++) {
        std::memset(sig, 0, 64);
        pm4::Stream s;
        s.acquire_all();
        s.release(dev.signals.gpu + 8, 0, true);      // GPU clock before
        uint32_t args[6] = {(uint32_t)src.gpu, (uint32_t)(src.gpu >> 32), (uint32_t)loops, (uint32_t)step,
                            (uint32_t)out.gpu, (uint32_t)(out.gpu >> 32)};
        uint32_t g[3] = {wgs, 1, 1}, t[3] = {256, 1, 1};
        s.dispatch(k, code.gpu, args, 6, g, t);
        s.release(dev.signals.gpu + 16, 0, true);     // GPU clock after
        s.release(dev.signals.gpu, 1);
        auto t0 = std::chrono::steady_clock::now();
        if (!dev.submit(s.w)) return die(dev.err());
        if (!dev.wait((volatile uint64_t*)sig, 1, 20000)) return die(dev.err());
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t ticks = sig[2] - sig[1];
        double gpu_s = ticks / 100e6;                   // GFX10 clock counter: 100 MHz
        double gbs = bytes / gpu_s / 1e9;
        std::printf("read-bw: %.1f MiB in %.3f ms on the GPU clock (%.3f ms wall): %.1f GB/s\n", bytes / 1048576.0, gpu_s * 1e3,
                    wall * 1e3, gbs);
        if (gbs > best) best = gbs;
    }
    // Check one thread's sum.
    uint32_t want = 0;
    for (size_t l = 0; l < loops * 4; l++)
        for (int c = 0; c < 4; c++) want += p[(l * step) / 4 + c];
    uint32_t got = ((uint32_t*)out.cpu)[0];
    std::printf("read-bw: best %.1f GB/s; thread 0's sum %s (0x%08x, want 0x%08x)\n", best, got == want ? "ok" : "WRONG", got, want);
    return got == want ? 0 : 1;
}

float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : std::ldexp((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}

uint16_t float_to_half(float f) {   // normal range only, round to nearest
    int e;
    float m = std::frexp(std::fabs(f), &e);   // f = m 2^e, m in [0.5, 1)
    uint32_t mant = (uint32_t)std::lround(m * 2048.0f);   // 11 bits incl. the leading 1
    if (mant == 2048) { mant = 1024; e++; }
    return (uint16_t)((f < 0 ? 0x8000 : 0) | ((uint32_t)(e + 14) << 10) | (mant & 1023));
}

// The 6-bit scale and min of sub-block j (0..7) of a Q4_K block's 12 bytes.
void q4k_scale_min(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_K mat-vec (mv_q4k.h) on the shapes of Qwen3 8B's Q4_K matrices: random
// blocks and x, the GPU's y checked against a CPU sum in double, then timed
// one dispatch at a time with the matrix in VRAM.
// q6: the same for Q6_K (mv_q6k.h) on Qwen3 8B's Q6_K shapes (no diag
// variants; mv-q6k-diag adds the plain-read control).
int cmd_mv_q4k(kfd::Device& dev, int runs, int per_cu, bool diag = false, bool v2 = false, bool q6 = false) {
    const size_t BB = q6 ? 210 : 144;   // block bytes
    auto gen = [v2](int cols, const std::string& hints, int d, int depth, bool inter = false) {
        return v2 ? ts5::direct::gen_mv_q4k2(cols, hints, d, depth) : ts5::direct::gen_mv_q4k(cols, hints, d, depth, inter);
    };
    std::printf("kernel: %s\n", q6 ? "mv_q6k (4 lanes per block)" : v2 ? "mv_q4k2 (2 lanes per block)" : "mv_q4k (8 lanes per block, headers by one dword load)");
    if (!dev.create_queue()) return die(dev.err());
    gfx10::Kernel ck;
    kfd::Buffer ccode;
    if (!load_kernel(dev, kKernel_copy, ck, ccode)) return 1;
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    auto run = [&](pm4::Stream& s, double* gpu_s) -> bool {
        std::memset(sig, 0, 64);
        pm4::Stream w;
        w.acquire_all();
        w.release(dev.signals.gpu + 8, 0, true);
        w.w.insert(w.w.end(), s.w.begin(), s.w.end());
        w.release(dev.signals.gpu + 16, 0, true);
        w.release(dev.signals.gpu, 1);
        if (!dev.submit(w.w) || !dev.wait((volatile uint64_t*)sig, 1, 20000)) return false;
        *gpu_s = (sig[2] - sig[1]) / 100e6;
        return true;
    };
    struct Shape { const char* name; uint32_t rows, cols; };
    const Shape shapes4[] = {{"attn_q/attn_output 4096x4096", 4096, 4096}, {"attn_k 1024x4096", 1024, 4096},
                             {"ffn_gate/up 12288x4096", 12288, 4096}, {"ffn_down 4096x12288", 4096, 12288}};
    const Shape shapes6[] = {{"attn_v 1024x4096", 1024, 4096}, {"ffn_down 4096x12288", 4096, 12288},
                             {"output 151936x4096", 151936, 4096}};
    const Shape* shapes = q6 ? shapes6 : shapes4;
    const size_t nshapes = q6 ? 3 : 4;
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    bool all_ok = true;
    std::printf("shape                          MB     err        single dispatches: GB/s (best / median / worst of %d)  us median\n", runs);
    // Everything for every shape is allocated before the first dispatch:
    // round 6 freed one shape's buffers and allocated the next one's
    // between dispatches, and the second shape's results never arrived
    // (NaN) and the third's copy hit GPU page faults.
    struct Prep {
        const Shape* sh;
        gfx10::Kernel k;
        kfd::Buffer code, dcode, ts, wh, wv, xb, yb;
        std::vector<double> ref, mag;
    };
    // The read kernel for mv-diag's control row.
    gfx10::Kernel rk;
    kfd::Buffer rd_code, rd_out;
    {
        std::string err;
        if (!gfx10::assemble(gen_read_kernel(4, "slc dlc", false, 256, true), rk, &err)) return die("read kernel: " + err);
        size_t cb = (rk.code.size() * 4 + 256 + 4095) & ~(size_t)4095;
        if (!dev.alloc(cb, kfd::Mem::Gtt, rd_code) || !dev.alloc(8u << 20, kfd::Mem::Gtt, rd_out)) return die(dev.err());
        for (size_t i = 0; i < cb / 4; i++) ((uint32_t*)rd_code.cpu)[i] = 0xbf9f0000;
        std::memcpy(rd_code.cpu, rk.code.data(), rk.code.size() * 4);
    }
    std::vector<Prep> preps(nshapes);
    for (size_t si = 0; si < preps.size(); si++) {
        Prep& P = preps[si];
        const Shape& sh = shapes[si];
        P.sh = &sh;
        const size_t nb = sh.cols / 256, row_bytes = nb * BB, bytes = row_bytes * sh.rows;
        std::string err;
        if (!gfx10::assemble(q6 ? ts5::direct::gen_mv_q6k((int)sh.cols) : gen((int)sh.cols, "", 0, 1, !v2), P.k, &err))
            return die("mv kernel: " + err);
        size_t cb = (P.k.code.size() * 4 + 256 + 4095) & ~(size_t)4095;
        if (!dev.alloc(cb, kfd::Mem::Gtt, P.code) || !dev.alloc(cb * 2, kfd::Mem::Gtt, P.dcode) ||
            !dev.alloc((size_t)dev.cu_count() * per_cu * 8 * 16, kfd::Mem::Gtt, P.ts)) return die(dev.err());
        for (size_t i = 0; i < cb / 4; i++) ((uint32_t*)P.code.cpu)[i] = 0xbf9f0000;
        std::memcpy(P.code.cpu, P.k.code.data(), P.k.code.size() * 4);
        if (!dev.alloc(bytes + 4096, kfd::Mem::Gtt, P.wh) || !dev.alloc(bytes + 4096, kfd::Mem::Vram, P.wv) ||
            !dev.alloc(sh.cols * 4, kfd::Mem::Gtt, P.xb) || !dev.alloc(sh.rows * 4, kfd::Mem::Gtt, P.yb))
            return die(dev.err());
        uint8_t* W = (uint8_t*)P.wh.cpu;
        for (size_t b = 0; q6 && b < nb * sh.rows; b++) {   // Q6_K: random ql, qh, scales; d small
            uint8_t* blk = W + b * 210;
            for (int i = 0; i < 208; i++) blk[i] = (uint8_t)(rnd() >> 24);
            uint16_t d = float_to_half(0.0001f + (rnd() >> 8) * (0.001f / 16777216.0f));
            std::memcpy(blk + 208, &d, 2);
        }
        for (size_t b = 0; !q6 && b < nb * sh.rows; b++) {
            uint8_t* blk = W + b * 144;
            uint16_t d = float_to_half(0.0005f + (rnd() >> 8) * (0.004f / 16777216.0f));
            uint16_t dm = float_to_half(0.0005f + (rnd() >> 8) * (0.004f / 16777216.0f));
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dm, 2);
            for (int i = 4; i < 144; i++) blk[i] = (uint8_t)(rnd() >> 24);
        }
        float* X = (float*)P.xb.cpu;
        for (uint32_t c = 0; c < sh.cols; c++) X[c] = (float)((int)(rnd() >> 16) - 32768) / 32768.0f;
        // CPU reference, and the size of each row's terms to judge the error by.
        P.ref.resize(sh.rows);
        P.mag.resize(sh.rows);
        for (uint32_t r = 0; q6 && r < sh.rows; r++) {   // Q6_K, as ggml's dequantize_row_q6_K
            double acc = 0, am = 0;
            for (size_t b = 0; b < nb; b++) {
                const uint8_t* blk = W + r * row_bytes + b * 210;
                uint16_t dh;
                std::memcpy(&dh, blk + 208, 2);
                const double d = half_to_float(dh);
                const int8_t* sc = (const int8_t*)(blk + 192);
                const float* x = X + b * 256;
                for (int n = 0; n < 2; n++) {
                    const uint8_t* ql = blk + 64 * n;
                    const uint8_t* qh = blk + 128 + 32 * n;
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        const double t[4] = {d * sc[8 * n + is] * q1 * x[128 * n + l], d * sc[8 * n + is + 2] * q2 * x[128 * n + l + 32],
                                             d * sc[8 * n + is + 4] * q3 * x[128 * n + l + 64], d * sc[8 * n + is + 6] * q4 * x[128 * n + l + 96]};
                        for (double v : t) { acc += v; am += std::fabs(v); }
                    }
                }
            }
            P.ref[r] = acc;
            P.mag[r] = am;
        }
        for (uint32_t r = 0; !q6 && r < sh.rows; r++) {
            double acc = 0, am = 0;
            for (size_t b = 0; b < nb; b++) {
                const uint8_t* blk = W + r * row_bytes + b * 144;
                uint16_t dh, mh;
                std::memcpy(&dh, blk, 2);
                std::memcpy(&mh, blk + 2, 2);
                double d = half_to_float(dh), dmin = half_to_float(mh);
                const uint8_t* q = blk + 16;
                for (int j = 0; j < 4; j++) {
                    int s0, m0, s1, m1;
                    q4k_scale_min(2 * j, blk + 4, s0, m0);
                    q4k_scale_min(2 * j + 1, blk + 4, s1, m1);
                    for (int l = 0; l < 32; l++) {
                        double w0 = d * s0 * (q[32 * j + l] & 0xF) - dmin * m0;
                        double w1 = d * s1 * (q[32 * j + l] >> 4) - dmin * m1;
                        double x0 = X[b * 256 + 64 * j + l], x1 = X[b * 256 + 64 * j + 32 + l];
                        acc += w0 * x0 + w1 * x1;
                        am += std::fabs(w0 * x0) + std::fabs(w1 * x1);
                    }
                }
            }
            P.ref[r] = acc;
            P.mag[r] = am;
        }
    }
    for (Prep& P : preps) {
        const Shape& sh = *P.sh;
        const size_t bytes = sh.cols / 256 * BB * sh.rows;
        const gfx10::Kernel& k = P.k;
        double gs;
        {   // matrix into VRAM
            pm4::Stream s;
            uint32_t n16 = (uint32_t)(bytes / 16);
            uint32_t args[5] = {(uint32_t)P.wh.gpu, (uint32_t)(P.wh.gpu >> 32), (uint32_t)P.wv.gpu, (uint32_t)(P.wv.gpu >> 32), n16};
            uint32_t g[3] = {(n16 + 255) / 256, 1, 1}, t[3] = {256, 1, 1};
            s.dispatch(ck, ccode.gpu, args, 5, g, t);
            if (!run(s, &gs)) return die(dev.err() + " (copying to VRAM)");
        }
        // per: rows per workgroup for the sliced variants; step: the row step
        // the round-robin default gets in the same argument.
        const uint32_t wgs = dev.cu_count() * per_cu, per = (sh.rows + wgs - 1) / wgs, step = v2 && !q6 ? per : wgs * 8;
        auto mv = [&](const kfd::Buffer& w, double* t_s) -> bool {
            std::memset(P.yb.cpu, 0xff, sh.rows * 4);
            pm4::Stream s;
            uint32_t args[10] = {(uint32_t)w.gpu, (uint32_t)(w.gpu >> 32), (uint32_t)P.xb.gpu, (uint32_t)(P.xb.gpu >> 32),
                                 (uint32_t)P.yb.gpu, (uint32_t)(P.yb.gpu >> 32), sh.rows, sh.cols, step, 0};
            uint32_t g[3] = {wgs, 1, 1}, t[3] = {256, 1, 1};
            s.dispatch(k, P.code.gpu, args, 10, g, t);
            return run(s, t_s);
        };
        // Steady state: n dispatches back to back in one submission, each
        // waiting for the one before (as a token's kernels do), timed as a
        // whole. Single dispatches with the CPU waiting between them let the
        // GPU clock drop: round 11's wave records ran at about 490 MHz on
        // the short shapes and 1.6-2.3 GHz on the long ones.
        auto batch = [&](const gfx10::Kernel& kk, uint64_t code_gpu, const kfd::Buffer& w, int n, double* per_s,
                         uint32_t rows_arg = 0) -> bool {
            pm4::Stream s;
            uint32_t args[10] = {(uint32_t)w.gpu, (uint32_t)(w.gpu >> 32), (uint32_t)P.xb.gpu, (uint32_t)(P.xb.gpu >> 32),
                                 (uint32_t)P.yb.gpu, (uint32_t)(P.yb.gpu >> 32), sh.rows, (uint32_t)P.ts.gpu,
                                 rows_arg ? rows_arg : step, (uint32_t)(P.ts.gpu >> 32)};
            uint32_t g[3] = {wgs, 1, 1}, t[3] = {256, 1, 1};
            for (int i = 0; i < n; i++) {
                if (i) s.wait_kernels();
                s.dispatch(kk, code_gpu, args, 10, g, t);
            }
            double total;
            if (!run(s, &total)) return false;
            *per_s = total / n;
            return true;
        };
        auto check = [&](const char* where) -> double {
            const float* Y = (const float*)P.yb.cpu;
            double worst = 0;
            uint32_t bad = 0, first_bad = 0;
            for (uint32_t r = 0; r < sh.rows; r++) {
                double e = std::fabs(Y[r] - P.ref[r]) / (P.mag[r] > 0 ? P.mag[r] : 1);
                if (!(e == e)) e = 1e9;   // NaN: the row was never written
                if (e > 1e-5 && bad++ == 0) first_bad = r;
                worst = std::max(worst, e);
            }
            if (bad) {
                std::printf("  %s %s: WRONG, %u of %u rows, worst error %.3g, first bad row %u: gpu %g cpu %g\n", sh.name, where, bad,
                            sh.rows, worst, first_bad, Y[first_bad], P.ref[first_bad]);
                all_ok = false;
            }
            return worst;
        };
        if (!mv(P.wh, &gs)) return die(dev.err() + " (mv_q4k from system memory)");
        check("system memory");
        std::vector<double> r;
        double e = 0;
        for (int i = 0; i < runs; i++) {
            if (!mv(P.wv, &gs)) return die(dev.err() + " (mv_q4k from VRAM)");
            if (i == 0) e = check("VRAM");
            r.push_back(bytes / gs / 1e9);
        }
        std::sort(r.begin(), r.end());
        double med = r[r.size() / 2];
        // Steady: warm the clock up with 3 batches of 100 dispatches (a
        // batch is about 5,100 ring dwords; submit() takes up to half the
        // 16,384-dword ring), then time 3 more.
        double per_s;
        // The first shape gets 20 warm-up batches (it came in under-clocked
        // in rounds 14 and 17 with 3).
        for (int i = 0; i < (&P == &preps[0] ? 20 : 3); i++)
            if (!batch(k, P.code.gpu, P.wv, 100, &per_s)) return die(dev.err() + " (warm-up)");
        std::vector<double> st;
        for (int i = 0; i < 3; i++) {
            if (!batch(k, P.code.gpu, P.wv, 100, &per_s)) return die(dev.err() + " (steady)");
            st.push_back(per_s);
        }
        std::sort(st.begin(), st.end());
        double e_steady = check("steady");
        std::vector<double> st20;
        for (int i = 0; i < 3; i++) {
            if (!batch(k, P.code.gpu, P.wv, 20, &per_s)) return die(dev.err() + " (steady 20)");
            st20.push_back(per_s);
        }
        std::sort(st20.begin(), st20.end());
        std::printf("%-30s %6.1f %9.2e  %6.1f / %6.1f / %6.1f  %8.1f   steady %6.1f GB/s, %7.1f us  (err %.2e); batches of 20: %7.1f us\n", sh.name, bytes / 1e6, e,
                    r.back(), med, r.front(), bytes / med / 1e3, bytes / st[1] / 1e9, st[1] * 1e6, e_steady, st20[1] * 1e6);
        std::fflush(stdout);
        // diag: the same kernel with parts left out (results not checked),
        // then with 1 and 3 steps of loads in flight (results checked).
        for (int d = 1; diag && !q6 && d < 26; d++) {
            // 1-7: parts left out; 8: timestamps; 9: timestamps, nothing but the frame; 10, 11: 2 and 3 steps in
            // flight; 12, 13: loads only with 2 and 3 steps in flight; 14-17 rows dealt round-robin: full,
            // loads only, full with 2 steps, timestamps
            // 18-21: cache hints on the weight loads: slc dlc (the old default) and glc dlc, full and loads only
            // 23: round-robin wave records in batches of 100 (round 19: 100-dispatch
            // batches ran ~6 us slower per call than batches of 20; clock or queue?)
            // 24, 25: the same records without the block math (mode 1 | 8), batches
            // of 20 and 100: does the long-run clock drop follow the math (power)?
            const bool inter = (d >= 14 && d <= 17) || d >= 23;
            const int nrun = d == 23 || d == 25 ? 100 : runs;
            const char* hints = d == 18 || d == 19 ? "slc dlc" : d == 20 || d == 21 ? "glc dlc" : "";
            const int mode = d <= 8 ? d : d == 9 ? 15 : d == 12 || d == 13 || d == 15 || d == 19 || d == 21 ? 3 : d == 17 || d == 23 ? 8 : d >= 24 ? 9 : 0;
            const int depth = d == 10 || d == 12 || d == 16 ? 2 : d == 11 || d == 13 ? 3 : 1;
            // 22: sliced rows, 1 step, full (the default before round 19)
            const uint32_t dper = inter ? wgs * 8 : per;
            gfx10::Kernel dk;
            std::string err;
            if (!gfx10::assemble(gen((int)sh.cols, hints, mode, depth, inter), dk, &err))
                return die("mv_q4k diag: " + err);
            if (dk.code.size() * 4 + 256 > P.dcode.size) return die("diag code too big");
            for (size_t i = 0; i < P.dcode.size / 4; i++) ((uint32_t*)P.dcode.cpu)[i] = 0xbf9f0000;
            std::memcpy(P.dcode.cpu, dk.code.data(), dk.code.size() * 4);
            std::vector<double> dt;   // steady: right after the warm batches above, 3 batches of `runs`
            std::memset(P.yb.cpu, 0xff, sh.rows * 4);
            for (int i = 0; i < 3; i++) {
                if (!batch(dk, P.dcode.gpu, P.wv, nrun, &gs, dper)) return die(dev.err() + " (diag " + std::to_string(d) + ")");
                dt.push_back(gs * 1e6);
            }
            std::sort(dt.begin(), dt.end());
            if (inter) std::printf("    [rows round-robin]");
            if (d >= 18 && d <= 21) std::printf("    [hints \"%s\"]", hints);
            if (d == 22) std::printf("    [rows sliced]");
            if (d == 23 || d == 25) std::printf("    [batches of 100]");
            if (d >= 24) std::printf("    [no math]");
            if (mode & 8) {   // the last run's wave records (shader clock, HW_ID1)
                const uint32_t* T = (const uint32_t*)P.ts.cpu;
                const uint32_t nw = wgs * 8;
                uint32_t t0 = T[0];
                for (uint32_t w = 1; w < nw; w++)
                    if ((int32_t)(T[4 * w] - t0) < 0) t0 = T[4 * w];
                std::vector<uint32_t> start, staged, total, endt;
                std::map<uint32_t, int> per_cu, per_simd;   // workgroups per CU, waves per SIMD
                for (uint32_t w = 0; w < nw; w++) {
                    uint32_t st = T[4 * w] - t0, hw = T[4 * w + 1];
                    start.push_back(st);
                    staged.push_back(T[4 * w + 2]);
                    total.push_back(T[4 * w + 3]);
                    endt.push_back(st + T[4 * w + 3]);
                    uint32_t simd = hw & 0x1d3f00, cu = (hw & 0x1d3c00) | ((hw >> 9) & 1);   // SE, SA, WGP (+ SIMD / CU of the WGP)
                    per_simd[simd]++;
                    if (w % 8 == 0) per_cu[cu]++;
                }
                std::sort(start.begin(), start.end());
                std::sort(staged.begin(), staged.end());
                std::sort(total.begin(), total.end());
                std::sort(endt.begin(), endt.end());
                int cu_max = 0, simd_max = 0;
                for (auto& p : per_cu) cu_max = std::max(cu_max, p.second);
                for (auto& p : per_simd) simd_max = std::max(simd_max, p.second);
                std::printf("    %s records (last dispatch): %8.1f us median, about %.0f MHz; clocks: starts median %u max %u, staged after %u median, run %u min / %u median / "
                            "%u max, last end %u; %zu CUs used (busiest %d workgroups), %zu SIMDs (busiest %d waves)\n",
                            mode == 8 ? "full kernel" : "frame only", dt[dt.size() / 2], endt.back() / dt[dt.size() / 2], start[nw / 2], start.back(), staged[nw / 2],
                            total.front(), total[nw / 2], total.back(), endt.back(), per_cu.size(), cu_max, per_simd.size(), simd_max);
            } else if (mode)
                std::printf("    without%s%s%s, %d step(s) in flight: %8.1f us median\n", mode & 1 ? " math" : "", mode & 2 ? " lds-reads" : "",
                            mode & 4 ? " weight-loads" : "", depth, dt[dt.size() / 2]);
            else {
                double e2 = check(inter ? "round-robin" : d == 22 ? "sliced" : d >= 18 ? "hints" : depth == 2 ? "depth 2" : "depth 3");
                std::printf("    %d steps in flight: %8.1f us median, %6.1f GB/s, err %.2e\n", depth, dt[dt.size() / 2],
                            bytes / dt[dt.size() / 2] / 1e3, e2);
            }
            std::fflush(stdout);
        }
        if (diag) {   // control: the plain read kernel (chunked, 4 loads, slc dlc) over the same matrix bytes, steady
            const uint32_t threads = wgs * 256;
            const uint32_t loops = (uint32_t)(bytes / ((size_t)threads * 16 * 4));
            const double used = (double)loops * threads * 16 * 4;
            uint32_t args[6] = {(uint32_t)P.wv.gpu, (uint32_t)(P.wv.gpu >> 32), loops, 4096, (uint32_t)rd_out.gpu, (uint32_t)(rd_out.gpu >> 32)};
            uint32_t g[3] = {wgs, 1, 1}, t[3] = {256, 1, 1};
            std::vector<double> rt;
            for (int rep = 0; rep < 3; rep++) {
                pm4::Stream s;
                for (int i = 0; i < 100; i++) {
                    if (i) s.wait_kernels();
                    s.dispatch(rk, rd_code.gpu, args, 6, g, t);
                }
                double total;
                if (!run(s, &total)) return die(dev.err() + " (read control)");
                rt.push_back(total / 100);
            }
            std::sort(rt.begin(), rt.end());
            std::printf("    control: plain reads of %.1f MB: %8.1f us, %6.1f GB/s\n", used / 1e6, rt[1] * 1e6, used / rt[1] / 1e9);
            std::fflush(stdout);
        }
    }
    std::printf("%s: %s (%u workgroups, 256 threads each)\n", q6 ? "mv-q6k" : "mv-q4k", all_ok ? "all rows match the CPU" : "SOME ROWS WRONG",
                dev.cu_count() * per_cu);
    return all_ok ? 0 : 1;
}

}  // namespace

// Attention for one token (attn.h) over n cached positions, Qwen3 8B's
// heads: random q, K, V checked against a CPU softmax in double, from
// system memory and from VRAM, then timed in batches of 20 (part + reduce).
int cmd_attn(kfd::Device& dev, const std::vector<uint32_t>& lens) {
    if (!dev.create_queue()) return die(dev.err());
    gfx10::Kernel ck, kp, kr;
    kfd::Buffer ccode, pcode, rcode;
    if (!load_kernel(dev, kKernel_copy, ck, ccode)) return 1;
    {
        const std::string ps = ts5::direct::gen_attn_part(), rs = ts5::direct::gen_attn_reduce();
        if (!load_kernel(dev, ps.c_str(), kp, pcode) || !load_kernel(dev, rs.c_str(), kr, rcode)) return 1;
    }
    std::printf("attn_part: %zu words, %d vgprs; attn_reduce: %zu words\n", kp.code.size(), kp.vgprs, kr.code.size());
    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    auto run = [&](pm4::Stream& s, double* gpu_s) -> bool {
        std::memset(sig, 0, 64);
        pm4::Stream w;
        w.acquire_all();
        w.release(dev.signals.gpu + 8, 0, true);
        w.w.insert(w.w.end(), s.w.begin(), s.w.end());
        w.release(dev.signals.gpu + 16, 0, true);
        w.release(dev.signals.gpu, 1);
        if (!dev.submit(w.w) || !dev.wait((volatile uint64_t*)sig, 1, 20000)) return false;
        *gpu_s = (sig[2] - sig[1]) / 100e6;
        return true;
    };
    uint32_t seed = 4242;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rf = [&]() { return (float)((int)(rnd() >> 16) - 32768) / 32768.0f; };
    // Everything allocated up front (round 6: no alloc/free between dispatches).
    struct Case { uint32_t n, chunks, per; kfd::Buffer q, kh, vh, kv, vv, part, out; std::vector<double> ref; };
    std::vector<Case> cases(lens.size());
    for (size_t ci = 0; ci < lens.size(); ci++) {
        Case& C = cases[ci];
        C.n = lens[ci];
        // At most 10 splits per KV head (80 workgroups: two per CU), every
        // split at least one 256-position chunk.
        const uint32_t nch = (C.n + 255) / 256, per = (nch + 9) / 10;
        C.per = per * 256;
        C.chunks = (nch + per - 1) / per;   // splits
        const size_t kvb = (size_t)C.n * 8 * 256;
        if (!dev.alloc(32 * 128 * 4, kfd::Mem::Gtt, C.q) || !dev.alloc(kvb, kfd::Mem::Gtt, C.kh) || !dev.alloc(kvb, kfd::Mem::Gtt, C.vh) ||
            !dev.alloc(kvb, kfd::Mem::Vram, C.kv) || !dev.alloc(kvb, kfd::Mem::Vram, C.vv) ||
            !dev.alloc((size_t)C.chunks * 8 * 2112, kfd::Mem::Gtt, C.part) || !dev.alloc(32 * 128 * 4, kfd::Mem::Gtt, C.out))
            return die(dev.err());
        float* Q = (float*)C.q.cpu;
        for (int i = 0; i < 32 * 128; i++) Q[i] = rf() * 2.0f;
        uint16_t* K = (uint16_t*)C.kh.cpu;
        uint16_t* V = (uint16_t*)C.vh.cpu;
        for (size_t i = 0; i < kvb / 2; i++) {
            float a = rf(), b = rf();
            K[i] = a == 0 ? 0 : float_to_half(a);
            V[i] = b == 0 ? 0 : float_to_half(b);
        }
        C.ref.assign(32 * 128, 0.0);
        std::vector<double> sc(C.n);
        for (int h = 0; h < 32; h++) {
            const int g = h / 4;
            double m = -1e300;
            for (uint32_t t = 0; t < C.n; t++) {
                double d = 0;
                for (int i = 0; i < 128; i++) d += (double)Q[h * 128 + i] * half_to_float(K[((size_t)t * 8 + g) * 128 + i]);
                sc[t] = d / std::sqrt(128.0);
                m = std::max(m, sc[t]);
            }
            double l = 0;
            for (uint32_t t = 0; t < C.n; t++) { sc[t] = std::exp(sc[t] - m); l += sc[t]; }
            for (uint32_t t = 0; t < C.n; t++)
                for (int i = 0; i < 128; i++) C.ref[h * 128 + i] += sc[t] / l * half_to_float(V[((size_t)t * 8 + g) * 128 + i]);
        }
    }
    const float scale = (float)(1.4426950408889634 / std::sqrt(128.0));
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    bool all_ok = true;
    for (Case& C : cases) {
        const size_t kvb = (size_t)C.n * 8 * 256;
        double gs;
        for (int w = 0; w < 2; w++) {   // K, V into VRAM
            const kfd::Buffer& src = w ? C.vh : C.kh;
            const kfd::Buffer& dst = w ? C.vv : C.kv;
            pm4::Stream s;
            uint32_t n16 = (uint32_t)(kvb / 16);
            uint32_t args[5] = {(uint32_t)src.gpu, (uint32_t)(src.gpu >> 32), (uint32_t)dst.gpu, (uint32_t)(dst.gpu >> 32), n16};
            uint32_t g[3] = {(n16 + 255) / 256, 1, 1}, t[3] = {256, 1, 1};
            s.dispatch(ck, ccode.gpu, args, 5, g, t);
            if (!run(s, &gs)) return die(dev.err() + " (copying to VRAM)");
        }
        auto add = [&](pm4::Stream& s, const kfd::Buffer& kb, const kfd::Buffer& vb, bool wait_first, int only = 0) {   // only: 1 part, 2 reduce
            uint32_t a1[12] = {(uint32_t)C.q.gpu, (uint32_t)(C.q.gpu >> 32), (uint32_t)kb.gpu, (uint32_t)(kb.gpu >> 32),
                               (uint32_t)vb.gpu, (uint32_t)(vb.gpu >> 32), C.n, scale_bits, (uint32_t)C.part.gpu, (uint32_t)(C.part.gpu >> 32),
                               C.per, 0};
            uint32_t a2[10] = {(uint32_t)C.part.gpu, (uint32_t)(C.part.gpu >> 32), (uint32_t)C.out.gpu, (uint32_t)(C.out.gpu >> 32),
                               C.chunks, 0, 0, 0, 0, 0};
            uint32_t g1[3] = {C.chunks * 8, 1, 1}, g2[3] = {8, 1, 1}, t[3] = {256, 1, 1};
            if (wait_first) s.wait_kernels();
            if (only != 2) s.dispatch(kp, pcode.gpu, a1, 12, g1, t);
            if (!only) s.wait_kernels();
            if (only != 1) s.dispatch(kr, rcode.gpu, a2, 10, g2, t);
        };
        auto check = [&](const char* where) -> double {
            const float* O = (const float*)C.out.cpu;
            double worst = 0;
            int bad = 0, first = 0;
            for (int i = 0; i < 32 * 128; i++) {
                double e = std::fabs(O[i] - C.ref[i]);
                if (!(e == e)) e = 1e9;
                if (e > 1e-3 && bad++ == 0) first = i;
                worst = std::max(worst, e);
            }
            if (bad) {
                std::printf("  n=%u %s: WRONG, %d of 4096 values, worst error %.3g, first bad %d (head %d dim %d): gpu %g cpu %g\n", C.n,
                            where, bad, worst, first, first / 128, first % 128, ((const float*)C.out.cpu)[first], C.ref[first]);
                all_ok = false;
            }
            return worst;
        };
        double e_sys, e_vram;
        {
            std::memset(C.out.cpu, 0xff, 32 * 128 * 4);
            pm4::Stream s;
            add(s, C.kh, C.vh, false);
            if (!run(s, &gs)) return die(dev.err() + " (attn, system memory)");
            e_sys = check("system memory");
            std::memset(C.out.cpu, 0xff, 32 * 128 * 4);
            pm4::Stream s2;
            add(s2, C.kv, C.vv, false);
            if (!run(s2, &gs)) return die(dev.err() + " (attn, VRAM)");
            e_vram = check("VRAM");
        }
        auto timed = [&](int per_batch, int warm, std::vector<double>& st, int only = 0) -> bool {
            for (int rep = 0; rep < warm + 3; rep++) {
                pm4::Stream s;
                for (int i = 0; i < per_batch; i++) add(s, C.kv, C.vv, i > 0, only);
                double total;
                if (!run(s, &total)) return false;
                if (rep >= warm) st.push_back(total / per_batch);
            }
            std::sort(st.begin(), st.end());
            return true;
        };
        {   // warm the clock up: batches until 50 ms of GPU time (round 27: the
            // first length ran cold, and n = 1000 read 56 us first and 30 us second)
            double warm = 0;
            while (warm < 0.05) {
                pm4::Stream s;
                for (int i = 0; i < 60; i++) add(s, C.kv, C.vv, i > 0);
                double total;
                if (!run(s, &total)) return die(dev.err() + " (attn warm-up)");
                warm += total;
            }
        }
        std::vector<double> st, st100, stp, str_;
        if (!timed(20, 10, st) || !timed(60, 3, st100) || !timed(60, 3, stp, 1) || !timed(60, 3, str_, 2))
            return die(dev.err() + " (attn batches)");
        double e_steady = check("steady");
        std::printf("n=%-6u K+V %6.1f MB  err %.2e / %.2e / %.2e  %u splits; batches of 20: %7.1f us, %6.1f GB/s of K+V; of 60 (a ring holds about 70): %7.1f us; part alone %7.1f us, reduce alone %6.1f us\n", C.n,
                    2 * kvb / 1e6, e_sys, e_vram, e_steady, C.chunks, st[1] * 1e6, 2 * kvb / st[1] / 1e9, st100[1] * 1e6, stp[1] * 1e6, str_[1] * 1e6);
        std::fflush(stdout);
    }
    std::printf("attn: %s\n", all_ok ? "all values match the CPU" : "SOME VALUES WRONG");
    return all_ok ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "info";
    if (cmd == "gen-mv2") {
        std::fputs(ts5::direct::gen_mv_q4k2(argc > 2 ? std::atoi(argv[2]) : 4096, "slc dlc", argc > 3 ? std::atoi(argv[3]) : 0,
                                            argc > 4 ? std::atoi(argv[4]) : 1)
                       .c_str(),
                   stdout);
        return 0;
    }
    if (cmd == "gen-attn") {   // prints attn_part, or attn_reduce with "reduce"
        std::fputs((argc > 2 && std::string(argv[2]) == "reduce" ? ts5::direct::gen_attn_reduce() : ts5::direct::gen_attn_part()).c_str(), stdout);
        return 0;
    }
    if (cmd == "gen-xt") {   // prints to_xT (prefill.h) for K columns and T tokens
        {
            const int K = argc > 2 ? std::atoi(argv[2]) : 4096, T = argc > 3 ? std::atoi(argv[3]) : 16, NT = argc > 4 ? std::atoi(argv[4]) : 256;
            std::fputs((NT <= 1 ? ts5::direct::gen_to_xT2(K, T, NT == 1) : ts5::direct::gen_to_xT(K, T, NT)).c_str(), stdout);   // NT 1 / 0: to_xT2 with / without xT
        }
        return 0;
    }
    if (cmd == "gen-ops") {   // prints rmsnorm, add_rmsnorm, qk_post and silu_mul
        std::fputs((ts5::direct::gen_rmsnorm(4096, 1e-6f) + ts5::direct::gen_rmsnorm(4096, 1e-6f, true) +
                    ts5::direct::gen_qk_post(32, 1e-6f) + ts5::direct::gen_silu_mul() + ts5::direct::gen_argmax() +
                    ts5::direct::gen_embed_q4k(4096))
                       .c_str(),
                   stdout);
        return 0;
    }
    if (cmd == "gen-mvs") {   // a staged mat-vec (mv_stage.h): 0/1 Q4_K norm 4096 / SiLU 12288, 2/3 the same for Q6_K
        using ts5::direct::Stage;
        const int v = argc > 2 ? std::atoi(argv[2]) : 0;
        const Stage st = v & 1 ? Stage::Silu : Stage::Norm;
        const int cols = v & 1 ? 12288 : 4096;
        std::fputs((v & 2 ? ts5::direct::gen_mv_q6k(cols, 0, st) : ts5::direct::gen_mv_q4k(cols, "", 0, 1, true, st)).c_str(), stdout);
        return 0;
    }
    if (cmd == "gen-mm") {   // prints mm_q4k (or mm_q6k with a 4th argument "q6") for K columns and T tokens
        const int K = argc > 2 ? std::atoi(argv[2]) : 4096, T = argc > 3 ? std::atoi(argv[3]) : 16;
        const bool q6 = argc > 4 && std::string(argv[4]) == "q6";
        const int ex = argc > 5 ? std::atoi(argv[5]) : 0;   // gen-mm K T q4 EXP [SK]: mm_q4k's timing experiment (SK: mm_q4k_h's wave groups)
        std::fputs((q6 ? (ex == 12 || ex == 22 ? ts5::direct::gen_mm_q6k_h(K, T, argc > 6 ? std::atoi(argv[6]) : 1, ex == 12) : ex == 10 ? ts5::direct::gen_mm_q6k_sb(K, T) : ts5::direct::gen_mm_q6k(K, T)) : ex == 20 ? ts5::direct::gen_mm_q4k_w(K, T, argc > 6 ? std::atoi(argv[6]) : 1) : ex >= 12 ? ts5::direct::gen_mm_q4k_h(K, T, ex >= 16 ? 0 : ex - 12, argc > 6 ? std::atoi(argv[6]) : 1, ex == 17 ? 2 : 1, ex == 21 ? 64 : 32, ex != 22, ex == 25) : ex >= 10 ? ts5::direct::gen_mm_q4k_sb(K, T, ex - 9) : ts5::direct::gen_mm_q4k(K, T, 4, ex)).c_str(), stdout);
        return 0;
    }
    if (cmd == "gen-mv6") {   // prints mv_q6k for a column count [diag]
        std::fputs(ts5::direct::gen_mv_q6k(argc > 2 ? std::atoi(argv[2]) : 4096, argc > 3 ? std::atoi(argv[3]) : 0).c_str(), stdout);
        return 0;
    }
    if (cmd == "gen-mv") {   // prints mv_q4k for a column count
        std::fputs(ts5::direct::gen_mv_q4k(argc > 2 ? std::atoi(argv[2]) : 4096, "slc dlc", argc > 3 ? std::atoi(argv[3]) : 0,
                                           argc > 4 ? std::atoi(argv[4]) : 1)
                       .c_str(),
                   stdout);
        return 0;
    }
    if (cmd == "gen") {
        // Prints a generated read kernel: gen N wave64(0/1) [hints...] [chunked]
        int n = argc > 2 ? std::atoi(argv[2]) : 4;
        bool w64 = argc > 3 && std::atoi(argv[3]);
        std::string h;
        bool chunked = false;
        for (int i = 4; i < argc; i++) {
            if (std::string(argv[i]) == "chunked") chunked = true;
            else h += (h.empty() ? "" : " ") + std::string(argv[i]);
        }
        std::fputs(gen_read_kernel(n, h, w64, 256, chunked).c_str(), stdout);
        return 0;
    }
    if (cmd == "asm-only") {
        // Assembles every built-in kernel; needs no GPU (for CI).
        for (auto& kk : kAllKernels) {
            gfx10::Kernel k;
            std::string err;
            if (!gfx10::assemble(kk.src, k, &err)) return die(std::string(kk.name) + ": " + err);
            std::printf("%s: %zu words, %d vgprs, rsrc1 0x%08x rsrc2 0x%08x\n", kk.name, k.code.size(), k.vgprs, gfx10::pgm_rsrc1(k),
                        gfx10::pgm_rsrc2(k));
        }
        for (bool ch : {false, true})
            for (bool w64 : {false, true})
                for (int n : {1, 2, 3, 4, 8, 16})
                    for (const char* h : {"", "slc", "glc slc dlc", "glc dlc", "slc dlc", "glc slc"}) {
                        gfx10::Kernel k;
                        std::string err;
                        if (!gfx10::assemble(gen_read_kernel(n, h, w64, 256, ch), k, &err)) return die("read_gen: " + err);
                    }
        std::printf("read_gen: 144 variants ok\n");
        for (int cols : {2048, 4096, 12288, 14336}) {
            gfx10::Kernel k;
            std::string err;
            if (!gfx10::assemble(ts5::direct::gen_mv_q4k(cols), k, &err)) return die("mv_q4k " + std::to_string(cols) + ": " + err);
            std::printf("mv_q4k %d cols: %zu words, %d vgprs, lds %u\n", cols, k.code.size(), k.vgprs, k.lds_bytes);
        }
        for (int cols : {4096, 8192, 12288}) {
            gfx10::Kernel k;
            std::string err;
            if (!gfx10::assemble(ts5::direct::gen_mv_q4k2(cols), k, &err)) return die("mv_q4k2 " + std::to_string(cols) + ": " + err);
            std::printf("mv_q4k2 %d cols: %zu words, %d vgprs, lds %u\n", cols, k.code.size(), k.vgprs, k.lds_bytes);
        }
        for (const std::string& src : {ts5::direct::gen_attn_part(), ts5::direct::gen_attn_reduce()}) {
            gfx10::Kernel k;
            std::string err;
            if (!gfx10::assemble(src, k, &err)) return die("attn: " + err);
            std::printf("%s: %zu words, %d vgprs, %d sgprs\n", k.name.c_str(), k.code.size(), k.vgprs, k.sgprs);
        }
        for (int cols : {2048, 4096, 12288}) {
            gfx10::Kernel k;
            std::string err;
            if (!gfx10::assemble(ts5::direct::gen_mv_q6k(cols), k, &err)) return die("mv_q6k " + std::to_string(cols) + ": " + err);
            std::printf("mv_q6k %d cols: %zu words, %d vgprs, lds %u\n", cols, k.code.size(), k.vgprs, k.lds_bytes);
        }
        for (const std::string& src : {ts5::direct::gen_rmsnorm(4096, 1e-6f), ts5::direct::gen_rmsnorm(4096, 1e-6f, true),
                                       ts5::direct::gen_qk_post(32, 1e-6f), ts5::direct::gen_silu_mul(), ts5::direct::gen_argmax(),
                                       ts5::direct::gen_embed_q4k(4096)}) {
            gfx10::Kernel k;
            std::string err;
            if (!gfx10::assemble(src, k, &err)) return die("ops: " + err);
            std::printf("%s: %zu words, %d vgprs, %d sgprs\n", k.name.c_str(), k.code.size(), k.vgprs, k.sgprs);
        }
        return 0;
    }
    if (cmd == "token") return ts5::direct::cmd_token(argc, argv);
    if (cmd == "mm-q4k" || cmd == "mm-q6k") return ts5::direct::cmd_mm_q4k(argc, argv);
    if (cmd == "xt-bench") return ts5::direct::cmd_xt_bench(argc, argv);
    if (cmd == "dot-test") return ts5::direct::cmd_dot_test(argc, argv);
    if (cmd == "fma-peak") return ts5::direct::cmd_fma_peak(argc, argv);
    kfd::Device dev;
    if (!dev.open()) return die(dev.err());
    if (cmd == "info") return cmd_info(dev);
    if (cmd == "fill") return cmd_fill(dev);
    if (cmd == "read-bw") return cmd_read_bw(dev, argc > 2 ? (size_t)std::atol(argv[2]) : 1024);
    if (cmd == "bw-sweep") {
        // bw-sweep [MiB]: around round 3's best (3 loads, 1 workgroup per CU,
        // glc dlc / slc dlc: 433.5 GB/s), now with bigger workgroups, and every
        // memory kind. 5 runs each.
        std::vector<ReadCfg> cfgs;
        for (int tpw : {256, 512, 1024})
            for (int n : {3, 4, 6})
                for (const char* h : {"glc dlc", "slc dlc"})
                    for (int pc : {1, 2}) cfgs.push_back({n, h, false, pc, tpw});
        return cmd_bw_sweep(dev, argc > 2 ? (size_t)std::atol(argv[2]) : 1024,
                            {kfd::Mem::Vram, kfd::Mem::GttUncached, kfd::Mem::Gtt}, cfgs, 5);
    }
    if (cmd == "bw-last") {
        // bw-last [MiB] [runs]: last read pass. Grid vs chunked slices, the
        // always-slow "3 loads glc dlc" row, more workgroups per CU, and where
        // each workgroup ran (to explain the 254 GB/s runs). VRAM only.
        std::vector<ReadCfg> cfgs;
        for (bool ch : {false, true}) {
            cfgs.push_back({4, "slc dlc", false, 1, 256, ch});
            cfgs.push_back({4, "glc dlc", false, 1, 256, ch});
            cfgs.push_back({3, "glc dlc", false, 1, 256, ch});
            cfgs.push_back({4, "slc dlc", false, 2, 256, ch});
            cfgs.push_back({4, "slc dlc", false, 4, 256, ch});
            cfgs.push_back({4, "slc dlc", false, 8, 128, ch});
        }
        return cmd_bw_sweep(dev, argc > 2 ? (size_t)std::atol(argv[2]) : 1024, {kfd::Mem::Vram}, cfgs,
                            argc > 3 ? std::atoi(argv[3]) : 30);
    }
    if (cmd == "attn") {   // attn [n ...]: positions in the cache (default 1000 4096 16384)
        std::vector<uint32_t> lens;
        for (int i = 2; i < argc; i++) lens.push_back((uint32_t)std::atol(argv[i]));
        if (lens.empty()) lens = {1000, 4096, 16384};
        return cmd_attn(dev, lens);
    }
    if (cmd == "mv-q6k" || cmd == "mv-q6k-diag")
        return cmd_mv_q4k(dev, argc > 2 ? std::atoi(argv[2]) : 30, argc > 3 ? std::atoi(argv[3]) : 1, cmd == "mv-q6k-diag", false, true);
    if (cmd == "mv-q4k" || cmd == "mv-diag")
        return cmd_mv_q4k(dev, argc > 2 ? std::atoi(argv[2]) : 30, argc > 3 ? std::atoi(argv[3]) : 1, cmd == "mv-diag",
                          argc > 4 && std::string(argv[4]) == "v2");
    if (cmd == "bw-repeat") {
        // bw-repeat N [wave64]: one good setup N times (hang hunt), VRAM
        int n = argc > 2 ? std::atoi(argv[2]) : 500;
        bool w64 = argc > 3 && std::string(argv[3]) == "wave64";
        return cmd_bw_sweep(dev, 1024, {kfd::Mem::Vram}, {{3, "slc dlc", w64, 1, 256}}, n);
    }
    return die("unknown command " + cmd + " (info, fill, read-bw [MiB], bw-sweep [MiB], bw-last [MiB] [runs], bw-repeat [N] [wave64], mv-q4k [runs] [wgs/CU])");
}
