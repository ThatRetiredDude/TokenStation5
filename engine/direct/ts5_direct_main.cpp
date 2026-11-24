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
#include "pm4.h"

using namespace ts5;

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
int cmd_mv_q4k(kfd::Device& dev, int runs, int per_cu, bool diag = false, bool v2 = false) {
    auto gen = [v2](int cols, const std::string& hints, int d, int depth) {
        return v2 ? ts5::direct::gen_mv_q4k2(cols, hints, d, depth) : ts5::direct::gen_mv_q4k(cols, hints, d, depth);
    };
    std::printf("kernel: %s\n", v2 ? "mv_q4k2 (2 lanes per block)" : "mv_q4k (8 lanes per block, headers by one dword load)");
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
    const Shape shapes[] = {{"attn_q/attn_output 4096x4096", 4096, 4096}, {"attn_k 1024x4096", 1024, 4096},
                            {"ffn_gate/up 12288x4096", 12288, 4096}, {"ffn_down 4096x12288", 4096, 12288}};
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
    std::vector<Prep> preps(sizeof(shapes) / sizeof(shapes[0]));
    for (size_t si = 0; si < preps.size(); si++) {
        Prep& P = preps[si];
        const Shape& sh = shapes[si];
        P.sh = &sh;
        const size_t nb = sh.cols / 256, row_bytes = nb * 144, bytes = row_bytes * sh.rows;
        std::string err;
        if (!gfx10::assemble(gen((int)sh.cols, "slc dlc", 0, 1), P.k, &err)) return die("mv_q4k: " + err);
        size_t cb = (P.k.code.size() * 4 + 256 + 4095) & ~(size_t)4095;
        if (!dev.alloc(cb, kfd::Mem::Gtt, P.code) || !dev.alloc(cb * 2, kfd::Mem::Gtt, P.dcode) ||
            !dev.alloc((size_t)dev.cu_count() * per_cu * 8 * 16, kfd::Mem::Gtt, P.ts)) return die(dev.err());
        for (size_t i = 0; i < cb / 4; i++) ((uint32_t*)P.code.cpu)[i] = 0xbf9f0000;
        std::memcpy(P.code.cpu, P.k.code.data(), P.k.code.size() * 4);
        if (!dev.alloc(bytes, kfd::Mem::Gtt, P.wh) || !dev.alloc(bytes, kfd::Mem::Vram, P.wv) ||
            !dev.alloc(sh.cols * 4, kfd::Mem::Gtt, P.xb) || !dev.alloc(sh.rows * 4, kfd::Mem::Gtt, P.yb))
            return die(dev.err());
        uint8_t* W = (uint8_t*)P.wh.cpu;
        for (size_t b = 0; b < nb * sh.rows; b++) {
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
        for (uint32_t r = 0; r < sh.rows; r++) {
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
        const size_t bytes = sh.cols / 256 * 144 * sh.rows;
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
        const uint32_t wgs = dev.cu_count() * per_cu, per = (sh.rows + wgs - 1) / wgs;
        auto mv = [&](const kfd::Buffer& w, double* t_s) -> bool {
            std::memset(P.yb.cpu, 0xff, sh.rows * 4);
            pm4::Stream s;
            uint32_t args[10] = {(uint32_t)w.gpu, (uint32_t)(w.gpu >> 32), (uint32_t)P.xb.gpu, (uint32_t)(P.xb.gpu >> 32),
                                 (uint32_t)P.yb.gpu, (uint32_t)(P.yb.gpu >> 32), sh.rows, sh.cols, per, 0};
            uint32_t g[3] = {wgs, 1, 1}, t[3] = {256, 1, 1};
            s.dispatch(k, P.code.gpu, args, 10, g, t);
            return run(s, t_s);
        };
        // Steady state: n dispatches back to back in one submission, each
        // waiting for the one before (as a token's kernels do), timed as a
        // whole. Single dispatches with the CPU waiting between them let the
        // GPU clock drop: round 11's wave records ran at about 490 MHz on
        // the short shapes and 1.6-2.3 GHz on the long ones.
        auto batch = [&](const gfx10::Kernel& kk, uint64_t code_gpu, const kfd::Buffer& w, int n, double* per_s) -> bool {
            pm4::Stream s;
            uint32_t args[10] = {(uint32_t)w.gpu, (uint32_t)(w.gpu >> 32), (uint32_t)P.xb.gpu, (uint32_t)(P.xb.gpu >> 32),
                                 (uint32_t)P.yb.gpu, (uint32_t)(P.yb.gpu >> 32), sh.rows, (uint32_t)P.ts.gpu, per,
                                 (uint32_t)(P.ts.gpu >> 32)};
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
        for (int i = 0; i < 3; i++)
            if (!batch(k, P.code.gpu, P.wv, 100, &per_s)) return die(dev.err() + " (warm-up)");
        std::vector<double> st;
        for (int i = 0; i < 3; i++) {
            if (!batch(k, P.code.gpu, P.wv, 100, &per_s)) return die(dev.err() + " (steady)");
            st.push_back(per_s);
        }
        std::sort(st.begin(), st.end());
        double e_steady = check("steady");
        std::printf("%-30s %6.1f %9.2e  %6.1f / %6.1f / %6.1f  %8.1f   steady %6.1f GB/s, %7.1f us  (err %.2e)\n", sh.name, bytes / 1e6, e,
                    r.back(), med, r.front(), bytes / med / 1e3, bytes / st[1] / 1e9, st[1] * 1e6, e_steady);
        std::fflush(stdout);
        // diag: the same kernel with parts left out (results not checked),
        // then with 1 and 3 steps of loads in flight (results checked).
        for (int d = 1; diag && d < 12; d++) {
            // 1-7: parts left out; 8: timestamps; 9: timestamps, nothing but the frame; 10, 11: 2 and 3 steps in flight
            const int mode = d <= 8 ? d : d == 9 ? 15 : 0, depth = d == 10 ? 2 : d == 11 ? 3 : 1;
            gfx10::Kernel dk;
            std::string err;
            if (!gfx10::assemble(gen((int)sh.cols, "slc dlc", mode, depth), dk, &err))
                return die("mv_q4k diag: " + err);
            if (dk.code.size() * 4 + 256 > P.dcode.size) return die("diag code too big");
            for (size_t i = 0; i < P.dcode.size / 4; i++) ((uint32_t*)P.dcode.cpu)[i] = 0xbf9f0000;
            std::memcpy(P.dcode.cpu, dk.code.data(), dk.code.size() * 4);
            std::vector<double> dt;   // steady: right after the warm batches above, 3 batches of `runs`
            std::memset(P.yb.cpu, 0xff, sh.rows * 4);
            for (int i = 0; i < 3; i++) {
                if (!batch(dk, P.dcode.gpu, P.wv, runs, &gs)) return die(dev.err() + " (diag " + std::to_string(d) + ")");
                dt.push_back(gs * 1e6);
            }
            std::sort(dt.begin(), dt.end());
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
                std::printf("    without%s%s%s: %8.1f us median\n", d & 1 ? " math" : "", d & 2 ? " lds-reads" : "",
                            d & 4 ? " weight-loads" : "", dt[dt.size() / 2]);
            else {
                double e2 = check(depth == 2 ? "depth 2" : "depth 3");
                std::printf("    %d steps in flight: %8.1f us median, %6.1f GB/s, err %.2e\n", depth, dt[dt.size() / 2],
                            bytes / dt[dt.size() / 2] / 1e3, e2);
            }
            std::fflush(stdout);
        }
    }
    std::printf("mv-q4k: %s (%u workgroups, 256 threads each)\n", all_ok ? "all rows match the CPU" : "SOME ROWS WRONG",
                dev.cu_count() * per_cu);
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "info";
    if (cmd == "gen-mv2") {
        std::fputs(ts5::direct::gen_mv_q4k2(argc > 2 ? std::atoi(argv[2]) : 4096, "slc dlc", argc > 3 ? std::atoi(argv[3]) : 0,
                                            argc > 4 ? std::atoi(argv[4]) : 1)
                       .c_str(),
                   stdout);
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
        return 0;
    }
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
