// ts5-direct: tries the direct engine on the GPU.
//   ts5-direct info              what the kernel reports about the GPU
//   ts5-direct fill              smoke test: a kernel fills a buffer, the CPU checks it
//   ts5-direct read-bw [MiB]     read bandwidth on the direct path (default 1024 MiB)
// Kernels are assembled at start from the text built into the program.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gfx10_asm.h"
#include "kernels.h"   // generated: kernel sources as strings
#include "kfd.h"
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

}  // namespace

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "info";
    if (cmd == "asm-only") {
        // Assembles every built-in kernel; needs no GPU (for CI).
        for (auto& kk : kAllKernels) {
            gfx10::Kernel k;
            std::string err;
            if (!gfx10::assemble(kk.src, k, &err)) return die(std::string(kk.name) + ": " + err);
            std::printf("%s: %zu words, %d vgprs, rsrc1 0x%08x rsrc2 0x%08x\n", kk.name, k.code.size(), k.vgprs, gfx10::pgm_rsrc1(k),
                        gfx10::pgm_rsrc2(k));
        }
        return 0;
    }
    kfd::Device dev;
    if (!dev.open()) return die(dev.err());
    if (cmd == "info") return cmd_info(dev);
    if (cmd == "fill") return cmd_fill(dev);
    if (cmd == "read-bw") return cmd_read_bw(dev, argc > 2 ? (size_t)std::atol(argv[2]) : 1024);
    return die("unknown command " + cmd + " (info, fill, read-bw [MiB])");
}
