// The direct engine's line to the GPU: the kernel's compute interface
// (/dev/kfd) for memory and a user-mode compute queue. After setup, work
// goes to the GPU by writing packets into the queue's ring and its
// doorbell, with no system call per kernel.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ts5::kfd {

struct Buffer {
    uint64_t gpu = 0;      // GPU address (the same as the CPU address)
    void* cpu = nullptr;   // CPU mapping, or null for GPU-only memory
    size_t size = 0;
    uint64_t handle = 0;
};

enum class Mem {
    Gtt,          // system memory, CPU-mapped, cached on the GPU
    GttUncached,  // system memory the CPU and GPU see at once (queues, fences)
    Vram,         // the GPU's own carve-out, not CPU-mapped
};

class Device {
public:
    ~Device();
    // Opens the first GPU node. On failure returns false and says why in err().
    bool open();
    const std::string& err() const { return err_; }

    // What the kernel reports about the GPU (KFD topology properties).
    std::string name;
    uint32_t gpu_id = 0, kfd_major = 0, kfd_minor = 0;
    uint32_t simd_count = 0, simd_per_cu = 0, max_waves_per_simd = 0, num_xcc = 1, drm_minor = 0;
    uint32_t gfx_target = 0;   // e.g. 100103 for gfx1013
    uint32_t cu_count() const { return simd_per_cu ? simd_count / simd_per_cu / (num_xcc ? num_xcc : 1) : 0; }
    std::vector<std::pair<std::string, uint64_t>> props;

    bool alloc(size_t size, Mem kind, Buffer& out);
    void free(Buffer& b);

    // The compute queue. submit() copies the packets into the ring and rings
    // the doorbell; wait() polls a fence value written by a RELEASE_MEM and
    // reports a GPU memory fault or hang instead of waiting forever.
    bool create_queue(size_t ring_bytes = 1 << 16);
    bool submit(const std::vector<uint32_t>& words);
    bool wait(volatile uint64_t* fence, uint64_t value, int timeout_ms);

    // Memory the queue's packets may write fences and timestamps to.
    Buffer signals;

private:
    bool fail(const std::string& why);
    bool read_topology();
    bool map_to_gpu(uint64_t handle);
    int kfd_ = -1, drm_ = -1;
    std::string err_;
    std::vector<Buffer> owned_;
    // queue
    Buffer ring_, rptr_, wptr_, eop_, cwsr_, event_page_;
    uint32_t queue_id_ = 0;
    bool have_queue_ = false;
    volatile uint64_t* doorbell_ = nullptr;
    void* doorbells_map_ = nullptr;
    uint64_t put_ = 0;   // dwords written to the ring so far
    uint32_t ev_memory_ = 0, ev_hw_ = 0;
};

}  // namespace ts5::kfd
