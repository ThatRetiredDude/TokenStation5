// /dev/kfd: memory and a user-mode compute queue (see kfd.h).
//
// Queue buffers follow the kernel's own checks (kfd_queue.c): the ring, the
// read/write pointer page, the end-of-pipe buffer and the context save area
// are each their own GPU allocation of the size the kernel expects.
#include "kfd.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <linux/kfd_ioctl.h>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace ts5::kfd {
namespace {

constexpr const char* kTopo = "/sys/devices/virtual/kfd/kfd/topology/nodes";

int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do r = ioctl(fd, req, arg);
    while (r == -1 && (errno == EINTR || errno == EAGAIN));
    return r;
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

bool Device::fail(const std::string& why) {
    err_ = why;
    if (errno) err_ += std::string(" (") + std::strerror(errno) + ")";
    return false;
}

bool Device::read_topology() {
    DIR* d = opendir(kTopo);
    if (!d) return fail(std::string("no KFD topology at ") + kTopo);
    std::vector<int> nodes;
    while (auto* e = readdir(d)) {
        if (e->d_name[0] >= '0' && e->d_name[0] <= '9') nodes.push_back(std::atoi(e->d_name));
    }
    closedir(d);
    std::sort(nodes.begin(), nodes.end());
    for (int n : nodes) {
        std::string base = std::string(kTopo) + "/" + std::to_string(n);
        uint32_t id = (uint32_t)std::strtoul(read_file(base + "/gpu_id").c_str(), nullptr, 10);
        if (!id) continue;   // a CPU node
        gpu_id = id;
        name = read_file(base + "/name");
        while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
        std::istringstream ps(read_file(base + "/properties"));
        std::string k;
        uint64_t v;
        while (ps >> k >> v) {
            props.push_back({k, v});
            if (k == "simd_count") simd_count = (uint32_t)v;
            else if (k == "simd_per_cu") simd_per_cu = (uint32_t)v;
            else if (k == "max_waves_per_simd") max_waves_per_simd = (uint32_t)v;
            else if (k == "num_xcc") num_xcc = (uint32_t)v;
            else if (k == "drm_render_minor") drm_minor = (uint32_t)v;
            else if (k == "gfx_target_version") gfx_target = (uint32_t)v;
        }
        return true;
    }
    errno = 0;
    return fail("KFD lists no GPU");
}

bool Device::open() {
    errno = 0;
    if (!read_topology()) return false;
    kfd_ = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (kfd_ < 0) return fail("can't open /dev/kfd");
    std::string drm = "/dev/dri/renderD" + std::to_string(drm_minor);
    drm_ = ::open(drm.c_str(), O_RDWR | O_CLOEXEC);
    if (drm_ < 0) return fail("can't open " + drm);
    kfd_ioctl_get_version_args ver{};
    if (xioctl(kfd_, AMDKFD_IOC_GET_VERSION, &ver)) return fail("KFD version");
    kfd_major = ver.major_version;
    kfd_minor = ver.minor_version;
    kfd_ioctl_acquire_vm_args vm{};
    vm.drm_fd = (uint32_t)drm_;
    vm.gpu_id = gpu_id;
    if (xioctl(kfd_, AMDKFD_IOC_ACQUIRE_VM, &vm)) return fail("KFD acquire VM");
#ifdef AMDKFD_IOC_RUNTIME_ENABLE
    if (kfd_major > 1 || kfd_minor >= 14) {
        kfd_ioctl_runtime_enable_args rt{};
        xioctl(kfd_, AMDKFD_IOC_RUNTIME_ENABLE, &rt);   // only needed by newer kernels; harmless to fail
    }
#endif
    errno = 0;
    return alloc(4096, Mem::GttUncached, signals);
}

bool Device::map_to_gpu(uint64_t handle) {
    uint32_t ids[1] = {gpu_id};
    kfd_ioctl_map_memory_to_gpu_args m{};
    m.handle = handle;
    m.device_ids_array_ptr = (uint64_t)(uintptr_t)ids;
    m.n_devices = 1;
    if (xioctl(kfd_, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &m) || m.n_success != 1) return fail("map memory to GPU");
    return true;
}

bool Device::alloc(size_t size, Mem kind, Buffer& out) {
    size = round_up(size, 4096);
    // The GPU address is the CPU address: reserve the range first.
    void* va = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (va == MAP_FAILED) return fail("reserve address range");
    uint32_t flags = KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
    switch (kind) {
    case Mem::Gtt: flags |= KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC; break;
    case Mem::GttUncached:
        flags |= KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                 KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
        break;
    case Mem::Vram: flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM; break;
    }
    kfd_ioctl_alloc_memory_of_gpu_args a{};
    a.va_addr = (uint64_t)(uintptr_t)va;
    a.size = size;
    a.gpu_id = gpu_id;
    a.flags = flags;
    if (xioctl(kfd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a)) {
        munmap(va, size);
        return fail("allocate " + std::to_string(size) + " bytes of GPU memory");
    }
    out = Buffer();
    out.gpu = a.va_addr;
    out.size = size;
    out.handle = a.handle;
    if (kind != Mem::Vram) {
        void* p = mmap(va, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, drm_, (off_t)a.mmap_offset);
        if (p == MAP_FAILED || p != va) return fail("map GPU memory for the CPU");
        out.cpu = p;
    }
    if (!map_to_gpu(a.handle)) return false;
    owned_.push_back(out);
    return true;
}

void Device::free(Buffer& b) {
    if (!b.handle) return;
    uint32_t ids[1] = {gpu_id};
    kfd_ioctl_unmap_memory_from_gpu_args u{};
    u.handle = b.handle;
    u.device_ids_array_ptr = (uint64_t)(uintptr_t)ids;
    u.n_devices = 1;
    xioctl(kfd_, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &u);
    kfd_ioctl_free_memory_of_gpu_args f{};
    f.handle = b.handle;
    xioctl(kfd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &f);
    munmap((void*)(uintptr_t)b.gpu, b.size);
    for (auto it = owned_.begin(); it != owned_.end(); ++it)
        if (it->handle == b.handle) { owned_.erase(it); break; }
    b = Buffer();
}

bool Device::create_queue(size_t ring_bytes) {
    errno = 0;
    // Context save area, sized the way the kernel checks it (kfd_queue.c, GFX10).
    uint32_t cus = cu_count();
    uint32_t waves = cus * max_waves_per_simd * simd_per_cu;
    size_t wg_data = round_up((size_t)cus * (0x40000 + 0x4000 + 0x10000 + 0x1000), 4096);
    size_t ctl_stack = round_up((size_t)waves * 12 + 8 + 40, 4096);
    size_t debug_mem = round_up((size_t)waves * 32, 64);
    size_t cwsr_size = wg_data + ctl_stack;
    if (!alloc(ring_bytes, Mem::GttUncached, ring_) || !alloc(4096, Mem::GttUncached, ptrs_) || !alloc(4096, Mem::Gtt, eop_) ||
        !alloc(round_up((cwsr_size + debug_mem) * (num_xcc ? num_xcc : 1), 4096), Mem::Gtt, cwsr_))
        return false;
    std::memset(ring_.cpu, 0, ring_.size);
    std::memset(ptrs_.cpu, 0, ptrs_.size);
    // Events: GPU memory faults and hardware errors, so wait() can say what went wrong.
    if (!alloc(0x8000, Mem::GttUncached, event_page_)) return false;
    kfd_ioctl_create_event_args ev{};
    ev.event_page_offset = event_page_.handle;
    ev.event_type = KFD_IOC_EVENT_SIGNAL;
    xioctl(kfd_, AMDKFD_IOC_CREATE_EVENT, &ev);
    kfd_ioctl_create_event_args em{};
    em.event_type = KFD_IOC_EVENT_MEMORY;
    if (!xioctl(kfd_, AMDKFD_IOC_CREATE_EVENT, &em)) ev_memory_ = em.event_id;
    kfd_ioctl_create_event_args eh{};
    eh.event_type = KFD_IOC_EVENT_HW_EXCEPTION;
    if (!xioctl(kfd_, AMDKFD_IOC_CREATE_EVENT, &eh)) ev_hw_ = eh.event_id;
    errno = 0;

    kfd_ioctl_create_queue_args q{};
    q.ring_base_address = ring_.gpu;
    q.ring_size = (uint32_t)ring_.size;
    q.read_pointer_address = ptrs_.gpu;
    q.write_pointer_address = ptrs_.gpu + 8;
    q.gpu_id = gpu_id;
    q.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;   // PM4
    q.queue_percentage = KFD_MAX_QUEUE_PERCENTAGE;
    q.queue_priority = 7;
    q.eop_buffer_address = eop_.gpu;
    q.eop_buffer_size = eop_.size;
    q.ctx_save_restore_address = cwsr_.gpu;
    q.ctx_save_restore_size = (uint32_t)cwsr_size;
    q.ctl_stack_size = (uint32_t)ctl_stack;
    if (xioctl(kfd_, AMDKFD_IOC_CREATE_QUEUE, &q))
        return fail("create the compute queue (CUs " + std::to_string(cus) + ", waves " + std::to_string(waves) +
                    ", save area " + std::to_string(cwsr_size) + ", control stack " + std::to_string(ctl_stack) + ")");
    queue_id_ = q.queue_id;
    have_queue_ = true;
    // The doorbells are two pages; the queue's is at an offset in them.
    uint64_t base = q.doorbell_offset & ~(uint64_t)0x1fff;
    doorbells_map_ = mmap(nullptr, 0x2000, PROT_READ | PROT_WRITE, MAP_SHARED, kfd_, (off_t)base);
    if (doorbells_map_ == MAP_FAILED) { doorbells_map_ = nullptr; return fail("map the doorbell"); }
    doorbell_ = (volatile uint64_t*)((char*)doorbells_map_ + (q.doorbell_offset - base));
    put_ = 0;
    return true;
}

bool Device::submit(const std::vector<uint32_t>& words) {
    if (!have_queue_) { errno = 0; return fail("no queue"); }
    uint32_t* ring = (uint32_t*)ring_.cpu;
    size_t ring_dw = ring_.size / 4;
    volatile uint64_t* rptr = (volatile uint64_t*)ptrs_.cpu;
    if (words.size() >= ring_dw) { errno = 0; return fail("command stream longer than the ring"); }
    // Wait for room: the GPU's read pointer counts dwords consumed.
    auto t0 = std::chrono::steady_clock::now();
    while (put_ + words.size() - (*rptr) >= ring_dw) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) { errno = 0; return fail("the queue stopped taking work"); }
        std::this_thread::yield();
    }
    for (uint32_t w : words) ring[(put_++) % ring_dw] = w;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *(volatile uint64_t*)((char*)ptrs_.cpu + 8) = put_;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *doorbell_ = put_;
    return true;
}

bool Device::wait(volatile uint64_t* fence, uint64_t value, int timeout_ms) {
    auto t0 = std::chrono::steady_clock::now();
    int spins = 0;
    while (*fence < value) {
        if (++spins > 4096) {
            spins = 0;
            // Any fault reported?
            kfd_event_data evs[2]{};
            uint32_t n = 0;
            if (ev_memory_) evs[n++].event_id = ev_memory_;
            if (ev_hw_) evs[n++].event_id = ev_hw_;
            if (n) {
                kfd_ioctl_wait_events_args w{};
                w.events_ptr = (uint64_t)(uintptr_t)evs;
                w.num_events = n;
                w.wait_for_all = 0;
                w.timeout = 0;
                if (!xioctl(kfd_, AMDKFD_IOC_WAIT_EVENTS, &w) && w.wait_result == KFD_IOC_WAIT_RESULT_COMPLETE) {
                    for (uint32_t i = 0; i < n; i++) {
                        if (evs[i].event_id == ev_memory_ && evs[i].memory_exception_data.gpu_id) {
                            char buf[160];
                            std::snprintf(buf, sizeof buf, "GPU memory fault at 0x%llx (not present %u, read only %u, no execute %u)",
                                          (unsigned long long)evs[i].memory_exception_data.va,
                                          evs[i].memory_exception_data.failure.NotPresent,
                                          evs[i].memory_exception_data.failure.ReadOnly,
                                          evs[i].memory_exception_data.failure.NoExecute);
                            errno = 0;
                            return fail(buf);
                        }
                        if (evs[i].event_id == ev_hw_ && evs[i].hw_exception_data.gpu_id) {
                            errno = 0;
                            return fail("GPU hardware error (reset " + std::to_string(evs[i].hw_exception_data.reset_type) + ")");
                        }
                    }
                }
            }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
                errno = 0;
                return fail("timed out waiting for the GPU (fence " + std::to_string(*fence) + ", want " + std::to_string(value) +
                            ", ring read " + std::to_string(*(volatile uint64_t*)ptrs_.cpu) + " of " + std::to_string(put_) + ")");
            }
            std::this_thread::yield();
        }
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return true;
}

Device::~Device() {
    if (have_queue_) {
        kfd_ioctl_destroy_queue_args d{};
        d.queue_id = queue_id_;
        xioctl(kfd_, AMDKFD_IOC_DESTROY_QUEUE, &d);
    }
    if (doorbells_map_) munmap(doorbells_map_, 0x2000);
    while (!owned_.empty()) { Buffer b = owned_.back(); free(b); }
    if (drm_ >= 0) close(drm_);
    if (kfd_ >= 0) close(kfd_);
}

}  // namespace ts5::kfd
