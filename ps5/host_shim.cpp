// The console functions ps5/title.cpp calls, on Linux, so ps5/host-test.sh
// can run the title's own main against a folder that stands in for /data.
//
//   LoadExec("exit")            ends the process
//   LoadExec("/app0/eboot.bin") starts this binary again, as the console does
//   notifications               go to stdout (the title's log) and to
//                               TS5_DATA/notifications.txt
//   vk_icdGetInstanceProcAddr   the system Vulkan loader's entry point
//   the controller              never there: the host test presses buttons
//                               through the press file
//   the resolver, kern.arandom  getaddrinfo and /dev/urandom (engine/src/net.cpp)
//   audio out and in            no port opens, so the audio test reports no
//                               audio
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netdb.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {

int sceSystemServiceLoadExec(const char *path, const char *const *) {
    fflush(nullptr);
    if (strcmp(path, "exit") == 0) _exit(0);
    char *argv[] = {const_cast<char *>("eboot.bin"), nullptr};
    execv("/proc/self/exe", argv);
    _exit(127);
}

int sceSystemServiceHideSplashScreen(void) { return 0; }

int sceKernelSendNotificationRequest(int, void *request, size_t, int) {
    const char *text = static_cast<const char *>(request) + 45;
    if (FILE *f = fopen(TS5_DATA "/notifications.txt", "a")) {
        fprintf(f, "%s\n", text);
        fclose(f);
    }
    return 0;
}

int sceKernelUsleep(unsigned int microseconds) { return usleep(microseconds); }

int sceNetInit(void) { return 0; }
int sceNetPoolCreate(const char *, int, int) { return 1; }
int sceNetResolverCreate(const char *, int, int) { return 1; }
int sceNetResolverDestroy(int) { return 0; }
int sceNetResolverStartNtoa(int, const char *host, uint32_t *addr, int, int, int) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    if (getaddrinfo(host, nullptr, &hints, &res) != 0) return -1;
    *addr = ((sockaddr_in *)res->ai_addr)->sin_addr.s_addr;
    freeaddrinfo(res);
    return 0;
}
// kern.arandom only (glibc has no sysctl).
int sysctl(const int *name, unsigned namelen, void *old, size_t *oldlen, const void *, size_t) {
    if (namelen != 2 || name[0] != 1 || name[1] != 37) return -1;
    int fd = open("/dev/urandom", O_RDONLY);
    ssize_t n = fd < 0 ? -1 : read(fd, old, *oldlen);
    if (fd >= 0) close(fd);
    if (n <= 0) return -1;
    *oldlen = (size_t)n;
    return 0;
}

// The console's memory pools have no Linux counterpart: the log says unknown.
size_t sceKernelGetDirectMemorySize(void) { return 0; }
int sceKernelAvailableDirectMemorySize(off_t, off_t, size_t, off_t *, size_t *) { return -1; }
int sceKernelAvailableFlexibleMemorySize(size_t *) { return -1; }

int sceAudioOutInit(void) { return 0; }
int sceAudioOutOpen(int, int, int, uint32_t, uint32_t, uint32_t) { return (int)0x80260005; }  // no port
int sceAudioOutOutput(int, const void *) { return -1; }
int sceAudioOutSetVolume(int, int, int *) { return -1; }
int sceAudioOutClose(int) { return -1; }
int sceAudioInOpen(int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return (int)0x80260109; }  // no mic
int sceAudioInInput(int, void *) { return -1; }
int sceAudioInClose(int) { return -1; }

int sceUserServiceInitialize(const void *) { return 0; }
int sceUserServiceGetInitialUser(int *user) {
    *user = 1;
    return 0;
}
int scePadInit(void) { return 0; }
int scePadOpen(int, int, int, const void *) { return -1; }
int scePadRead(int, void *, int) { return 0; }

PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name) {
    static PFN_vkGetInstanceProcAddr gipa = [] {
        void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        return lib ? (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr") : nullptr;
    }();
    return gipa ? gipa(instance, name) : nullptr;
}
}
