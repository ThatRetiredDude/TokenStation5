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
//   the mic                     none, unless the host test puts a 16 kHz mono
//                               WAV at TS5_HOME/mic.wav: then the mic plays
//                               it, in real time, then silence
//   the on-screen keyboard      open until the host test writes
//                               TS5_HOME/typed: its text is what was typed,
//                               and "(cancel)" cancels
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

int sceSystemServicePowerTick(void) { return 0; }

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
static FILE *mic;
int sceAudioInOpen(int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {
    FILE *f = fopen(TS5_HOME "/mic.wav", "rb");
    if (!f) return (int)0x80260109;  // no mic
    // To the samples: past the RIFF header and every chunk before "data".
    char id[4];
    uint32_t size;
    fseek(f, 12, SEEK_SET);
    while (fread(id, 1, 4, f) == 4 && fread(&size, 4, 1, f) == 1 && memcmp(id, "data", 4) != 0) fseek(f, size, SEEK_CUR);
    mic = f;
    return 0x30000007;
}
int sceAudioInInput(int, void *samples) {
    usleep(256 * 1000000 / 16000);  // a grain's time
    const size_t n = mic ? fread(samples, 2, 256, mic) : 0;
    memset(static_cast<int16_t *>(samples) + n, 0, (256 - n) * 2);
    return 0;
}
int sceAudioInClose(int) {
    if (mic) fclose(mic);
    mic = nullptr;
    return 0;
}

int sceUserServiceInitialize(const void *) { return 0; }
int sceUserServiceGetInitialUser(int *user) {
    *user = 1;
    return 0;
}
int sceUserServiceGetForegroundUser(int *user) {
    *user = 1;
    return 0;
}

int sceCommonDialogInitialize(void) { return 0; }
int sceSysmoduleLoadModule(uint16_t) { return 0; }
// The dialog's text buffer and its length, while it's open.
static char16_t *typed_into;
static uint32_t typed_max;
static int typed_end = -1;  // -1 until TS5_HOME/typed appears
int sceImeDialogInit(const void *param, const void *) {
    if (typed_into) return (int)0x80bc0001;  // busy
    const char *p = static_cast<const char *>(param);
    memcpy(&typed_max, p + 36, 4);
    memcpy(&typed_into, p + 40, sizeof typed_into);
    typed_end = -1;
    return 0;
}
int sceImeDialogGetStatus(void) {
    if (!typed_into) return 0;
    if (typed_end < 0) {
        FILE *f = fopen(TS5_HOME "/typed", "rb");
        if (!f) return 1;  // running
        char buf[8192];
        const size_t n = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        unlink(TS5_HOME "/typed");
        buf[n] = 0;
        typed_end = strcmp(buf, "(cancel)") == 0 ? 1 : 0;
        // ASCII is all the host test types.
        uint32_t i = 0;
        for (; typed_end == 0 && i < typed_max && buf[i]; i++) typed_into[i] = (char16_t)(unsigned char)buf[i];
        typed_into[i] = 0;
    }
    return 2;  // finished
}
int sceImeDialogGetResult(void *result) {
    if (!typed_into || typed_end < 0) return (int)0x80bc0002;
    memset(result, 0, 16);
    memcpy(result, &typed_end, 4);
    return 0;
}
int sceImeDialogTerm(void) {
    typed_into = nullptr;
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
