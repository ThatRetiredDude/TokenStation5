// The console functions ps5/title.cpp calls, on Linux, so ps5/host-test.sh
// can run the title's own main against a folder that stands in for /data.
//
//   LoadExec("exit")            ends the process
//   LoadExec("/app0/eboot.bin") starts this binary again, as the console does
//   notifications               go to stdout (the title's log) and to
//                               TS5_DATA/notifications.txt
//   vk_icdGetInstanceProcAddr   the system Vulkan loader's entry point
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
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

PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name) {
    static PFN_vkGetInstanceProcAddr gipa = [] {
        void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        return lib ? (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr") : nullptr;
    }();
    return gipa ? gipa(instance, name) : nullptr;
}
}
