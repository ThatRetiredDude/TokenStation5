// TokenStation5's PS5 title: ts5 as a native app, on the console's GPU.
//
// The console starts a title with no arguments, so this one reads its command
// line from /data/ts5/ts5.args: one ts5 command line, as on Linux, such as
//
//   serve -m /data/ts5/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf -c 8192
//
// Without that file it serves the first model it finds in /data/ts5/models,
// then in /data/PS5LM/models. Everything ts5 prints goes to /data/ts5/ts5.log
// (the previous run's is ts5.prev.log), and a notification on the TV says
// when the API is up and where.
//
// The PC drives it over FTP (scripts/ts5-ps5.sh): a file named quit in
// /data/ts5 closes the title, and restart starts it again with the args file
// as it is then. /data/ts5/ts5.once holds a one-off command (check, bench):
// the next start runs it instead, then restarts into the args file's.
//
// A title must never return from main or call exit(): the console reports
// "Something went wrong" and can lose the app. It leaves with
// sceSystemServiceLoadExec("exit") instead.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "cli.h"

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif

extern "C" {
int sceSystemServiceLoadExec(const char *path, const char *const *args);
int sceSystemServiceHideSplashScreen(void);
int sceKernelSendNotificationRequest(int device, void *request, size_t size, int blocking);
int sceKernelUsleep(unsigned int microseconds);

// RADV is linked in whole and there is no Vulkan loader: its ICD entry point
// answers vkGetInstanceProcAddr for every command, and vk_backend.cpp starts
// from whatever ts5_vk_entry() returns.
PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name);
PFN_vkGetInstanceProcAddr ts5_vk_entry() { return vk_icdGetInstanceProcAddr; }
}

namespace {

// The console's /data; ps5/host-test.sh points it at a folder on the PC.
#ifndef TS5_DATA
#define TS5_DATA "/data"
#endif
constexpr const char *kDir = TS5_DATA "/ts5";
constexpr const char *kModels = TS5_DATA "/ts5/models";
constexpr const char *kArgs = TS5_DATA "/ts5/ts5.args";
constexpr const char *kOnce = TS5_DATA "/ts5/ts5.once";
constexpr const char *kLog = TS5_DATA "/ts5/ts5.log";
constexpr const char *kPrevLog = TS5_DATA "/ts5/ts5.prev.log";
constexpr const char *kQuit = TS5_DATA "/ts5/quit";
constexpr const char *kRestart = TS5_DATA "/ts5/restart";
const char *const kModelDirs[] = {kModels, TS5_DATA "/PS5LM/models"};

// A notification on the TV. The request is the system's layout: 45 bytes of
// header fields this title leaves at zero, then the text.
void notify(const std::string &text) {
    struct {
        char header[45];
        char text[3075];
    } request = {};
    snprintf(request.text, sizeof request.text, "%s", text.c_str());
    sceKernelSendNotificationRequest(0, &request, sizeof request, 0);
    printf("ts5-title: notify: %s\n", text.c_str());
}

[[noreturn]] void leave(const char *path) {
    printf("ts5-title: leaving (%s)\n", path);
    fflush(nullptr);
    sceSystemServiceLoadExec(path, nullptr);
    for (;;) sceKernelUsleep(1000000);
}

bool exists(const char *path) { return access(path, F_OK) == 0; }

// The PC's quit and restart requests, checked twice a second.
void watch_control_files() {
    for (;;) {
        if (exists(kQuit)) {
            unlink(kQuit);
            notify("TokenStation5: closing");
            leave("exit");
        }
        if (exists(kRestart)) {
            unlink(kRestart);
            notify("TokenStation5: restarting");
            leave("/app0/eboot.bin");
        }
        sceKernelUsleep(500000);
    }
}

// The args file split into words: whitespace separates them, double quotes
// group them, and lines starting with # are comments.
std::vector<std::string> read_args_file(const char *path) {
    std::vector<std::string> out;
    FILE *f = fopen(path, "r");
    if (!f) return out;
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        const char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#') continue;
        std::string word;
        bool in_word = false, quoted = false;
        for (; *p; p++) {
            if (*p == '"') {
                quoted = !quoted;
                in_word = true;
            } else if (!quoted && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
                if (in_word) out.push_back(word);
                word.clear();
                in_word = false;
            } else {
                word += *p;
                in_word = true;
            }
        }
        if (in_word) out.push_back(word);
    }
    fclose(f);
    return out;
}

// The first GGUF by name in the model folders, skipping vision projectors and
// split files, which ts5 does not load.
std::string find_model() {
    for (const char *dir : kModelDirs) {
        DIR *d = opendir(dir);
        if (!d) continue;
        std::vector<std::string> names;
        while (dirent *e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0 && n.find("mmproj") == std::string::npos &&
                n.find("-of-0") == std::string::npos)
                names.push_back(n);
        }
        closedir(d);
        if (!names.empty()) return std::string(dir) + "/" + *std::min_element(names.begin(), names.end());
    }
    return {};
}

// The console's address on the LAN: the source address a UDP socket would
// use. Connecting a UDP socket sends nothing.
std::string lan_address() {
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) return {};
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(53);
    inet_pton(AF_INET, "192.0.2.1", &to.sin_addr);
    sockaddr_in me{};
    socklen_t len = sizeof me;
    char buf[INET_ADDRSTRLEN] = {};
    if (connect(s, (sockaddr *)&to, sizeof to) != 0 || getsockname(s, (sockaddr *)&me, &len) != 0 ||
        !inet_ntop(AF_INET, &me.sin_addr, buf, sizeof buf))
        buf[0] = 0;
    close(s);
    return buf;
}

}  // namespace

int main(int, char **) {
    mkdir(kDir, 0777);
    mkdir(kModels, 0777);
    rename(kLog, kPrevLog);
    // Both in append mode, so neither writes over the other's lines.
    if (freopen(kLog, "a", stdout)) setvbuf(stdout, nullptr, _IOLBF, 0);
    if (freopen(kLog, "a", stderr)) setvbuf(stderr, nullptr, _IONBF, 0);
    printf("ts5-title: TokenStation5 %s\n", TS5_VERSION);

    // RADV's one heap over the console's unified memory: 11.44 GiB for the
    // GPU instead of a split. Read when the Vulkan instance is created.
    setenv("radv_enable_unified_heap_on_apu", "true", 1);
    setenv("HOME", kDir, 0);

    unlink(kQuit);
    unlink(kRestart);
    std::thread(watch_control_files).detach();
    sceSystemServiceHideSplashScreen();

    // A one-off command is taken before it runs: one that brings the title
    // down must not run again at every launch.
    std::vector<std::string> args = read_args_file(kOnce);
    const bool once = !args.empty();
    unlink(kOnce);
    if (!once) args = read_args_file(kArgs);
    if (args.empty()) {
        const std::string model = find_model();
        if (model.empty()) {
            notify("TokenStation5: no model. Copy a .gguf to /data/ts5/models");
            sceKernelUsleep(3000000);
            leave("exit");
        }
        args = {"serve", "-m", model};
    }
    std::string line;
    for (const auto &a : args) line += " " + a;
    printf("ts5-title: ts5%s\n", line.c_str());

    std::vector<char *> argv{const_cast<char *>("ts5")};
    for (auto &a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    const std::string &cmd = args[0];
    std::string model;
    for (size_t i = 0; i + 1 < args.size(); i++)
        if (args[i] == "-m") model = args[i + 1].substr(args[i + 1].find_last_of('/') + 1);
    notify("TokenStation5: loading " + model);

    int rc = 1;
    try {
        rc = ts5::cli_main((int)argv.size() - 1, argv.data(), [](const std::string &name, int port) {
            const std::string ip = lan_address();
            notify("TokenStation5: " + name + " ready at http://" + (ip.empty() ? "<console ip>" : ip) + ":" +
                   std::to_string(port) + "/v1");
        });
    } catch (const std::exception &e) {
        // An uncaught exception would end the process the way a title must not.
        printf("ts5-title: %s\n", e.what());
    }
    printf("ts5-title: %s ended with status %d\n", cmd.c_str(), rc);
    // A one-off run's log is the previous one once the title has restarted.
    const std::string log = once ? "/data/ts5/ts5.prev.log" : "/data/ts5/ts5.log";
    if (cmd == "check")
        notify(rc == 0 ? "TokenStation5: GPU check passed" : "TokenStation5: GPU check FAILED, see " + log);
    else if (rc != 0)
        notify("TokenStation5 stopped with an error, see " + log);
    else
        notify("TokenStation5: " + cmd + " done, see " + log);
    sceKernelUsleep(3000000);
    leave(once ? "/app0/eboot.bin" : "exit");
}
