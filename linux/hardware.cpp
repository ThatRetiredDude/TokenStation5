#include "hardware.h"

#include <dirent.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include "settings.h"

namespace linuxp {

namespace {

// From the .deb, or from the portable build's install.sh (/usr is read-only
// on Bazzite and SteamOS). The sudo rule names the one that's installed.
const char *helper() {
    if (const char *h = getenv("TS5_HW_HELPER")) return h;
    static const char *found =
        access("/usr/lib/tokenstation/hw", X_OK) == 0 ? "/usr/lib/tokenstation/hw" : "/usr/local/lib/tokenstation/hw";
    return found;
}
constexpr double kGiB = 1073741824.0;

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::string read_line(const std::string &path) {
    std::string s;
    if (FILE *f = fopen(path.c_str(), "r")) {
        char b[256];
        if (fgets(b, sizeof b, f)) s = b;
        fclose(f);
    }
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

// Every hwmon temperature: "amdgpu edge", "k10temp Tctl" and the like.
std::vector<std::pair<std::string, double>> read_temps() {
    std::vector<std::pair<std::string, double>> out;
    DIR *d = opendir("/sys/class/hwmon");
    if (!d) return out;
    while (dirent *e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string dir = std::string("/sys/class/hwmon/") + e->d_name;
        const std::string chip = read_line(dir + "/name");
        for (int i = 1; i <= 16; i++) {
            const std::string v = read_line(dir + "/temp" + std::to_string(i) + "_input");
            if (v.empty()) continue;
            std::string label = read_line(dir + "/temp" + std::to_string(i) + "_label");
            if (label.empty()) label = "temp" + std::to_string(i);
            const double c = atof(v.c_str()) / 1000.0;
            if (c > -40 && c < 150) out.push_back({chip + " " + label, c});
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

bool present() {
    if (getenv("TS5_FAKE_PS5")) return true;  // tests on a machine that isn't one
    return access("/dev/mp1", F_OK) == 0 || access("/dev/icc", F_OK) == 0;
}

}  // namespace

// sudo -n HELPER ARGS..., its output in out; false if it failed (out then
// has why, from its stderr too).
bool Hardware::run(const std::vector<std::string> &args, std::string &out) {
    std::string cmd = "sudo -n ";
    cmd += helper();
    for (const std::string &a : args) cmd += " '" + a + "'";  // fixed words and numbers only
    cmd += " 2>&1";
    out.clear();
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) return false;
    char b[512];
    while (fgets(b, sizeof b, p)) out += b;
    const int st = pclose(p);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

void Hardware::start(Notify notify) {
    notify_ = std::move(notify);
    if (!present()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        info_.present = true;
    }
    std::thread([this] {
        // As the settings say, from the start: a desktop session has no
        // service that did it at boot.
        std::string out;
        const std::string fan = app::setting("TS5_PS5_FAN", "auto"), boost = app::setting("TS5_PS5_BOOST", "0");
        {
            std::lock_guard<std::mutex> lk(mu_);
            fan_mode_ = fan;
        }
        if (!run({"fan", fan == "max" || fan == "cool" ? fan : "auto"}, out))
            printf("ts5-title: hardware: fan: %s\n", out.c_str());
        if (!run({"boost", boost == "0" ? "off" : "on"}, out))
            printf("ts5-title: hardware: boost: %s\n", out.c_str());
        refresh();
    }).detach();
    watcher_ = std::thread([this] { watch(); });
}

Hardware::~Hardware() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (watcher_.joinable()) watcher_.join();
}

void Hardware::fan(const std::string &mode) {
    std::thread([this, mode] {
        std::string out;
        if (!run({"fan", mode}, out)) printf("ts5-title: hardware: fan %s: %s\n", mode.c_str(), out.c_str());
    }).detach();
}

// The PS5's own fan curve follows the board sensor, which reads 50-62 C
// while the die (k10temp Tctl, what the kernel's 85 C clock limiter
// watches) is at 85-94 C and being clipped: the fan sits near idle through
// a whole prompt (measured, 2026-11-04). Performance doesn't wait for a
// temperature: full speed the moment the engine starts working, the curve
// again 30 s after it stops, once the die reads under 70 C. And full speed
// whenever the die passes 75 C for any other reason.
void Hardware::work(bool busy) {
    bool to_max = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        working_ += busy ? 1 : -1;
        if (working_ < 0) working_ = 0;
        if (!busy) idle_at_ = now_s();
        if (busy && fan_mode_ == "perf" && !fan_max_) fan_max_ = to_max = true;
    }
    if (to_max) fan("max");
}

double die_c() {
    for (const auto &t : read_temps())
        if (t.first == "k10temp Tctl") return t.second;
    return -1;
}

void Hardware::watch() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!quit_) {
        cv_.wait_for(lk, std::chrono::seconds(2));
        if (quit_ || fan_mode_ != "perf") continue;
        if (fan_max_) {
            if (working_ == 0 && now_s() - idle_at_ > 30) {
                lk.unlock();
                const double c = die_c();
                lk.lock();
                if (c < 70 && working_ == 0 && fan_mode_ == "perf" && fan_max_) {
                    fan_max_ = false;
                    fan("auto");
                }
            }
        } else {
            lk.unlock();
            const double c = die_c();
            lk.lock();
            if (c > 75 && fan_mode_ == "perf" && !fan_max_) {
                fan_max_ = true;
                idle_at_ = now_s();
                fan("max");
            }
        }
    }
}

void Hardware::refresh() {
    std::string out;
    const bool ok = run({"status"}, out);
    std::map<std::string, std::string> kv;
    size_t at = 0;
    while (at < out.size()) {
        const size_t nl = out.find('\n', at);
        const std::string line = out.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        at = nl == std::string::npos ? out.size() : nl + 1;
        const size_t eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (!ok) {
        info_.problem = out.find("password") != std::string::npos || out.find("sudo") != std::string::npos
                            ? "To change these here, add your user to the tokenstation group (sudo adduser $USER "
                              "tokenstation) and log in again."
                            : "The PS5 hardware helper didn't run: " + out;
        printf("ts5-title: hardware: status: %s\n", out.c_str());
        return;
    }
    info_.problem = kv["ps5_control"].empty() ? "Fan and boost need ps5_control (ps5-linux-tools)." : "";
    const double vram = atof(kv["vram_total"].c_str()), gtt = atof(kv["gtt_total"].c_str());
    info_.ram_gib = atof(kv["ram_total"].c_str()) / kGiB;
    info_.vram_gib = vram / kGiB;
    info_.gtt_gib = gtt / kGiB;
    info_.gpu_gib = (vram + gtt) / kGiB;
    info_.boot_gtt_gib = atof(kv["boot_gtt_pages"].c_str()) * 4096 / kGiB;
    info_.cus_active = atoi(kv["cus_active"].c_str());
    const int mode = atoi(kv["boot_cu_mode"].c_str());
    info_.boot_cus = mode == 3 ? 40 : mode == 2 ? 38 : 36;
}

ui::HwInfo Hardware::info() {
    std::lock_guard<std::mutex> lk(mu_);
    if (info_.present && now_s() - temps_at_ > 2) {
        info_.temps = read_temps();
        temps_at_ = now_s();
    }
    return info_;
}

void Hardware::apply(const std::string &name, const std::string &value) {
    std::vector<std::string> args;
    std::string done;
    if (name == "TS5_PS5_FAN") {
        bool busy;
        {
            std::lock_guard<std::mutex> lk(mu_);
            fan_mode_ = value;
            busy = working_ > 0;
            fan_max_ = value == "perf" && busy;
        }
        args = {"fan", value == "max" || value == "cool" ? value : value == "perf" && busy ? "max" : "auto"};
        done = value == "max"    ? "The fan runs at full speed."
               : value == "cool" ? "The fan ramps up sooner: full speed from a 65 C board reading."
               : value == "perf" ? "The fan runs at full speed while a model works, and follows the PS5's curve when idle."
                                 : "The fan follows the PS5's own curve.";
    } else if (name == "TS5_PS5_BOOST") {
        args = {"boost", value == "0" ? "off" : "on"};
        done = value == "0" ? "Clock boost off." : "Clock boost on: CPU 3.5 GHz, GPU 2.23 GHz.";
    } else if (name == "TS5_PS5_GPU_GIB") {
        const int g = atoi(value.c_str());
        args = {"memory", value == "max" ? "max" : std::to_string(g >= 8 && g <= 14 ? g : 0)};
        done = "GPU memory set for the next start. Restart the PS5 and run the loader again.";
    } else if (name == "TS5_PS5_CUS") {
        const int n = atoi(value.c_str());
        args = {"cu", std::to_string(n == 38 || n == 40 ? n : 36)};
        done = "Compute units set for the next start. Restart the PS5 and run the loader again, then test them.";
    } else {
        return;
    }
    // Stepping through the choices saves each one: only the last, once the
    // choosing stops for a moment, goes to the boot settings.
    uint64_t gen;
    {
        std::lock_guard<std::mutex> lk(mu_);
        gen = ++gens_[name];
    }
    std::thread([this, name, gen, args, done] {
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (gens_[name] != gen) return;
        }
        std::string out;
        const bool ok = run(args, out);
        printf("ts5-title: hardware: %s %s: %s%s\n", args[0].c_str(), args[1].c_str(), ok ? "done" : "failed: ",
               ok ? "" : out.c_str());
        if (notify_) notify_(ok ? "TokenStation5: " + done : "TokenStation5: that PS5 setting didn't change. Settings shows why.");
        refresh();
    }).detach();
}

bool Hardware::testing() {
    std::lock_guard<std::mutex> lk(mu_);
    return info_.test == ui::HwInfo::Running;
}

void Hardware::test(int seconds, std::function<void()> done) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (info_.test == ui::HwInfo::Running) return;
        info_.test = ui::HwInfo::Running;
        info_.test_lines.clear();
    }
    std::thread([this, seconds, done] {
        std::string out;
        const bool ok = run({"cutest", std::to_string(seconds)}, out);
        printf("ts5-title: hardware: compute unit test: %s\n%s", ok ? "passed" : "failed", out.c_str());
        // The summary: the verdict and the lines about errors and results.
        std::vector<std::string> lines;
        size_t at = 0;
        while (at < out.size()) {
            const size_t nl = out.find('\n', at);
            const std::string line = out.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
            at = nl == std::string::npos ? out.size() : nl + 1;
            if (line.find("verdict") != std::string::npos || line.find("errors") != std::string::npos ||
                line.find("testing with") != std::string::npos || line.find("mismatch") != std::string::npos ||
                line.find("tok/s") != std::string::npos || line.find("|") == 0)
                lines.push_back(line);
        }
        if (lines.empty() && !out.empty()) lines.push_back(out.substr(0, out.find('\n')));
        if (lines.size() > 10) lines.erase(lines.begin(), lines.end() - 10);
        {
            std::lock_guard<std::mutex> lk(mu_);
            info_.test = ok ? ui::HwInfo::Passed : ui::HwInfo::Failed;
            info_.test_lines = lines;
        }
        if (notify_)
            notify_(ok ? "TokenStation5: the compute unit test passed."
                       : "TokenStation5: the compute unit test failed. Settings shows what went wrong.");
        if (done) done();
    }).detach();
}

}  // namespace linuxp
