// ts5-tui's sensors (sensors.h).
#include "sensors.h"

#include <dirent.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tui {

std::string read_file(const std::string &path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

namespace {

bool read_num(const std::string &path, double &v) {
    const std::string s = read_file(path);
    if (s.empty()) return false;
    char *end = nullptr;
    v = strtod(s.c_str(), &end);
    return end != s.c_str();
}

std::vector<std::string> list_dir(const std::string &dir, const char *prefix) {
    std::vector<std::string> out;
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d))
            if (!strncmp(e->d_name, prefix, strlen(prefix))) out.push_back(dir + "/" + e->d_name);
        closedir(d);
    }
    return out;
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

struct Paths {
    std::string drm;  // /sys/class/drm/cardN/device with mem_info_gtt_total
    std::string die, board, edge, fan_pwm, fan_rpm, fan_enable, fan_top;
    std::chrono::steady_clock::time_point when{};
    bool found = false;
};

// TS5_TUI_SYSROOT: a folder that stands in for / (the tests' fake PS5).
std::string root() {
    const char *r = getenv("TS5_TUI_SYSROOT");
    return r ? r : "";
}

Paths find_paths() {
    Paths p;
    for (auto &card : list_dir(root() + "/sys/class/drm", "card")) {
        if (card.find('-', card.rfind('/')) != std::string::npos) continue;  // card0-HDMI-A-1 and such
        if (!read_file(card + "/device/mem_info_gtt_total").empty()) {
            p.drm = card + "/device";
            break;
        }
    }
    for (auto &hw : list_dir(root() + "/sys/class/hwmon", "hwmon")) {
        const std::string name = trim(read_file(hw + "/name"));
        if (name == "k10temp" || name == "zenpower") {
            // Tctl is temp1 on k10temp.
            for (int i = 1; i <= 8; i++) {
                const std::string lbl = trim(read_file(hw + "/temp" + std::to_string(i) + "_label"));
                if (lbl == "Tctl" || lbl == "Tdie" || (i == 1 && lbl.empty())) {
                    p.die = hw + "/temp" + std::to_string(i) + "_input";
                    if (lbl == "Tctl") break;
                }
            }
        } else if (name == "ps5_fan") {
            for (int i = 1; i <= 8; i++) {
                const std::string lbl = trim(read_file(hw + "/temp" + std::to_string(i) + "_label"));
                const std::string in = hw + "/temp" + std::to_string(i) + "_input";
                if (lbl == "MainSoC" || (p.board.empty() && !read_file(in).empty())) p.board = in;
            }
            p.fan_pwm = hw + "/pwm1";
            p.fan_enable = hw + "/pwm1_enable";
            p.fan_top = hw + "/pwm1_auto_point4_pwm";
            if (!read_file(hw + "/fan1_input").empty()) p.fan_rpm = hw + "/fan1_input";
        } else if (name == "amdgpu") {
            if (!read_file(hw + "/temp1_input").empty()) p.edge = hw + "/temp1_input";
            if (p.fan_pwm.empty() && !read_file(hw + "/pwm1").empty()) p.fan_pwm = hw + "/pwm1";
            if (p.fan_rpm.empty() && !read_file(hw + "/fan1_input").empty()) p.fan_rpm = hw + "/fan1_input";
        }
    }
    p.when = std::chrono::steady_clock::now();
    p.found = true;
    return p;
}

}  // namespace

Sensors read_sensors() {
    static Paths p;
    const auto now = std::chrono::steady_clock::now();
    if (!p.found || now - p.when > std::chrono::seconds(60)) p = find_paths();
    Sensors s;
    double v;
    if (!p.drm.empty()) {
        s.any = true;
        read_num(p.drm + "/mem_info_gtt_used", s.gtt_used);
        read_num(p.drm + "/mem_info_gtt_total", s.gtt_total);
        read_num(p.drm + "/mem_info_vram_used", s.vram_used);
        read_num(p.drm + "/mem_info_vram_total", s.vram_total);
        if (read_num(p.drm + "/gpu_busy_percent", v)) s.gpu_busy = {true, v};
        // pp_dpm_sclk: "1: 2230Mhz *" marks the clock now.
        std::istringstream in(read_file(p.drm + "/pp_dpm_sclk"));
        std::string line;
        while (std::getline(in, line))
            if (line.find('*') != std::string::npos) {
                const size_t colon = line.find(':');
                if (colon != std::string::npos) s.gpu_mhz = {true, atof(line.c_str() + colon + 1)};
            }
        const std::string prod = trim(read_file(p.drm + "/product_name"));
        s.gpu_name = prod;
    }
    std::istringstream mi(read_file(root() + "/proc/meminfo"));
    std::string k;
    double kb;
    std::string unit;
    while (mi >> k >> kb) {
        std::getline(mi, unit);
        if (k == "MemTotal:") s.ram_total = kb * 1024;
        if (k == "MemAvailable:") s.ram_avail = kb * 1024;
    }
    if (s.ram_total > 0) s.any = true;
    if (!p.die.empty() && read_num(p.die, v)) s.die = {true, v / 1000};
    if (!p.board.empty() && read_num(p.board, v)) s.board = {true, v / 1000};
    if (!p.edge.empty() && read_num(p.edge, v)) s.edge = {true, v / 1000};
    if (!p.fan_pwm.empty() && read_num(p.fan_pwm, v)) s.fan_pct = {true, v * 100 / 255};
    if (!p.fan_rpm.empty() && read_num(p.fan_rpm, v)) s.fan_rpm = {true, v};
    if (!p.fan_enable.empty() && read_num(p.fan_enable, v)) {
        // As /usr/lib/tokenstation/hw status names them.
        if (v == 0) {
            s.fan_mode = "max";
            s.fan_pct = {true, 100};
        } else if (v == 1) {
            s.fan_mode = "manual";
        } else {
            double top = 0;
            s.fan_mode = read_num(p.fan_top, top) && top == 255 ? "cool" : "auto";
        }
    }
    if (s.die.have || s.board.have) s.any = true;
    return s;
}

}  // namespace tui
