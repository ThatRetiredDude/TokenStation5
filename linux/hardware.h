// The PS5's hardware under Linux, for the Settings page's PS5 HARDWARE
// section: temperatures (hwmon, read as the app's own user), and the fan,
// clock boost, GPU memory and compute units, read and changed through
// /usr/lib/tokenstation/hw as root (sudo -n: /etc/sudoers.d/tokenstation
// lets the tokenstation group run it, and only it, without a password).
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "settings_page.h"

namespace linuxp {

class Hardware {
public:
    using Notify = std::function<void(const std::string &)>;
    // Finds the PS5's controls; with them, puts the fan and boost as the
    // settings say (TS5_PS5_FAN, TS5_PS5_BOOST) and reads the rest.
    void start(Notify notify);
    // The page's view; temperatures read at most every 2 s.
    ui::HwInfo info();
    // A PS5 HARDWARE setting the page just saved: done in the background,
    // and a notification says how it went.
    void apply(const std::string &name, const std::string &value);
    // The compute unit test (tokenstation-cu test); done() runs after it.
    void test(int seconds, std::function<void()> done);
    bool testing();

private:
    void refresh();  // hw status, in the background
    bool run(const std::vector<std::string> &args, std::string &out);
    std::mutex mu_;
    Notify notify_;
    ui::HwInfo info_;
    double temps_at_ = -1;
    std::map<std::string, uint64_t> gens_;  // per setting: its latest change
};

}  // namespace linuxp
