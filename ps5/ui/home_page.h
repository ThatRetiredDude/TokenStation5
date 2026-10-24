// The Home page, the title's start page: how the API has been used, like an
// AI provider's usage page. Tiles for what runs now and the range's tokens,
// requests and writing speed; tokens by model as stacked bars; requests by
// app; writing speed by model; and a live line of the last two minutes.
// Left and right change the range (24 hours, 7 days, 30 days). Its numbers
// come from UsageStore (usage_store.h), the live line from the API's
// counters sampled once a second.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "canvas.h"
#include "runner.h"
#include "server.h"
#include "usage_store.h"

namespace ui {

struct HomeInfo {
    std::shared_ptr<const app::UsageView> usage;  // null until the store has one
    std::vector<float> live;       // tokens written a second, a point a second, oldest first
    std::vector<uint8_t> reading;  // 1 for a second the API read a prompt
    app::Status status;
    std::string running;  // the serving model's name
    std::string job;      // the name of the model a job is on
    ts5::Server::Activity api;
    std::string own_ip;   // the console's address: its requests are "Chat on this PS5"
    std::string url;      // the address apps use
    bool on_network = false;
    double now = 0;
    std::string download;  // "Downloading Qwen3 8B · 38%" while one runs
};

class HomePage {
public:
    explicit HomePage(const Fonts &fonts);
    app::UsageView::Range range() const { return range_; }
    // Left and right change the range.
    void press(uint32_t buttons);
    void draw(Canvas &c, const HomeInfo &info);

    // Where the panels are, for the page's background.
    static void panels(std::vector<std::array<int, 4>> &out);

private:
    void draw_tiles(Canvas &c, const HomeInfo &info);
    void draw_tokens(Canvas &c, const HomeInfo &info);
    void draw_apps(Canvas &c, const HomeInfo &info);
    void draw_speed(Canvas &c, const HomeInfo &info);
    void draw_live(Canvas &c, const HomeInfo &info);

    const Font &h2_, &body_, &bodyb_, &small_, &smallb_, &mono_;
    app::UsageView::Range range_ = app::UsageView::Day;
};

}  // namespace ui
