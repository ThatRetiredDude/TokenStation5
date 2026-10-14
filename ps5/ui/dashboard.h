// The TV screen: the models on the console and a few suggested ones, the
// selected model's details and results, what the title is doing, and the API's
// activity. Driven with the controller: up/down choose a model, cross serves
// it, square runs the GPU check, triangle the speed test, circle stops
// serving, OPTIONS closes the title.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "canvas.h"
#include "library.h"
#include "runner.h"
#include "server.h"

namespace ui {

// Everything one frame shows, gathered under the title's locks.
struct View {
    std::string version, url, folder;  // folder: the models folder as the PC sees it over FTP
    uint64_t gpu_memory = 0;
    int n_ctx = 0;  // the context a Serve loads with (0: what fits)
    std::vector<app::ModelInfo> models;
    std::map<std::string, app::Results> results;  // by file name
    app::Status status;
    std::vector<app::Event> events;
    ts5::Server::Activity api;
    double now = 0;    // seconds, steady clock (app::now_s)
    int64_t wall = 0;  // time(nullptr)
};

struct Action {
    enum Kind { None, Serve, Check, Bench, Stop, Close } kind = None;
    std::string path;
};

class Dashboard {
public:
    Dashboard(const Fonts &fonts, const uint8_t *logo, size_t logo_size);
    // Applies a frame's button presses; at most one action comes of them.
    Action press(uint32_t buttons, const View &v);
    void draw(Canvas &c, const View &v);
    // Something on screen moves (a spinner), so frames keep coming.
    bool animating(const View &v) const;

private:
    struct Item;
    std::vector<Item> items(const View &v) const;
    int selected(const std::vector<Item> &items) const;
    void draw_background(Canvas &c);
    void draw_header(Canvas &c, const View &v);
    void draw_list(Canvas &c, const View &v, const std::vector<Item> &items, int sel);
    void draw_details(Canvas &c, const View &v, const Item &item);
    void draw_activity(Canvas &c, const View &v);
    void draw_footer(Canvas &c);
    void draw_confirm(Canvas &c);

    const Font &title_, &h2_, &body_, &bodyb_, &small_, &smallb_, &mono_;
    int logo_w_ = 0, logo_h_ = 0;
    const uint32_t *logo_ = nullptr;
    std::vector<uint32_t> logo_px_;
    std::vector<uint32_t> background_;
    std::string selected_key_;  // the selected item's path or suggestion key
    int scroll_ = 0;
    bool confirm_close_ = false;
};

}  // namespace ui
