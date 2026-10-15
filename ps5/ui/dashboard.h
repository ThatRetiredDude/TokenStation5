// The TV screen, three pages under the header; L1 goes to the next.
// Models: the models on the console and a few suggested ones, the selected
// model's details and results, what the title is doing, and the API's
// activity; up/down choose a model, cross starts it, square runs the GPU
// check, triangle the speed test, circle stops it. Every page says things in
// plain words (plain.h): no paths, commands, settings names or engine errors,
// which stay in the log. Chat (chat_page.h):
// a conversation with the serving model; cross types a message on the
// console's keyboard, hold R1 to talk, up/down scroll, triangle stops the
// reply, square starts a new chat, circle goes back to Models. Settings
// (settings_page.h): the title's settings, kept in ts5.env; circle goes back
// to Models. OPTIONS closes the title.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "canvas.h"
#include "chat.h"
#include "chat_page.h"
#include "settings_page.h"
#include "voice.h"
#include "library.h"
#include "runner.h"
#include "server.h"

namespace ui {

// Everything one frame shows, gathered under the title's locks.
struct View {
    std::string version, url;
    bool on_network = false;  // the console has a LAN address, so url is one apps can use
    uint64_t gpu_memory = 0;
    int n_ctx = 0;  // the context a Serve loads with (0: what fits)
    std::vector<app::ModelInfo> models;
    std::map<std::string, app::Results> results;  // by file name
    app::Status status;
    std::vector<app::Event> events;
    ts5::Server::Activity api;
    app::ChatState chat;
    app::VoiceState voice;
    bool speaking = false;  // a reply is being read aloud (speak.h)
    SettingsInfo settings;  // its speech models and now come from the fields above
    double now = 0;    // seconds, steady clock (app::now_s)
    int64_t wall = 0;  // time(nullptr)
};

struct Action {
    enum Kind { None, Serve, Check, Bench, Stop, Close, ChatStop, ChatClear, ChatType, Set, TypeSetting } kind = None;
    std::string path;
    std::string name, value;  // Set: NAME=VALUE in ts5.env ("": the default); TypeSetting: NAME
    std::string label;        // TypeSetting: the setting's name on the page
};

class Dashboard {
public:
    enum Page { Models, Chat, Settings };
    Dashboard(const Fonts &fonts, const uint8_t *logo, size_t logo_size);
    Page page() const { return page_; }
    void show(Page p);
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
    void draw_chat_footer(Canvas &c, const View &v);
    void draw_settings_footer(Canvas &c);
    SettingsInfo settings_info(const View &v) const;
    void draw_tabs(Canvas &c);
    void draw_confirm(Canvas &c);

    const Font &title_, &h2_, &body_, &bodyb_, &small_, &smallb_;
    int logo_w_ = 0, logo_h_ = 0;
    const uint32_t *logo_ = nullptr;
    std::vector<uint32_t> logo_px_;
    std::vector<uint32_t> background_, chat_background_, settings_background_;
    Page page_ = Models;
    ChatPage chat_;
    SettingsPage settings_;
    std::string selected_key_;  // the selected item's path or suggestion key
    int scroll_ = 0;
    bool confirm_close_ = false;
};

}  // namespace ui
