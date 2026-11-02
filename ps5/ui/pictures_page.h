// The Pictures page: a picture from a prompt, made on the PS5's GPU with a
// picture model from the Models page (images.h). On the left, what to make:
// the prompt (cross types it on the console's keyboard, then the model row
// is chosen), the model and the size (left and right change them); cross on
// either makes the picture. On the right, the last picture, made here or
// asked for by an app, as big as it fits. Triangle makes another of the same
// prompt with a new seed. Every picture is saved in the title's pictures
// folder by itself (FTP gets them off the PS5). GALLERY, at the bottom on
// the left, shows that folder's pictures on the right: cross opens one,
// left and right go through them, square twice deletes one, circle goes
// back. The chat model lets go of the GPU while a picture is made and comes
// back after (runner.h).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "canvas.h"

namespace ui {

// A picture made, as the page shows it.
struct Picture {
    int w = 0, h = 0;
    std::vector<uint32_t> px;  // 0xRRGGBB, row by row
    std::string prompt, model;  // model: its name on screen
    int64_t seed = 0;
    double secs = 0;  // making it, without loading the model
};

// What the page shows besides its own choices.
struct PicturesInfo {
    struct Model {
        std::string key, name;  // key: its name in the API (images.h)
        int size = 512;         // its pictures' width and height, unless chosen
        bool fits = true;       // in the GPU's memory
    };
    std::vector<Model> models;  // the picture models on the PS5
    bool making = false;        // a picture of the page's is being made
    bool gpu_busy = false;      // the GPU does something else (a model loads, a check)
    std::string step;           // making: "step 3 of 20"
    double started = 0, now = 0;  // seconds, steady clock
    std::string problem;        // why the last one failed, plain words
    std::shared_ptr<const Picture> picture;  // the last one made
    std::string saved;          // the file it was saved to ("": not saved)
    // The pictures folder, newest first, with each one made small to show
    // (null until it's read).
    struct Saved {
        std::string name;
        std::shared_ptr<const Picture> thumb;
    };
    std::vector<Saved> gallery;
    std::string viewing;                     // the gallery's picture asked to open
    std::shared_ptr<const Picture> opened;  // it, once read (null until then, or unreadable)
    bool open_failed = false;
};

// The picture made to fit w by h (each pixel the average of the ones it
// covers, or the nearest one when bigger).
void scale_picture(const Picture &p, int w, int h, std::vector<uint32_t> &out);
// Thumbnails' largest side.
constexpr int kThumb = 232;

// What a press asks for.
struct PicturesPress {
    enum Kind { None, Type, Make, Again, Open, Delete } kind = None;
    std::string model;   // Make, Again: the model's key; Open, Delete: the gallery picture's file name
    std::string prompt;  // Type: the prompt so far; Make, Again: the prompt
    int width = 0, height = 0;
    int64_t seed = -1;  // Again: -1 (a new one)
};

class PicturesPage {
public:
    explicit PicturesPage(const Fonts &fonts);
    PicturesPress press(uint32_t buttons, const PicturesInfo &info);
    // The prompt typed on the keyboard. With one, the model row is chosen,
    // so the next cross makes the picture instead of typing again.
    void set_prompt(const std::string &p) {
        prompt_ = p;
        if (!p.empty()) row_ = 1;
    }
    // The choices on the left, then the picture on the right.
    void draw(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h);
    void draw_picture(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h);
    // The footer's words for cross, and for square ("": nothing).
    std::string cross_hint(const PicturesInfo &info) const;
    std::string square_hint(const PicturesInfo &info) const;
    // The gallery is on the right: circle goes back in it, not Home.
    bool in_gallery() const { return mode_ != Making; }

private:
    struct Size {
        int w, h;
    };
    std::vector<Size> sizes(const PicturesInfo &info) const;
    const PicturesInfo::Model *model(const PicturesInfo &info) const;
    bool can_make(const PicturesInfo &info) const;

    const Font &h2_, &body_, &bodyb_, &small_, &smallb_;
    std::string prompt_;
    std::string model_key_;  // the chosen model ("": the first)
    int size_ = 0;           // into sizes()
    int row_ = 0;            // 0 the prompt, 1 the model, 2 the size, 3 GALLERY
    enum Mode { Making, Gallery, Viewing } mode_ = Making;
    int pick_ = 0;            // the gallery's chosen picture
    int cols_ = 1, first_row_ = 0;  // the gallery's grid as last drawn, and its top row
    std::string deleting_;  // square was pressed once on this picture
    int picked(const PicturesInfo &info) const;
    void draw_gallery(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h);
    void draw_opened(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h);
    std::vector<uint32_t> opened_px_;
    std::shared_ptr<const Picture> opened_of_;
    int opened_w_ = 0, opened_h_ = 0;
    // The picture on show, scaled to fit, made again when it or the space changes.
    std::shared_ptr<const Picture> scaled_of_;
    int scaled_w_ = 0, scaled_h_ = 0;
    std::vector<uint32_t> scaled_;
};

}  // namespace ui
