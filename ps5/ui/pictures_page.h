// The Pictures page: a picture from a prompt, made on the PS5's GPU with a
// picture model from the Models page (images.h). On the left, what to make:
// the prompt (cross types it on the console's keyboard), the model and the
// size (left and right change them); cross on either makes the picture. On
// the right, the last picture, as big as it fits. Triangle makes another of
// the same prompt with a new seed, square saves the one on show in the
// title's pictures folder (FTP gets them off the PS5). The chat model lets
// go of the GPU while a picture is made and comes back after (runner.h).
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
};

// What a press asks for.
struct PicturesPress {
    enum Kind { None, Type, Make, Again, Save } kind = None;
    std::string model;   // Make, Again: the model's key
    std::string prompt;  // Type: the prompt so far; Make, Again: the prompt
    int width = 0, height = 0;
    int64_t seed = -1;  // Again: -1 (a new one)
};

class PicturesPage {
public:
    explicit PicturesPage(const Fonts &fonts);
    PicturesPress press(uint32_t buttons, const PicturesInfo &info);
    // The prompt typed on the keyboard.
    void set_prompt(const std::string &p) { prompt_ = p; }
    // The choices on the left, then the picture on the right.
    void draw(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h);
    void draw_picture(Canvas &c, const PicturesInfo &info, int x, int y, int w, int h);
    // The footer's words for cross.
    std::string cross_hint(const PicturesInfo &info) const;

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
    int row_ = 0;            // 0 the prompt, 1 the model, 2 the size
    // The picture on show, scaled to fit, made again when it or the space changes.
    std::shared_ptr<const Picture> scaled_of_;
    int scaled_w_ = 0, scaled_h_ = 0;
    std::vector<uint32_t> scaled_;
};

}  // namespace ui
