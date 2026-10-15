// The Logs page: what the title wrote in ts5.log (or ts5.prev.log, the run
// before), the one page where raw text shows: file names, settings names,
// the engine's, the GPU's and the network's own words. A key never does: a
// NAME_KEY= or NAME_TOKEN= value and a tvly- key are hidden before a line is
// drawn. The newest line is at the bottom and the page follows it unless
// scrolled up. Up and down scroll a line, left and right a page, square
// shows only the problems, triangle switches to the run before.
//
// The file is read only while the page shows: its last 256 KiB, again when
// its size changes, at most every 2 s. Leaving the page drops the text.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "canvas.h"

namespace ui {

class LogsPage {
public:
    explicit LogsPage(const Fonts &fonts);
    // This run's log and the run before's.
    void set_files(std::string log, std::string prev_log);
    // Shown: this run's, scrolled to the end; with problems, only them.
    void open(bool problems);
    void close();
    void press(uint32_t buttons);
    // Reads the file if it's time (now: seconds, steady clock), then draws.
    void draw(Canvas &c, int x, int y, int w, int h, const std::string &version, double now);
    bool problems() const { return problems_; }
    bool last_run() const { return prev_; }

    // A line as the page may show it: keys hidden.
    static std::string hide_keys(const std::string &line);
    // A line that says something went wrong (failed, an error, can't,
    // couldn't, refused).
    static bool problem(const std::string &line);

private:
    void load(double now);

    const Font &h2_, &small_, &smallb_, &mono_;
    std::string log_, prev_log_;
    std::vector<std::string> lines_;  // the file's whole lines, keys hidden
    long long size_ = -1;             // of the file read, -1: none yet
    bool missing_ = false;
    double read_at_ = -1e9;
    bool problems_ = false, prev_ = false;
    int from_end_ = 0;  // rows scrolled up from the newest
    int page_rows_ = 20;
};

}  // namespace ui
