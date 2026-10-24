// Searching Hugging Face for models to download on the console, from the
// Models page. A search asks the Hub for GGUF repos matching the words (most
// downloaded first), keeps the ones whose architecture ts5 runs, lists each
// one's files, reads the header of one file per repo (its first 2 MiB, with
// a Range request) for the memory estimate, and offers the files of a type
// ts5's GPU code runs that fit in the PS5's memory with at least 8,192
// tokens of context (or the model's own context, if that's less). Each one
// found becomes a catalog entry (downloads.h), pinned to the repo's commit
// with the Hub's size and SHA-256, so it downloads, stops, goes on and is
// checked like the built-in ones.
#pragma once
#include <atomic>
#include <memory>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace app {

struct HfState {
    enum Phase { Idle, Searching, Done, Failed } phase = Idle;
    std::string query;
    int checked = 0, to_check = 0;  // repos looked at so far, of how many
    std::vector<size_t> results;    // catalog indexes (downloads.h), as found
    int too_big = 0, cant_run = 0;  // files and repos left out, and why
    std::string problem;            // Failed: in plain words
};

class HfSearch {
public:
    // Searches for query in the background, for a GPU with gpu_memory bytes;
    // a search still running is dropped (its thread ends on its own).
    void start(const std::string &query, uint64_t gpu_memory);
    HfState state();
    // Bumped on every change worth a redraw.
    uint64_t version() const { return s_->version.load(); }

    // Shared with the search threads, which outlive nothing they need.
    struct Shared {
        std::mutex mu;
        HfState state;
        std::atomic<uint64_t> gen{0}, version{0};
    };

private:
    std::shared_ptr<Shared> s_ = std::make_shared<Shared>();
};

}  // namespace app
