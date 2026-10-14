// ts5's command line as a function, for the Linux binary and the PS5 title.
#pragma once
#include <functional>
#include <string>

namespace ts5 {

// Called once `serve` is listening, with the model's file name and the port.
using ServingHook = std::function<void(const std::string &model, int port)>;

// Runs one command line (argv[0] is the program name) and returns its exit
// status. `serve` returns only when it cannot listen. Never calls exit().
int cli_main(int argc, char **argv, const ServingHook &on_serving = nullptr);

}  // namespace ts5
