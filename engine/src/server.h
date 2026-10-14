#pragma once
#include <string>

#include "engine.h"

namespace ts5 {

// Serves the OpenAI-compatible API (chat completions, completions, models,
// health) until the process ends. One generation runs at a time.
int serve(Engine &e, const std::string &model_name, const std::string &host, int port);

}  // namespace ts5
