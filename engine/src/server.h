#pragma once
#include <functional>
#include <string>

#include "engine.h"

namespace ts5 {

// Serves the OpenAI-compatible API (chat completions, completions, models,
// health) until the process ends. One generation runs at a time. Returns only
// when it cannot listen; on_listening runs once the socket is up.
int serve(Engine &e, const std::string &model_name, const std::string &host, int port,
          const std::function<void(int port)> &on_listening = nullptr);

}  // namespace ts5
