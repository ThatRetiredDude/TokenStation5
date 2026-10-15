// The web_search tool that ts5 runs itself, with Tavily (https://tavily.com):
// the model asks for a search, ts5 calls Tavily over HTTPS and gives the
// results back to the model, so every client gets search, even ones without
// tool support.
//
// Settings (environment, or ts5.env on the console):
//   TAVILY_API_KEY       the key; without it there is no web_search tool
//   TS5_SEARCH_RESULTS   results per search (default 5)
//   TS5_TAVILY_URL       another endpoint with Tavily's API (default https://api.tavily.com)
#pragma once
#include <string>

#include "json.h"

namespace ts5 {

// TAVILY_API_KEY, or "".
std::string tavily_key();

// The tool as offered to the model. today: "14 Oct 2026", so the model can
// search for what is recent.
Json web_search_tool(const std::string &today);

// Runs a web_search call (its arguments, as JSON text). Returns what the
// model gets back: the results, or why there are none. log: one line for the log.
// The text for the model; *failed (when given) says the search itself did
// not work (network, a refused key, a used-up plan), not that it found nothing.
std::string web_search(const std::string &arguments, std::string &log, bool *failed = nullptr);

// Why a search failed, in the plain words web_search gave the model ("the
// Tavily key was refused"); the log line has the raw reason.
std::string search_failure(const std::string &result);

}  // namespace ts5
