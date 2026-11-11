// The web_search tool that ts5 runs itself: the model asks for a search, ts5
// runs it over HTTPS and gives the results back to the model, so every client
// gets search, even ones without tool support. With a Tavily key
// (https://tavily.com) it is a search of the web. Without one it is a search
// of Wikipedia (its own API, no key), for a request that asks for a search
// ("web_search": true or "auto", as ts5-tui does) or with
// TS5_WEB_SEARCH=always: an encyclopedia, with no news, prices or weather,
// and the model is told so.
//
// Settings (environment, or ts5.env on the console):
//   TAVILY_API_KEY       the key; without it the search is Wikipedia's
//   TS5_SEARCH_RESULTS   results per search (default 5)
//   TS5_TAVILY_URL       another endpoint with Tavily's API (default https://api.tavily.com)
//   TS5_WEB_SEARCH       "always" offers the tool on every chat; by default
//                        only on chats that look like they need the web
//   TS5_WIKIPEDIA        the Wikipedia searched without a key: its language
//                        ("en" by default, "sk", "de" ...), or "off" for none
//   TS5_WIKIPEDIA_URL    another endpoint with MediaWiki's API (default
//                        https://LANGUAGE.wikipedia.org)
#pragma once
#include <string>

#include "json.h"

namespace ts5 {

// TAVILY_API_KEY, or "".
std::string tavily_key();

// Where a search goes: "tavily" with a key, else "wikipedia", or "none"
// with TS5_WIKIPEDIA=off.
std::string search_source();

// Whether a chat's last user message looks like it needs the web: it asks to
// look something up or names a website, or it asks about what changes (news,
// weather, prices, scores, elections, "today", "latest", a year after 2023).
// ts5 offers web_search only then (or with "web_search": true, or
// TS5_WEB_SEARCH=always): with the tool in the prompt, Llama 3's template
// tells the model to answer with a call, and it searched on plain questions.
// Without it, a general question costs no credit and no prompt tokens.
// wikipedia: also when it names Wikipedia or asks to search.
bool needs_web(const std::string &user_text, bool wikipedia = false);

// TS5_WEB_SEARCH=always.
bool web_search_always();

// The tool as offered to the model. today: "14 Oct 2026", so the model can
// search for what is recent. wikipedia: the words for a search of Wikipedia
// alone (search_source() by default).
Json web_search_tool(const std::string &today);
Json web_search_tool(const std::string &today, bool wikipedia);

// Runs a web_search call (its arguments, as JSON text). Returns what the
// model gets back: the results, or why there are none. log: one line for the log.
// The text for the model; *failed (when given) says the search itself did
// not work (network, a refused key, a used-up plan), not that it found nothing.
std::string web_search(const std::string &arguments, std::string &log, bool *failed = nullptr);

// A web page as text, for a model to read (`ts5 page URL`, which ts5-tui's
// web page tool runs): url over http or https, 2 MiB of it at most; HTML
// without its scripts, styles and tags, the entities read and blank runs
// closed up; other text as it is. Cut to max_chars, saying how much is left
// out. log: one line for the log; *failed: the page could not be read.
std::string web_page(const std::string &url, size_t max_chars, std::string &log, bool *failed = nullptr);
// The text of an HTML page (web_page's reader).
std::string html_text(const std::string &html);

// Why a search failed, in the plain words web_search gave the model ("the
// Tavily key was refused"); the log line has the raw reason.
std::string search_failure(const std::string &result);

}  // namespace ts5
