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
#include <vector>

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
// What a page offers to act on, as a page view numbers it in its text.
struct PageElement {
    std::string kind;   // "link", "input" (a text box), "button", "check" (a tick box or one of a choice)
    std::string label;  // the link's or button's words, a box's hint
    std::string href;   // a link's address as the page has it
    std::string name, value, type;  // as a form sends it
    std::string mark;   // its data-ts5 attribute: how a browser finds it again
    bool checked = false;
    int form = -1;      // the form it is in (PageElements::forms), or -1
};
struct PageForm {
    std::string action, method;  // as the page has them ("" and "get" by default)
    struct Field {
        std::string name, value;
        int item;  // the element it is (its index in items), -1 for a hidden field or a list's choice
    };
    std::vector<Field> fields;  // in the page's order
};
struct PageElements {
    std::vector<PageElement> items;  // items[n - 1] is the text's [n]
    std::vector<PageForm> forms;
};
// An HTML page as text for a model that acts on it (ts5's browser): as
// html_text, with every link, text box and button numbered in place ("the
// Volga[2]", "[4: text box "Search"]", "[5: button "Go"]") and listed in
// els; the lists of links stay, what the page hides is left out.
std::string html_view(const std::string &html, PageElements &els);
// The page's <title>, on one line.
// The headers a web page is asked for with, by the page reader and the
// browser's text view: what a person's browser sends (a site is shown the
// same whoever reads it for them), unless TS5_BROWSER_AGENT=own, when the
// request goes out under ts5's own name.
std::vector<std::string> page_headers();

std::string html_title(const std::string &html);

// Why a search failed, in the plain words web_search gave the model ("the
// Tavily key was refused"); the log line has the raw reason.
std::string search_failure(const std::string &result);

}  // namespace ts5
