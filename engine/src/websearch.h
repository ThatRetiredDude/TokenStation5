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
//   TS5_ENCYCLOPEDIA     the encyclopedia searched without a key: "wikipedia"
//                        (the default), "grokipedia" (grokipedia.com, xAI's,
//                        through the site's own search; when that can't be
//                        searched, Wikipedia is) or "both". A request picks
//                        one with "web_search_source". Not on the console.
//   TS5_GROKIPEDIA_URL   another address with Grokipedia's search (default
//                        https://grokipedia.com)
#pragma once
#include <string>
#include <vector>

#include "json.h"

namespace ts5 {

// TAVILY_API_KEY, or "".
std::string tavily_key();

// Where a search goes: "tavily" with a key, else an encyclopedia:
// "wikipedia", "grokipedia" or "both" (asked: a request's
// "web_search_source", else TS5_ENCYCLOPEDIA, else Wikipedia), or "none"
// (TS5_WIKIPEDIA=off and no Grokipedia).
std::string search_source(const std::string &asked = "");
// "wikipedia", "grokipedia" or "both": a search of encyclopedias alone.
bool is_encyclopedia(const std::string &source);

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
// search for what is recent. source: where the search goes (search_source()
// by default); an encyclopedia's words say that it is one.
Json web_search_tool(const std::string &today);
Json web_search_tool(const std::string &today, const std::string &source);

// Runs a web_search call (its arguments, as JSON text). Returns what the
// model gets back: the results, or why there are none. log: one line for the log.
// The text for the model; *failed (when given) says the search itself did
// not work (network, a refused key, a used-up plan), not that it found nothing.
// asked: the request's "web_search_source" ("": this machine's).
std::string web_search(const std::string &arguments, std::string &log, bool *failed = nullptr, const std::string &asked = "");

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
// Which browser ts5 says it is to websites, from TS5_BROWSER_AGENT: "mac"
// (the default: Chrome as it is now on a MacBook, kChrome below), "linux"
// (Chrome on Linux) or "own" (the page reader under ts5's own name, a
// browser in its own words).
std::string browser_look();
// Chrome now, on a Mac: its version (a release of November 2026; raise it
// with Chrome's releases) and the macOS it runs on.
constexpr int kChromeMajor = 158;
constexpr const char *kChromeFull = "158.0.8127.18";
constexpr const char *kMacOS = "27.2.1";
// Chrome's User-Agent header for that look and a Chrome major version.
std::string chrome_agent(const std::string &look, int major);
// The brands Chrome names in its Sec-CH-UA header and to a page's scripts:
// "Chromium", "Google Chrome" when google, and a made-up third that
// changes with the version, in an order that changes with it too (as
// Chromium makes them, components/embedder_support/user_agent_utils.cc).
// version: the major version, or the full one for the full list.
std::vector<std::pair<std::string, std::string>> chrome_brands(int major, const std::string &version, bool google);
// The headers a web page is asked for with, by the page reader and the
// browser's text view: what that browser sends when a person opens a page
// by its address (a site is shown the same whoever reads it for them), or
// under ts5's own name with TS5_BROWSER_AGENT=own. secure: an https page
// (or one on this machine), the only ones Chrome tells its version and
// system in other headers too.
std::vector<std::string> page_headers(bool secure = true);
// An https address, or one on this machine: where Chrome sends those headers.
bool secure_url(const std::string &url);

std::string html_title(const std::string &html);

// Why a search failed, in the plain words web_search gave the model ("the
// Tavily key was refused"); the log line has the raw reason.
std::string search_failure(const std::string &result);

}  // namespace ts5
