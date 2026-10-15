#include "websearch.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "net.h"

namespace ts5 {

namespace {

std::string env_or(const char *name, const char *def) {
    const char *v = getenv(name);
    return v && *v ? v : def;
}

// At most n bytes, cut at a word and never inside a UTF-8 character.
std::string clip(const std::string &s, size_t n) {
    if (s.size() <= n) return s;
    size_t cut = n;
    while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) cut--;
    size_t space = s.rfind(' ', cut);
    if (space != std::string::npos && space > n / 2) cut = space;
    return s.substr(0, cut) + " ...";
}

}  // namespace

std::string tavily_key() { return env_or("TAVILY_API_KEY", ""); }

bool web_search_always() { return env_or("TS5_WEB_SEARCH", "") == "always"; }

bool needs_web(const std::string &user_text) {
    // Lower-case words, letters and digits only; a website (a word with
    // ".com" and the like, or "http") counts by itself.
    std::string text;
    for (char c : user_text) text += (char)tolower((unsigned char)c);
    for (const char *site : {"http://", "https://", "www.", ".com", ".org", ".net", ".io", ".gov", ".edu"})
        if (text.find(site) != std::string::npos) return true;
    std::vector<std::string> words;
    std::string w;
    for (char c : text) {
        if (isalnum((unsigned char)c)) {
            w += c;
        } else if (!w.empty()) {
            words.push_back(w);
            w.clear();
        }
    }
    if (!w.empty()) words.push_back(w);
    // Words that on their own mean "now" or "look it up". Words that are as
    // often about code or general knowledge ("search", "update", "source",
    // "now", "web", "live", "current", "recent") only count in the phrases
    // below, or not at all.
    static const char *single[] = {
        "google",   "googling", "bing",      "online",    "internet", "website",  "websites",
        "today",    "tonight",  "tomorrow",  "yesterday", "latest",   "newest",   "upcoming",
        "news",     "headlines", "weather",  "forecast",  "price",    "prices",   "stock",
        "stocks",   "bitcoin",  "crypto",    "scores",    "standings", "election", "elections",
        "released", "trending", "flights",   "cite",      "citation", "citations"};
    static const char *pairs[][2] = {
        {"look", "up"},   {"looking", "up"}, {"look", "online"}, {"search", "online"}, {"search", "the"},
        {"web", "search"}, {"the", "web"},   {"right", "now"},   {"of", "now"},        {"these", "days"},
        {"who", "won"},   {"what", "happened"}, {"this", "week"}, {"this", "weekend"}, {"this", "month"},
        {"this", "year"}, {"last", "week"},  {"last", "night"},  {"last", "month"},    {"next", "week"},
        {"next", "month"}, {"release", "date"}, {"exchange", "rate"}, {"find", "out"}, {"check", "online"}};
    for (size_t i = 0; i < words.size(); i++) {
        const std::string &a = words[i];
        for (const char *s : single)
            if (a == s) return true;
        // A year after the models' training ("2025", "2026"...).
        if (a.size() == 4 && a >= "2024" && a <= "2099" &&
            std::all_of(a.begin(), a.end(), [](char c) { return isdigit((unsigned char)c); }))
            return true;
        if (i + 1 < words.size())
            for (auto &p : pairs)
                if (a == p[0] && words[i + 1] == p[1]) {
                    // "search the" only with the web or the internet after it.
                    if (a == "search" && words[i + 1] == "the")
                        if (i + 2 >= words.size() || (words[i + 2] != "web" && words[i + 2] != "internet")) continue;
                    return true;
                }
        if (a == "look" && i + 2 < words.size() && words[i + 1] == "it" && words[i + 2] == "up") return true;
    }
    return false;
}

Json web_search_tool(const std::string &today) {
    std::string tool =
        "{\"type\": \"function\", \"function\": {\"name\": \"web_search\", \"description\": \"Search the web. "
        "Returns the top results, each with a title, URL and text from the page. Use it only when the answer "
        "depends on something after your training data (news, prices, weather, schedules, new releases, "
        "a specific website), or when the user asks you to look something up. Do not search for general "
        "knowledge, explanations, definitions, writing or code: answer those yourself. When you do search, "
        "answer from the results and cite the URLs you used. Today is " +
        today +
        ".\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", "
        "\"description\": \"What to search for, as you would type it into a search engine.\"}}, "
        "\"required\": [\"query\"]}}}";
    Json j;
    Json::parse(tool, j);
    return j;
}

std::string web_search(const std::string &arguments, std::string &log, bool *failed) {
    Json args;
    std::string query;
    if (Json::parse(arguments, args)) query = args["query"].str();
    if (query.empty()) {
        log = "web_search: no query in " + arguments;
        return "web_search needs a query, such as {\"query\": \"what to search for\"}.";
    }
    const std::string key = tavily_key();
    int n = atoi(env_or("TS5_SEARCH_RESULTS", "5").c_str());
    if (n < 1) n = 1;
    if (n > 20) n = 20;
    Json body = Json::object();
    body.set("query", Json::string(query));
    Json num;
    num.type = Json::NUM;
    num.s = std::to_string(n);
    body.set("max_results", num);
    body.set("search_depth", Json::string("basic"));

    const auto t0 = std::chrono::steady_clock::now();
    HttpResponse resp;
    std::string err;
    const std::string url = env_or("TS5_TAVILY_URL", "https://api.tavily.com") + "/search";
    bool ok = http_post(url, {"Content-Type: application/json", "Authorization: Bearer " + key}, body.dump(), resp,
                        err);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char took[32];
    snprintf(took, sizeof took, "%.1f s", secs);
    if (!ok) {
        if (failed) *failed = true;
        log = "web_search \"" + query + "\": " + err + " (" + took + ")";
        return "The search failed: " + err + ". Tell the user, and answer from what you know if you can.";
    }
    Json r;
    if (resp.status != 200 || !Json::parse(resp.body, r)) {
        std::string detail = r["detail"]["error"].str(r["detail"].str(r["error"].str()));
        std::string why = resp.status == 401   ? "the Tavily API key was refused"
                          : resp.status == 432 ? "the Tavily plan's search limit is used up"
                          : resp.status == 429 ? "too many searches at once"
                                               : "Tavily answered HTTP " + std::to_string(resp.status);
        if (!detail.empty()) why += " (" + detail + ")";
        if (failed) *failed = true;
        log = "web_search \"" + query + "\": " + why + " (" + took + ")";
        return "The search failed: " + why + ". Tell the user, and answer from what you know if you can.";
    }
    std::string out = "Results for \"" + query + "\":";
    int i = 0;
    for (auto &res : r["results"].a) {
        out += "\n\n[" + std::to_string(++i) + "] " + res["title"].str() + "\n" + res["url"].str() + "\n" +
               clip(res["content"].str(), 1200);
    }
    if (i == 0) out = "No results for \"" + query + "\".";
    log = "web_search \"" + query + "\": " + std::to_string(i) + " results (" + took + ")";
    return out;
}

}  // namespace ts5
