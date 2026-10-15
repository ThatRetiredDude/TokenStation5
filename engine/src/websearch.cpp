#include "websearch.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

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

Json web_search_tool(const std::string &today) {
    std::string tool =
        "{\"type\": \"function\", \"function\": {\"name\": \"web_search\", \"description\": \"Search the web. "
        "Returns the top results, each with a title, URL and text from the page. Use it for anything recent, "
        "anything you are unsure of, and anything the user asks you to look up; then answer from the results "
        "and cite the URLs you used. Today is " +
        today +
        ".\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", "
        "\"description\": \"What to search for, as you would type it into a search engine.\"}}, "
        "\"required\": [\"query\"]}}}";
    Json j;
    Json::parse(tool, j);
    return j;
}

std::string web_search(const std::string &arguments, std::string &log, bool *failed, std::string *why_failed) {
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
    // Why a search failed: the network's or Tavily's own words in the log,
    // plain ones for the model, which tells the user.
    if (!ok) {
        const std::string why = err.find("certificate") != std::string::npos || err.find("TLS") != std::string::npos
                                    ? "Tavily's identity couldn't be checked"
                                    : "the PS5 couldn't reach Tavily";
        if (failed) *failed = true;
        if (why_failed) *why_failed = why;
        log = "web_search \"" + query + "\": " + err + " (" + took + ")";
        return "The search failed: " + why + ". Tell the user, and answer from what you know if you can.";
    }
    Json r;
    if (resp.status != 200 || !Json::parse(resp.body, r)) {
        const std::string detail = r["detail"]["error"].str(r["detail"].str(r["error"].str()));
        const std::string why = resp.status == 401   ? "the Tavily key was refused"
                                : resp.status == 432 ? "the Tavily plan's searches are used up"
                                : resp.status == 429 ? "Tavily got too many searches at once"
                                                     : "Tavily had a problem";
        if (failed) *failed = true;
        if (why_failed) *why_failed = why;
        log = "web_search \"" + query + "\": Tavily answered HTTP " + std::to_string(resp.status) +
              (detail.empty() ? std::string() : " (" + detail + ")") + " (" + took + ")";
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
