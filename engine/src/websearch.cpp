#include "websearch.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// A failed search, as the model reads it: kFailed, why in plain words, kTell.
const char *const kFailed = "The search failed: ";
const char *const kTell = ". Tell the user, and answer from what you know if you can.";

}  // namespace

std::string tavily_key() { return env_or("TAVILY_API_KEY", ""); }

std::string search_source(const std::string &asked) {
    if (!tavily_key().empty()) return "tavily";
#ifdef TS5_PS5
    (void)asked;
    return "none";  // the console title's search is the key's alone
#else
    // The request's choice of encyclopedia, else this machine's.
    std::string want = asked == "wikipedia" || asked == "grokipedia" || asked == "both" ? asked : env_or("TS5_ENCYCLOPEDIA", "wikipedia");
    if (want != "grokipedia" && want != "both") want = "wikipedia";
    const bool wiki = env_or("TS5_WIKIPEDIA", "en") != "off";
    if (want == "wikipedia") return wiki ? "wikipedia" : "none";
    return want == "both" && wiki ? "both" : "grokipedia";
#endif
}

bool is_encyclopedia(const std::string &source) { return source == "wikipedia" || source == "grokipedia" || source == "both"; }

bool web_search_always() { return env_or("TS5_WEB_SEARCH", "") == "always"; }

bool needs_web(const std::string &user_text, bool wikipedia) {
    // Lower-case words, letters and digits only; a website (a word with
    // ".com" and the like, or "http") counts by itself.
    std::string text;
    for (char c : user_text) text += (char)tolower((unsigned char)c);
    if (wikipedia)
        for (const char *word : {"wikipedia", "wiki ", "grokipedia", "search", "encyclopedia", "encyclopaedia"})
            if (text.find(word) != std::string::npos) return true;
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

Json web_search_tool(const std::string &today) { return web_search_tool(today, search_source()); }

Json web_search_tool(const std::string &today, const std::string &source) {
    if (source != "tavily") {
        // An encyclopedia and nothing else: the model has to know, or it
        // answers "today's weather" from an article on the climate.
        const std::string what =
            source == "grokipedia" ? "Grokipedia (xAI's encyclopedia, written by its AI). Returns the top articles, each with "
                                     "its title, URL and text from it. It is the only search set up here: an encyclopedia, "
            : source == "both"     ? "Wikipedia and Grokipedia (xAI's encyclopedia, written by its AI). Returns the top "
                                     "articles of each, with title, URL and text. They are the only search set up here: "
                                     "encyclopedias, "
                                   : "Wikipedia. Returns the top articles, each with its title, URL and opening text. It is "
                                     "the only search set up here: an encyclopedia, ";
        const std::string tool =
            "{\"type\": \"function\", \"function\": {\"name\": \"web_search\", \"description\": \"Search " +
            what + "with no news, prices, weather or other live data. Use it when the user "
            "asks you to look something up, or when the answer depends on a fact you are not sure of. Do not "
            "search for explanations, writing or code: answer those yourself. When you do search, answer from "
            "the results and cite the URLs you used; if the results do not hold the answer, say so instead of "
            "guessing. Today is " +
            today +
            ".\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", "
            "\"description\": \"What to look up: a name or a few key words, not a whole question.\"}}, "
            "\"required\": [\"query\"]}}}";
        Json j;
        Json::parse(tool, j);
        return j;
    }
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

std::string search_failure(const std::string &result) {
    if (result.compare(0, strlen(kFailed), kFailed) != 0) return result;
    return result.substr(strlen(kFailed), result.rfind(kTell) - strlen(kFailed));
}

namespace {

std::string url_escape(const std::string &s) {
    std::string out;
    char hex[4];
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char)c;
        } else {
            snprintf(hex, sizeof hex, "%%%02X", c);
            out += hex;
        }
    }
    return out;
}

// A search of Wikipedia through MediaWiki's API: the articles that match,
// best first, each with its opening text. A query that finds nothing (the
// search wants every word in the article, and a model's query is often a
// whole question) goes once more as any of its words.
// The words of a search: letters and digits (and anything past ASCII), in
// lower case where they are ASCII, those of three letters or more and
// numbers.
std::vector<std::string> search_words(const std::string &query) {
    std::vector<std::string> words;
    std::string w;
    for (size_t i = 0; i <= query.size(); i++) {
        const unsigned char c = i < query.size() ? (unsigned char)query[i] : ' ';
        if (isalnum(c) || c >= 0x80) {
            w += (char)c;
        } else if (!w.empty()) {
            if (w.size() > 2 || isdigit((unsigned char)w[0])) words.push_back(w);
            w.clear();
        }
    }
    return words;
}

std::string lower_ascii(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

// Whether one of a search's own words (not "the", "what" ...) is in what was found.
struct Article {
    std::string title, url, text;
};
bool has_own_word(const std::vector<std::string> &qwords, const std::vector<Article> &found) {
    static const char *const kCommon[] = {"the",  "and",  "for",   "are",   "was",  "were",  "what", "who",  "how",
                                          "why",  "when", "where", "which", "does", "did",   "has",  "have", "had",
                                          "with", "from", "that",  "this",  "there", "about", "into", "can",  "you"};
    std::vector<std::string> own;
    for (auto &w : qwords) {
        const std::string l = lower_ascii(w);
        if (std::none_of(std::begin(kCommon), std::end(kCommon), [&](const char *c) { return l == c; })) own.push_back(l);
    }
    bool has = own.empty();
    for (auto &p : found) {
        const std::string hay = lower_ascii(p.title + "\n" + p.text);
        for (auto &w : own) has = has || hay.find(w) != std::string::npos;
    }
    return has;
}

// The line that ends a list of articles.
const char *const kNone = "\n\nIf none of these is about what was asked, search once more with only the name of the thing, or "
                          "answer from what you know and say that it could not be looked up.";

// first: the number of its first article; alone: Wikipedia is the one
// source of this search (the words say so, and the list ends with kNone).
// *count: how many articles it gave.
std::string wikipedia_search(const std::string &query, int n, std::string &log, bool *failed, int first = 1, bool alone = true,
                             int *count = nullptr) {
    const std::string lang = env_or("TS5_WIKIPEDIA", "en");
    const std::string base = env_or("TS5_WIKIPEDIA_URL", ("https://" + lang + ".wikipedia.org").c_str());
    const auto t0 = std::chrono::steady_clock::now();
    auto took = [&] {
        char b[32];
        snprintf(b, sizeof b, "%.1f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return std::string(b);
    };
    struct Page {
        double index;
        std::string title, url, text;
    };
    std::vector<Page> pages;
    const std::vector<std::string> qwords = search_words(query);
    std::string words = query;
    // The articles are a loose match: found by any of the words, or with none
    // of the search's own words in a title or an opening.
    bool loose = false;
    for (int round = 0; round < 2 && pages.empty(); round++) {
        if (round == 1) {
            // Any of the words: "PS5 OR firmware OR version".
            if (qwords.size() < 2) break;
            words.clear();
            for (auto &w : qwords) words += (words.empty() ? "" : " OR ") + w;
            loose = true;
        }
        const std::string url = base + "/w/api.php?action=query&format=json&formatversion=2&generator=search&gsrsearch=" +
                                url_escape(words) + "&gsrlimit=" + std::to_string(n) +
                                "&gsrnamespace=0&prop=extracts%7Cinfo&exintro=1&explaintext=1&exlimit=" +
                                std::to_string(n) + "&inprop=url&redirects=1";
        HttpResponse resp;
        std::string err;
        const bool ok = http_get(url, {"Accept: application/json"}, resp, err, 2u << 20, 20);
        Json r;
        if (!ok || resp.status != 200 || !Json::parse(resp.body, r)) {
            if (failed) *failed = true;
            log = "web_search \"" + query + "\": Wikipedia " +
                  (!ok ? err : resp.status != 200 ? "answered HTTP " + std::to_string(resp.status) : "sent no JSON") + " (" +
                  took() + ")";
            return kFailed + std::string(!ok ? "Wikipedia couldn't be reached" : "Wikipedia had a problem") + kTell;
        }
        for (auto &p : r["query"]["pages"].a)
            if (!p["title"].str().empty()) {
                std::string text = p["extract"].str();
                while (!text.empty() && isspace((unsigned char)text.back())) text.pop_back();
                pages.push_back({p["index"].num(1e9), p["title"].str(), p["fullurl"].str(), text});
            }
    }
    std::stable_sort(pages.begin(), pages.end(), [](const Page &a, const Page &b) { return a.index < b.index; });
    if (pages.empty()) {
        log = "web_search \"" + query + "\": 0 results (" + took() + ", Wikipedia)";
        return "Wikipedia has no article for \"" + query + "\". Try a name or fewer words, or answer from what you know "
               "and say that it could not be looked up.";
    }
    // Wikipedia's search splits and stems words ("Gated DeltaNet" finds
    // airline and river articles through "gate", "delta" and "net", seen on
    // the console): where no article has one of the search's own words in
    // its title or opening, the model is told the match is a loose one.
    if (!loose) {
        std::vector<Article> found;
        for (auto &p : pages) found.push_back({p.title, p.url, p.text});
        loose = !has_own_word(qwords, found);
    }
    const std::string only = alone ? " (the only search set up here: an encyclopedia, no news or live data)" : "";
    std::string out = loose ? "Wikipedia has no article that clearly matches \"" + query + "\"" + only +
                                  ". The closest it found, which may be about something else:"
                            : "Wikipedia articles for \"" + query + "\"" + only + ":";
    int i = 0;
    for (auto &p : pages)
        out += "\n\n[" + std::to_string(first + i++) + "] " + p.title + "\n" + p.url + "\n" +
               (p.text.empty() ? std::string("(no opening text)") : clip(p.text, 1200));
    if (alone) out += kNone;
    if (count) *count = i;
    log = "web_search \"" + query + "\": " + std::to_string(i) + " results (" + took() + ", Wikipedia" +
          (loose ? ", a loose match" : "") + ")";
    return out;
}

#ifndef TS5_PS5
// ---- Grokipedia (grokipedia.com, xAI's encyclopedia): the site's own search.

// Where the target of a Markdown link or picture ends: the ")" that closes
// the "(" at `open`, past "\)" and pairs of brackets inside it (Grokipedia's
// picture files are named "..._\(28493220635\)"). npos when the line or the
// text ends first: a passage cut in the middle of one.
size_t target_end(const std::string &md, size_t open) {
    int depth = 0;
    for (size_t i = open; i < md.size() && i - open < 2000; i++) {
        const char c = md[i];
        if (c == '\\') {
            i++;
        } else if (c == '\n') {
            break;
        } else if (c == '(') {
            depth++;
        } else if (c == ')' && --depth == 0) {
            return i;
        }
    }
    return std::string::npos;
}

// Markdown as plain words on one line: a link is its words, a picture and
// a source mark ("[](https://...)") are nothing, a heading's line goes
// (after `lead` characters of text the first heading ends it: an article's
// opening, not its next section), and so do the marks for bold and code. A
// link or picture that the text is cut in the middle of goes too. So does
// what Grokipedia puts between an article's name and its first sentence: a
// box of facts ("<!--Infobox Start ...-->", lines of "| name = value",
// "<!--Infobox End-->"), any other comment or table row, and the line
// under a picture that says what it shows ("*Tux, the mascot of Linux*").
std::string markdown_plain(const std::string &md, size_t lead = std::string::npos) {
    std::string out;
    bool line_start = true, caption = false;
    for (size_t i = 0; i < md.size(); i++) {
        const char c = md[i];
        if (line_start && (c == ' ' || c == '\t')) continue;
        if (md.compare(i, 4, "<!--") == 0) {
            size_t end = md.find("-->", i);
            if (end != std::string::npos && md.substr(i, end - i).find("Infobox Start") != std::string::npos) {
                const size_t box = md.find("Infobox End", end);
                end = box == std::string::npos ? box : md.find("-->", box);
            }
            if (end == std::string::npos) break;
            i = end + 2;
            continue;
        }
        if (line_start && c != '\n' && c != '\r') {
            const size_t nl = md.find('\n', i);
            size_t last = (nl == std::string::npos ? md.size() : nl) - 1;
            while (last > i && (md[last] == ' ' || md[last] == '\r')) last--;
            const bool said = caption && c == '*' && md[i + 1] != '*' && last > i + 1 && md[last] == '*' && md[last - 1] != '*';
            caption = false;
            if (c == '#' && out.size() >= lead) break;
            if (c == '#' || c == '|' || said) {
                if (nl == std::string::npos) break;
                i = nl;
                continue;
            }
        }
        // "[[Himalayas]]", "[[Nepal|Nepal's]]": a link written the wiki way.
        if (c == '[' && i + 1 < md.size() && md[i + 1] == '[') {
            const size_t end = md.find("]]", i + 2);
            if (end != std::string::npos && end - i < 200 && md.find('\n', i) > end) {
                std::string words = md.substr(i + 2, end - i - 2);
                const size_t bar = words.rfind('|');
                if (bar != std::string::npos) words.erase(0, bar + 1);
                out += words;
                i = end + 1;
                line_start = false;
                continue;
            }
        }
        const bool picture = c == '!' && i + 1 < md.size() && md[i + 1] == '[';
        if (c == '[' || picture) {
            const bool own_line = line_start;
            const size_t open = i + (picture ? 1 : 0);
            // (What a picture shows can be a paragraph, with \" and \( in it.)
            const size_t most = picture ? 1500 : 300;
            size_t close = open + 1;
            while (close < md.size() && md[close] != ']' && md[close] != '\n' && close - open < most)
                close += md[close] == '\\' ? 2 : 1;
            const bool closed = close < md.size() && md[close] == ']';
            if (closed && close + 1 < md.size() && md[close + 1] == '(') {
                if (!picture) out += md.substr(open + 1, close - open - 1);
                const size_t end = target_end(md, close + 1);
                if (end == std::string::npos) {
                    // Cut in its target: to the end of the line, or of the text.
                    const size_t nl = md.find('\n', close);
                    if (nl == std::string::npos) break;
                    i = nl - 1;
                } else {
                    i = end;
                }
                // (A picture alone on its line: the next line may say what it shows.)
                size_t after = i + 1;
                while (after < md.size() && (md[after] == ' ' || md[after] == '\r')) after++;
                caption = picture && own_line && (after >= md.size() || md[after] == '\n');
                line_start = false;
                continue;
            }
            // "[](" at the very end, or a picture's name the text is cut in.
            if ((!closed && close >= md.size()) || (closed && close + 1 >= md.size() && (picture || close == open + 1))) break;
        }
        if (c == '\\' && i + 1 < md.size() && strchr("()[]_*#`!.-", md[i + 1])) {
            out += md[++i];
            line_start = false;
            continue;
        }
        if (c == '*' || c == '`') continue;
        if (c == '\n' || c == '\r' || c == '\t') {
            if (!out.empty() && out.back() != ' ') out += ' ';
            if (c == '\n') line_start = true;
            continue;
        }
        line_start = false;
        if (c == ' ' && (out.empty() || out.back() == ' ')) continue;
        out += c;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// The start of an article's text: the first `bytes` of the JSON string
// "content" in an answer that may be cut off after them (an article is up
// to a megabyte, and only its opening is wanted).
std::string content_start(const std::string &body, size_t bytes) {
    size_t from = body.find("\"content\"");
    if (from == std::string::npos) return "";
    from += 9;
    while (from < body.size() && (body[from] == ' ' || body[from] == ':')) from++;
    if (from >= body.size() || body[from] != '"') return "";
    from++;
    size_t i = from, good = from;
    while (i < body.size() && i - from < bytes && body[i] != '"') {
        if (body[i] == '\\') {
            const size_t len = i + 1 < body.size() && body[i + 1] == 'u' ? 6 : 2;
            if (i + len > body.size()) break;
            i += len;
        } else {
            i++;
        }
        good = i;
    }
    Json j;
    if (!Json::parse("\"" + body.substr(from, good - from) + "\"", j)) return "";
    return j.str();
}

// What a name's article is called in an address: "black sea" is Black_Sea
// (the site answers 404 to another case).
std::string title_slug(const std::string &query) {
    static const char *const kSmall[] = {"of", "the", "and", "in", "on", "for", "to", "a", "an", "de", "von", "van", "la"};
    std::string out, w;
    bool first = true;
    for (size_t i = 0; i <= query.size(); i++) {
        const char c = i < query.size() ? query[i] : ' ';
        if (c != ' ' && c != '_') {
            w += c;
            continue;
        }
        if (w.empty()) continue;
        const bool small = !first && std::any_of(std::begin(kSmall), std::end(kSmall), [&](const char *s) { return w == s; });
        if (!small) w[0] = (char)toupper((unsigned char)w[0]);
        out += (out.empty() ? "" : "_") + w;
        w.clear();
        first = false;
    }
    return out;
}

// Characters of an address's path: a slug as it is but for what would end it.
std::string path_escape(const std::string &s) {
    std::string out;
    char hex[4];
    for (unsigned char c : s) {
        if (isalnum(c) || strchr("-_.~()',:!*", c)) {
            out += (char)c;
        } else {
            snprintf(hex, sizeof hex, "%%%02X", c);
            out += hex;
        }
    }
    return out;
}

// A search of Grokipedia through the site's own search: the articles that
// match, best first; the first with the opening of its text (one more
// request), the others with the passage the search gives. ts5 asks under
// its own name, as it asks Wikipedia's API. A site that refuses (a check
// for robots, too many requests) is not asked again in another way but for
// the article by its name, as a page: *refused says so, and the caller
// goes to Wikipedia.
std::string grokipedia_search(const std::string &query, int n, std::string &log, bool *failed, int first = 1, bool alone = true,
                              int *count = nullptr) {
    const std::string base = env_or("TS5_GROKIPEDIA_URL", "https://grokipedia.com");
    const auto t0 = std::chrono::steady_clock::now();
    auto took = [&] {
        char b[32];
        snprintf(b, sizeof b, "%.1f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return std::string(b);
    };
    const auto fail = [&](const std::string &why, const std::string &detail) {
        if (failed) *failed = true;
        log = "web_search \"" + query + "\": Grokipedia " + detail + " (" + took() + ")";
        return kFailed + why + kTell;
    };
    std::vector<Article> found;
    HttpResponse resp;
    std::string err;
    const bool ok = http_get(base + "/api/full-text-search?query=" + url_escape(query) + "&limit=" + std::to_string(n) + "&offset=0",
                             {"Accept: application/json"}, resp, err, 2u << 20, 20);
    if (!ok) return fail("Grokipedia couldn't be reached", err);
    Json r;
    const bool json = resp.status == 200 && Json::parse(resp.body, r) && r.type == Json::OBJ;
    if (json) {
        for (auto &x : r["results"].a) {
            const std::string slug = x["slug"].str(), title = x["title"].str();
            if (slug.empty() || title.empty()) continue;
            // (A passage is cut from the article's Markdown, source marks and
            // all. One that is a word or two, such as "! Operations", says
            // nothing about the article: none is better.)
            std::string text = markdown_plain(x["snippet"].str());
            if (text.size() < 20) text.clear();
            found.push_back({title, base + "/page/" + path_escape(slug), text});
            if ((int)found.size() >= n) break;
        }
        // The first article's opening, in place of the search's one line.
        if (!found.empty()) {
            HttpResponse page;
            std::string e2;
            const std::string slug = r["results"].a[0]["slug"].str();
            if (http_get(base + "/api/page-preview?slug=" + url_escape(slug) + "&content=1", {"Accept: application/json"}, page,
                         e2, 64u << 10, 20) &&
                page.status == 200) {
                // The opening: what stands before the first section, or more
                // where that is only a line or two. (The article's name, a
                // heading at the start, goes with the other headings; the
                // name that the first sentence starts with stays: "Vienna
                // is the capital...".)
                const std::string text = markdown_plain(content_start(page.body, 8000), 400);
                if (text.size() > found[0].text.size()) found[0].text = text;
            }
        }
    } else {
        // Refused, or an answer that is no search result (a page that
        // checks for robots): the article of that name, as a web page,
        // when the search is a name (a few words).
        const bool refused = resp.status == 403 || resp.status == 429 || resp.status == 503 ||
                             (resp.status == 200 && resp.body.find('<') != std::string::npos);
        const std::string detail = refused ? "refused the search (HTTP " + std::to_string(resp.status) + ")"
                                   : resp.status != 200 ? "answered HTTP " + std::to_string(resp.status) : "sent no JSON";
        const std::string slug = title_slug(query);
        HttpResponse page;
        std::string e2;
        if (!refused || search_words(query).size() > 5 || slug.empty() ||
            !http_get(base + "/page/" + path_escape(slug), page_headers(), page, e2, 2u << 20, 20) || page.status != 200 ||
            page.body.find("<article") == std::string::npos)
            return fail(refused ? "Grokipedia refused the search" : "Grokipedia had a problem", detail);
        std::string name = slug;
        for (char &c : name)
            if (c == '_') c = ' ';
        // The article's name is the page's first line, a heading: it goes
        // (not a first sentence that starts with the name).
        std::string words = html_text(page.body.substr(page.body.find("<article")));
        words.erase(0, words.find_first_not_of(" \n"));
        if (words.compare(0, name.size() + 1, name + "\n") == 0) words.erase(0, name.size());
        // (markdown_plain: on one line, the spaces closed up.)
        const std::string text = markdown_plain(words);
        found.push_back({name, base + "/page/" + path_escape(slug), text});
        log = "web_search \"" + query + "\": 1 results (" + took() + ", Grokipedia, the article by its name: the search was refused)";
    }
    if (found.empty()) {
        log = "web_search \"" + query + "\": 0 results (" + took() + ", Grokipedia)";
        if (count) *count = 0;
        return "Grokipedia has no article for \"" + query + "\"." +
               (alone ? " Try a name or fewer words, or answer from what you know and say that it could not be looked up." : "");
    }
    const bool loose = !has_own_word(search_words(query), found);
    const std::string only = alone ? " (the only search set up here: xAI's encyclopedia, written by its AI; no news or live data)"
                                   : " (xAI's encyclopedia, written by its AI)";
    std::string out = loose ? "Grokipedia has no article that clearly matches \"" + query + "\"" + only +
                                  ". The closest it found, which may be about something else:"
                            : "Grokipedia articles for \"" + query + "\"" + only + ":";
    int i = 0;
    for (auto &p : found)
        out += "\n\n[" + std::to_string(first + i++) + "] " + p.title + "\n" + p.url + "\n" +
               (p.text.empty() ? std::string("(no text)") : clip(p.text, 1200));
    if (alone) out += kNone;
    if (count) *count = i;
    if (json)
        log = "web_search \"" + query + "\": " + std::to_string(i) + " results (" + took() + ", Grokipedia" +
              (loose ? ", a loose match" : "") + ")";
    return out;
}
#endif

}  // namespace

std::string web_search(const std::string &arguments, std::string &log, bool *failed, const std::string &asked) {
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
    if (key.empty()) {
        const std::string source = search_source(asked);
        if (source == "none") {
            if (failed) *failed = true;
            log = "web_search \"" + query + "\": no search is set up";
            return kFailed + std::string("no web search is set up on this machine") + kTell;
        }
#ifndef TS5_PS5
        if (source == "grokipedia") {
            bool bad = false;
            std::string out = grokipedia_search(query, n, log, &bad);
            if (!bad) return out;
            // Grokipedia can't be searched now: Wikipedia, unless it is off, and the model is told.
            if (env_or("TS5_WIKIPEDIA", "en") == "off") {
                if (failed) *failed = true;
                return out;
            }
            const std::string why = search_failure(out), glog = log;
            out = wikipedia_search(query, n, log, failed);
            log += "; before it: " + glog.substr(glog.find("\": ") == std::string::npos ? 0 : glog.find("\": ") + 3);
            return failed && *failed ? out : "(" + why + ", so this search went to Wikipedia.)\n\n" + out;
        }
        if (source == "both") {
            // Half the results from each, Wikipedia's first.
            const int each = (n + 1) / 2;
            bool wbad = false, gbad = false;
            int wn = 0, gn = 0;
            std::string wlog, glog;
            const std::string w = wikipedia_search(query, each, wlog, &wbad, 1, false, &wn);
            const std::string g = grokipedia_search(query, each, glog, &gbad, 1 + wn, false, &gn);
            const auto tail = [&](const std::string &l) { return l.substr(l.find("\": ") == std::string::npos ? 0 : l.find("\": ") + 3); };
            log = "web_search \"" + query + "\": " + tail(wlog) + "; " + tail(glog);
            if (wbad && gbad) {
                if (failed) *failed = true;
                return kFailed + std::string("neither Wikipedia nor Grokipedia could be searched") + kTell;
            }
            std::string out = "Two encyclopedias were searched (the only search set up here: no news or live data).";
            // (On the PS5 two models said a figure was in both when only Wikipedia's passage had it.)
            if (wn > 0 && gn > 0)
                out += " [1] to [" + std::to_string(wn) + "] are Wikipedia's and [" + std::to_string(wn + 1) + "] to [" +
                       std::to_string(wn + gn) + "] are Grokipedia's. A fact is from the one whose passage says it: "
                       "name that one, and say both only when both passages say it.";
            out += "\n\n";
            out += wbad ? "(Wikipedia could not be searched: " + search_failure(w) + ".)" : w;
            out += "\n\n";
            out += gbad ? "(Grokipedia could not be searched: " + search_failure(g) + ".)" : g;
            if (wn + gn > 0) out += kNone;
            else out += "\n\nTry a name or fewer words, or answer from what you know and say that it could not be looked up.";
            return out;
        }
#endif
        return wikipedia_search(query, n, log, failed);
    }
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
        log = "web_search \"" + query + "\": " + err + " (" + took + ")";
        return kFailed + why + kTell;
    }
    Json r;
    if (resp.status != 200 || !Json::parse(resp.body, r)) {
        const std::string detail = r["detail"]["error"].str(r["detail"].str(r["error"].str()));
        const std::string why = resp.status == 401   ? "the Tavily key was refused"
                                : resp.status == 432 ? "the Tavily plan's searches are used up"
                                : resp.status == 429 ? "Tavily got too many searches at once"
                                                     : "Tavily had a problem";
        if (failed) *failed = true;
        log = "web_search \"" + query + "\": Tavily answered HTTP " + std::to_string(resp.status) +
              (detail.empty() ? std::string() : " (" + detail + ")") + " (" + took + ")";
        return kFailed + why + kTell;
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

namespace {

bool starts_ci(const std::string &s, size_t at, const char *word) {
    for (size_t k = 0; word[k]; k++)
        if (at + k >= s.size() || tolower((unsigned char)s[at + k]) != word[k]) return false;
    return true;
}

// The '>' that ends the tag opening at s[i] ('<'), or npos. A quoted
// attribute value is passed whole: a '>' inside one is its text, not the
// tag's end (Wikipedia keeps JSON in data-mw='{...}', "<b>" and all). A
// quote that never closes ends at the first '>' as before.
size_t tag_end(const std::string &s, size_t i, size_t limit = std::string::npos) {
    const size_t n = std::min(limit, s.size());
    size_t from = i;  // past the last value that did close
    for (size_t k = i + 1; k < n; k++) {
        if (s[k] == '>') return k;
        if (s[k] != '=') continue;
        size_t v = k + 1;
        while (v < n && isspace((unsigned char)s[v])) v++;
        if (v >= n || (s[v] != '"' && s[v] != '\'')) continue;
        const size_t close = s.find(s[v], v + 1);
        if (close == std::string::npos || close >= n) break;
        from = k = close;
    }
    const size_t plain = s.find('>', from);
    return plain != std::string::npos && plain < n ? plain : std::string::npos;
}

// The end of the element that opens at `at` (<script ...>) and holds no
// markup of its own: just past its closing tag, or the text's end.
size_t skip_element(const std::string &s, size_t at, const std::string &name) {
    const std::string close = "</" + name;
    for (size_t i = at; i < s.size(); i++)
        if (s[i] == '<' && starts_ci(s, i, close.c_str())) {
            const size_t e = s.find('>', i);
            return e == std::string::npos ? s.size() : e + 1;
        }
    return s.size();
}

void put_utf8(std::string &out, unsigned long cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) out += (char)(0xC0 | (cp >> 6)), out += (char)(0x80 | (cp & 0x3F));
    else if (cp < 0x10000)
        out += (char)(0xE0 | (cp >> 12)), out += (char)(0x80 | ((cp >> 6) & 0x3F)), out += (char)(0x80 | (cp & 0x3F));
    else if (cp < 0x110000)
        out += (char)(0xF0 | (cp >> 18)), out += (char)(0x80 | ((cp >> 12) & 0x3F)),
            out += (char)(0x80 | ((cp >> 6) & 0x3F)), out += (char)(0x80 | (cp & 0x3F));
}

// An entity at s[at] ('&'): its text in out and its length, or 0.
size_t entity(const std::string &s, size_t at, std::string &out) {
    const size_t semi = s.find(';', at);
    if (semi == std::string::npos || semi - at > 10) return 0;
    const std::string name = s.substr(at + 1, semi - at - 1);
    if (name.size() > 1 && name[0] == '#') {
        char *end = nullptr;
        const bool hex = name[1] == 'x' || name[1] == 'X';
        const unsigned long cp = strtoul(name.c_str() + (hex ? 2 : 1), &end, hex ? 16 : 10);
        if (!end || *end || cp == 0) return 0;
        put_utf8(out, cp);
        return semi - at + 1;
    }
    static const struct { const char *name, *text; } known[] = {
        {"amp", "&"}, {"lt", "<"}, {"gt", ">"}, {"quot", "\""}, {"apos", "'"}, {"nbsp", " "},
        {"mdash", "\xE2\x80\x94"}, {"ndash", "\xE2\x80\x93"}, {"hellip", "\xE2\x80\xA6"}, {"rsquo", "\xE2\x80\x99"},
        {"lsquo", "\xE2\x80\x98"}, {"rdquo", "\xE2\x80\x9D"}, {"ldquo", "\xE2\x80\x9C"}, {"copy", "\xC2\xA9"},
        {"reg", "\xC2\xAE"}, {"deg", "\xC2\xB0"}, {"euro", "\xE2\x82\xAC"}, {"pound", "\xC2\xA3"}, {"times", "\xC3\x97"},
        {"middot", "\xC2\xB7"}, {"bull", "\xE2\x80\xA2"}, {"laquo", "\xC2\xAB"}, {"raquo", "\xC2\xBB"},
        {"shy", ""}, {"zwnj", ""}, {"zwj", ""}, {"ensp", " "}, {"emsp", " "}, {"thinsp", " "},
        {"trade", "\xE2\x84\xA2"}, {"rarr", "\xE2\x86\x92"}, {"larr", "\xE2\x86\x90"}, {"minus", "\xE2\x88\x92"},
        {"le", "\xE2\x89\xA4"}, {"ge", "\xE2\x89\xA5"}, {"ne", "\xE2\x89\xA0"}, {"bdquo", "\xE2\x80\x9E"},
        {"sbquo", "\xE2\x80\x9A"}, {"prime", "\xE2\x80\xB2"}, {"Prime", "\xE2\x80\xB3"},
    };
    for (auto &k : known)
        if (name == k.name) {
            out += k.text;
            return semi - at + 1;
        }
    // The Latin-1 letters and signs (&eacute; &uuml; &szlig; ...), U+00A0 to U+00FF in order.
    static const char *const latin1[] = {
        "nbsp",   "iexcl",  "cent",   "pound",  "curren", "yen",    "brvbar", "sect",   "uml",    "copy",   "ordf",   "laquo",
        "not",    "shy",    "reg",    "macr",   "deg",    "plusmn", "sup2",   "sup3",   "acute",  "micro",  "para",   "middot",
        "cedil",  "sup1",   "ordm",   "raquo",  "frac14", "frac12", "frac34", "iquest", "Agrave", "Aacute", "Acirc",  "Atilde",
        "Auml",   "Aring",  "AElig",  "Ccedil", "Egrave", "Eacute", "Ecirc",  "Euml",   "Igrave", "Iacute", "Icirc",  "Iuml",
        "ETH",    "Ntilde", "Ograve", "Oacute", "Ocirc",  "Otilde", "Ouml",   "times",  "Oslash", "Ugrave", "Uacute", "Ucirc",
        "Uuml",   "Yacute", "THORN",  "szlig",  "agrave", "aacute", "acirc",  "atilde", "auml",   "aring",  "aelig",  "ccedil",
        "egrave", "eacute", "ecirc",  "euml",   "igrave", "iacute", "icirc",  "iuml",   "eth",    "ntilde", "ograve", "oacute",
        "ocirc",  "otilde", "ouml",   "divide", "oslash", "ugrave", "uacute", "ucirc",  "uuml",   "yacute", "thorn",  "yuml"};
    for (size_t k = 0; k < sizeof latin1 / sizeof *latin1; k++)
        if (name == latin1[k]) {
            put_utf8(out, 0xA0 + (unsigned long)k);
            return semi - at + 1;
        }
    return 0;
}

// Where the tag <name opens (or </name closes) at or after `from`, or npos.
size_t find_tag(const std::string &html, const char *name, size_t from, bool closing = false) {
    const std::string open = std::string(closing ? "</" : "<") + name;
    for (size_t i = from; i + open.size() <= html.size(); i++) {
        if (html[i] != '<' || !starts_ci(html, i, open.c_str())) continue;
        const size_t e = i + open.size();
        if (e >= html.size() || !isalnum((unsigned char)html[e])) return i;
    }
    return std::string::npos;
}

// Where the element <name ...> whose opening tag ends at `at` is over: after
// its own closing tag, the ones of the same name inside it counted.
size_t element_end(const std::string &s, size_t at, const std::string &name) {
    int depth = 1;
    for (size_t i = at; i < s.size(); i++) {
        if (s[i] != '<') continue;
        const bool closing = i + 1 < s.size() && s[i + 1] == '/';
        const size_t n = i + (closing ? 2 : 1);
        if (n >= s.size() || !isalpha((unsigned char)s[n])) continue;
        const size_t e = n + name.size();
        const bool same = starts_ci(s, n, name.c_str()) && !(e < s.size() && isalnum((unsigned char)s[e]));
        const size_t gt = tag_end(s, i);
        if (same && !closing) depth++;
        else if (same && --depth == 0) return gt == std::string::npos ? s.size() : gt + 1;
        if (gt == std::string::npos) break;
        i = gt;  // what the tag's attributes hold is no markup
    }
    return s.size();
}

// The value of the attribute `name` in the tag that opens at s[i] ('<') and
// ends at s[e] ('>'), its entities read; "" without it. *has: it is there.
std::string attr_of(const std::string &s, size_t i, size_t e, const char *name, bool *has = nullptr) {
    if (has) *has = false;
    size_t k = i + 1;
    while (k < e && !isspace((unsigned char)s[k])) k++;  // the tag's name
    while (k < e) {
        while (k < e && (isspace((unsigned char)s[k]) || s[k] == '/')) k++;
        const size_t n0 = k;
        while (k < e && !isspace((unsigned char)s[k]) && s[k] != '=' && s[k] != '/') k++;
        const size_t n1 = k;
        if (n1 == n0) break;
        while (k < e && isspace((unsigned char)s[k])) k++;
        std::string value;
        if (k < e && s[k] == '=') {
            k++;
            while (k < e && isspace((unsigned char)s[k])) k++;
            size_t v0 = k, v1;
            if (k < e && (s[k] == '"' || s[k] == '\'')) {
                const char q = s[k];
                v0 = ++k;
                while (k < e && s[k] != q) k++;
                v1 = k;
                if (k < e) k++;
            } else {
                while (k < e && !isspace((unsigned char)s[k])) k++;
                v1 = k;
            }
            for (size_t j = v0; j < v1;) {
                if (s[j] == '&')
                    if (const size_t len = entity(s, j, value)) {
                        j += len;
                        continue;
                    }
                value += s[j++];
            }
        }
        bool same = n1 - n0 == strlen(name);
        for (size_t j = 0; same && j < n1 - n0; j++) same = tolower((unsigned char)s[n0 + j]) == name[j];
        if (same) {
            if (has) *has = true;
            return value;
        }
    }
    return "";
}

// A label: one line, its blank runs closed up, 80 bytes at most.
std::string label_of(const std::string &text) {
    std::string out;
    for (size_t i = 0; i < text.size(); i++) {
        const unsigned char c = (unsigned char)text[i];
        if (isspace(c) || (c == 0xC2 && i + 1 < text.size() && (unsigned char)text[i + 1] == 0xA0 && ++i)) {
            if (!out.empty() && out.back() != ' ') out += ' ';
        } else out += (char)c;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    if (out.size() > 80) {
        size_t cut = 77;
        while (cut > 0 && ((unsigned char)out[cut] & 0xC0) == 0x80) cut--;
        out.resize(cut);
        out += "...";
    }
    return out;
}

// Whether the list between a and b is nothing but links: a menu, a table of
// contents, the page in other languages. Five items or more, and next to no
// words outside the links.
bool link_list(const std::string &h, size_t a, size_t b) {
    int items = 0;
    bool link = false;
    size_t in = 0, out = 0;
    for (size_t i = a; i < b;) {
        if (h[i] == '<') {
            size_t n = i + 1;
            const bool closing = n < b && h[n] == '/';
            if (closing) n++;
            size_t ne = n;
            while (ne < b && isalnum((unsigned char)h[ne])) ne++;
            std::string name = h.substr(n, ne - n);
            for (char &ch : name) ch = (char)tolower((unsigned char)ch);
            if (name == "li" && !closing) items++;
            if (name == "a") link = !closing;
            const size_t e = tag_end(h, i, b);
            i = e == std::string::npos ? b : e + 1;
            continue;
        }
        if (h[i] == '&') {  // an entity is one letter
            const size_t semi = h.find(';', i);
            if (semi != std::string::npos && semi < b && semi - i <= 10) i = semi;
        }
        if (!isspace((unsigned char)h[i])) (link ? in : out)++;
        i++;
    }
    return items >= 5 && out * 10 <= in;
}

// How many of the links in the list between a and b lead to this page in
// another language (an <a> with hreflang); 0 when the list is something
// else (fewer than four in five of its items).
int other_languages(const std::string &h, size_t a, size_t b) {
    int items = 0, langs = 0;
    for (size_t i = a; i < b;) {
        if (h[i] != '<') {
            i++;
            continue;
        }
        const size_t e = tag_end(h, i, b);
        if (e == std::string::npos) break;
        if (starts_ci(h, i, "<li") && (h[i + 3] == '>' || isspace((unsigned char)h[i + 3]))) items++;
        if (starts_ci(h, i, "<a") && isspace((unsigned char)h[i + 2])) {
            bool has = false;
            attr_of(h, i, e, "hreflang", &has);
            langs += has;
        }
        i = e + 1;
    }
    return langs * 5 >= items * 4 ? langs : 0;
}

// HTML as text. menus false: without the page's menus and sidebars (<nav>,
// <aside>, and the lists that are nothing but links). els (a page view, for
// a model that acts on the page): each link, text box and button gets a
// number in the text, "[3]" after a link's words, "[4: text box "Search"]",
// "[5: button "Go"]", and is listed in els under it; the lists of links
// stay (they are what there is to click), what the page hides doesn't.
std::string text_of(const std::string &html, bool menus, PageElements *els = nullptr) {
    // Tags that start a new line, and the ones whose insides are no text.
    static const char *const kBlock[] = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
                                         "article", "header", "footer", "table", "ul", "ol", "pre", "blockquote",
                                         "title", "dt", "dd", "hr", "form", "nav", "main", "aside", "figure"};
    static const char *const kSkip[] = {"script", "style", "noscript", "svg", "template", "iframe", "select"};
    std::string raw;
    // The link or button being read: its words become its label when it closes.
    struct {
        bool on = false;
        std::string tag;  // its tag's name, and how many of them are open inside it
        int depth = 0;
        size_t from = 0;  // where its words start in raw
        PageElement el;
    } cap;
    int form = -1;  // the form the tags are inside of
    // A number for el, and its mark in the text.
    auto number = [&](PageElement el) {
        if (els->items.size() >= 400) return std::string();
        el.form = form;
        els->items.push_back(el);
        const std::string n = std::to_string(els->items.size());
        const std::string &l = els->items.back().label;
        if (el.kind == "link") return "[" + n + "]";
        if (el.kind == "button") return "[" + n + ": button" + (l.empty() ? "" : " \"" + l + "\"") + "]";
        if (el.kind == "check")
            return "[" + n + ": " + (el.checked ? "ticked" : "empty") + " box" + (l.empty() ? "" : " \"" + l + "\"") + "]";
        return "[" + n + ": text box" + (l.empty() ? "" : " \"" + l + "\"") +
               (el.value.empty() ? "" : " = \"" + label_of(el.value) + "\"") + "]";
    };
    auto close_cap = [&] {
        std::string words = label_of(raw.substr(std::min(cap.from, raw.size())));
        // A link to a place on this very page.
        const bool jump = cap.el.kind == "link" && cap.el.href.size() > 1 && cap.el.href[0] == '#';
        if (jump) {
            // A footnote's mark ("[9]", "[a]", "[note 2]") or its way back
            // ("^"): left out whole. Beside a link's own number it reads as
            // a second number (Wikipedia: "the Black Sea[113].[9][125]"),
            // and on the PS5 two models then could not tell which of them
            // browser_click takes.
            const bool mark = (words.size() >= 3 && words.size() <= 22 && words.front() == '[' && words.back() == ']' &&
                               words.find('[', 1) == std::string::npos) ||
                              words == "^" || words == "\xE2\x86\x91" || starts_ci(cap.el.href, 0, "#cite_");
            cap.on = false;
            if (mark) {
                raw.resize(std::min(cap.from, raw.size()));
                return;
            }
            // Any other is numbered only for a browser, which shows what
            // the page does on a click; its words stay either way.
            if (cap.el.mark.empty()) return;
        }
        if (words.empty()) words = cap.el.label;  // its aria-label or title
        cap.on = false;
        if (words.empty()) return;
        cap.el.label = words;
        if (cap.el.kind == "link") {
            while (raw.size() > cap.from && raw.back() == ' ') raw.pop_back();
            raw += number(cap.el);
        } else {
            raw.resize(std::min(cap.from, raw.size()));
            raw += number(cap.el);
        }
    };
    for (size_t i = 0; i < html.size();) {
        const char c = html[i];
        if (c == '<') {
            if (html.compare(i, 4, "<!--") == 0) {
                const size_t e = html.find("-->", i + 4);
                i = e == std::string::npos ? html.size() : e + 3;
                continue;
            }
            size_t n = i + 1;
            const bool closing = n < html.size() && html[n] == '/';
            if (closing) n++;
            size_t ne = n;
            while (ne < html.size() && isalnum((unsigned char)html[ne])) ne++;
            std::string name = html.substr(n, ne - n);
            for (char &ch : name) ch = (char)tolower((unsigned char)ch);
            if (name.empty() && !(n < html.size() && (html[n] == '!' || html[n] == '?'))) {
                raw += c;  // a bare <: text
                i++;
                continue;
            }
            const size_t e = tag_end(html, i);
            const size_t after = e == std::string::npos ? html.size() : e + 1;
            bool skip = false;
            for (const char *k : kSkip) skip |= !closing && name == k;
            skip |= !menus && !closing && (name == "nav" || name == "aside");
            if (skip) {
                // (A <select>'s choice still goes with its form.)
                if (els && name == "select" && form >= 0 && e != std::string::npos) {
                    const std::string field = attr_of(html, i, e, "name");
                    const size_t end = skip_element(html, after, name);
                    std::string value;
                    bool first = true;
                    for (size_t o = find_tag(html, "option", after); o != std::string::npos && o < end;
                         o = find_tag(html, "option", o + 1)) {
                        const size_t oe = tag_end(html, o);
                        if (oe == std::string::npos) break;
                        bool chosen = false, has = false;
                        attr_of(html, o, oe, "selected", &chosen);
                        std::string v = attr_of(html, o, oe, "value", &has);
                        if (!has) {
                            const size_t stop = std::min(html.find('<', oe + 1), end);
                            v = label_of(html.substr(oe + 1, stop - oe - 1));
                        }
                        if (first || chosen) value = v;
                        first = false;
                    }
                    if (!field.empty()) els->forms[(size_t)form].fields.push_back({field, value, -1});
                }
                i = skip_element(html, after, name);
                continue;
            }
            if (els && !closing && e != std::string::npos) {
                // What the page hides (a closed menu, a dialog not open).
                bool hidden = false, flag = false;
                attr_of(html, i, e, "hidden", &hidden);
                attr_of(html, i, e, "data-ts5-hidden", &flag);
                hidden |= flag || attr_of(html, i, e, "aria-hidden") == "true";
                if (!hidden) {
                    std::string style = attr_of(html, i, e, "style");
                    style.erase(std::remove_if(style.begin(), style.end(), [](unsigned char ch) { return isspace(ch); }),
                                style.end());
                    hidden = style.find("display:none") != std::string::npos ||
                             style.find("visibility:hidden") != std::string::npos;
                }
                static const char *const kVoid[] = {"input", "img", "br", "hr", "meta", "link", "area", "base", "col",
                                                    "embed", "source", "track", "wbr"};
                bool is_void = html[e - 1] == '/';
                for (const char *k : kVoid) is_void |= name == k;
                if (hidden && name != "html" && name != "body") {
                    i = is_void ? after : element_end(html, after, name);
                    continue;
                }
                const std::string mark = attr_of(html, i, e, "data-ts5");
                const std::string role = attr_of(html, i, e, "role");
                if (cap.on) {
                    if (name == cap.tag && !is_void) cap.depth++;
                    if (name == "img") raw += " " + attr_of(html, i, e, "alt") + " ";
                } else if (name == "form") {
                    form = (int)els->forms.size();
                    PageForm f;
                    f.action = attr_of(html, i, e, "action");
                    f.method = attr_of(html, i, e, "method");
                    for (char &ch : f.method) ch = (char)tolower((unsigned char)ch);
                    els->forms.push_back(f);
                } else if (name == "a" ||
                           // A button with no browser behind it does something only as its form's.
                           (name == "button" && (!mark.empty() || form >= 0)) ||
                           // Whatever else a browser marked as taking a click.
                           (!mark.empty() && !is_void && name != "textarea" && name != "select" && name != "form")) {
                    PageElement el;
                    el.mark = mark;
                    el.kind = name == "a" || role == "link" ? "link" : "button";
                    el.href = name == "a" ? attr_of(html, i, e, "href") : "";
                    el.name = attr_of(html, i, e, "name");
                    el.value = attr_of(html, i, e, "value");
                    el.type = attr_of(html, i, e, "type");
                    for (char &ch : el.type) ch = (char)tolower((unsigned char)ch);
                    el.label = label_of(attr_of(html, i, e, "aria-label"));
                    if (el.label.empty()) el.label = label_of(attr_of(html, i, e, "title"));
                    // A link that goes nowhere a request can follow is one
                    // only for a browser, which runs what the page does on a click.
                    const bool dead = el.href.empty() || el.href[0] == '#' || starts_ci(el.href, 0, "javascript:") ||
                                      starts_ci(el.href, 0, "mailto:") || starts_ci(el.href, 0, "tel:");
                    const bool inert = name == "button" && mark.empty() && (el.type == "button" || el.type == "reset");
                    // (A link to a place on this page is looked at once its
                    // words are known: close_cap.)
                    const bool jump = name == "a" && el.href.size() > 1 && el.href[0] == '#';
                    if ((name != "a" || !dead || !mark.empty() || jump) && !inert) {
                        cap.on = true;
                        cap.tag = name;
                        cap.depth = 1;
                        cap.from = raw.size();
                        cap.el = el;
                    }
                } else if (name == "input" || name == "textarea") {
                    PageElement el;
                    el.mark = mark;
                    el.name = attr_of(html, i, e, "name");
                    el.type = name == "textarea" ? "textarea" : attr_of(html, i, e, "type");
                    for (char &ch : el.type) ch = (char)tolower((unsigned char)ch);
                    if (el.type.empty()) el.type = "text";
                    bool live = false;
                    el.value = attr_of(html, i, e, "data-ts5-value", &live);  // what a browser has in it now
                    if (!live) el.value = attr_of(html, i, e, "value");
                    size_t next = after;
                    if (name == "textarea") {
                        next = skip_element(html, after, name);
                        if (!live) {
                            const size_t stop = html.rfind("</", next);
                            el.value = stop != std::string::npos && stop >= after ? html.substr(after, stop - after) : "";
                        }
                    }
                    el.label = label_of(attr_of(html, i, e, "placeholder"));
                    if (el.label.empty()) el.label = label_of(attr_of(html, i, e, "aria-label"));
                    if (el.label.empty()) el.label = label_of(attr_of(html, i, e, "title"));
                    if (el.label.empty()) el.label = label_of(el.name);
                    attr_of(html, i, e, "checked", &el.checked);
                    const std::string &t = el.type;
                    const int f = form;
                    if (t == "hidden") {
                        if (f >= 0 && !el.name.empty()) els->forms[(size_t)f].fields.push_back({el.name, el.value, -1});
                    } else if (t == "password") {
                        raw += "[password box: only the user can fill this in]";
                        // (Sent empty, as a browser sends a box nothing was typed in.)
                        if (f >= 0 && !el.name.empty()) els->forms[(size_t)f].fields.push_back({el.name, "", -1});
                    } else if (t == "file") {
                        if (f >= 0 && !el.name.empty()) els->forms[(size_t)f].fields.push_back({el.name, "", -1});
                    } else if (t == "submit" || t == "button" || t == "image" || t == "reset") {
                        el.kind = "button";
                        el.label = label_of(el.value.empty() ? (t == "submit" ? "Submit" : el.label) : el.value);
                        if (t != "reset") raw += number(el);
                    } else if (t == "checkbox" || t == "radio") {
                        el.kind = "check";
                        if (el.value.empty()) el.value = "on";
                        raw += number(el);
                        if (f >= 0 && !el.name.empty())
                            els->forms[(size_t)f].fields.push_back({el.name, el.value, (int)els->items.size() - 1});
                    } else {
                        el.kind = "input";
                        raw += number(el);
                        if (f >= 0 && !el.name.empty())
                            els->forms[(size_t)f].fields.push_back({el.name, el.value, (int)els->items.size() - 1});
                    }
                    i = next;
                    continue;
                }
            }
            if (els && closing) {
                if (cap.on && name == cap.tag && --cap.depth == 0) close_cap();
                else if (!cap.on && name == "form") form = -1;
            }
            if (!menus && !els && !closing && (name == "ul" || name == "ol")) {
                const size_t end = element_end(html, after, name);
                if (link_list(html, after, end)) {
                    i = end;
                    continue;
                }
            }
            // (A page view.) The page in other languages: Wikipedia's list
            // of 178 filled the first part of every article, and pushed the
            // article's own links to numbers a model then picked wrongly
            // (a browser hides the list behind a button). A line stands for it.
            if (els && !closing && !cap.on && (name == "ul" || name == "ol")) {
                const size_t end = element_end(html, after, name);
                const int langs = other_languages(html, after, end);
                if (langs >= 10) {
                    raw += "\n(" + std::to_string(langs) + " links to this page in other languages are left out here.)\n";
                    i = end;
                    continue;
                }
            }
            for (const char *k : kBlock)
                if (name == k) {
                    if (name == "li" && closing) break;
                    raw += name == "li" ? "\n- " : "\n";
                    if (name[0] == 'h' && name.size() == 2 && isdigit((unsigned char)name[1])) raw += "\n";
                    break;
                }
            if (name == "td" || name == "th") raw += closing ? " | " : "";
            i = after;
            continue;
        }
        if (c == '&') {
            if (const size_t n = entity(html, i, raw)) {
                i += n;
                continue;
            }
        }
        raw += c == '\r' || c == '\t' || c == '\n' ? ' ' : c;
        i++;
    }
    // Lines trimmed, runs of spaces closed up, two newlines at most.
    std::string out, line;
    int blank = 2;  // nothing at the top
    auto flush = [&] {
        while (!line.empty() && line.back() == ' ') line.pop_back();
        if (line.empty() || line == "-" || line == "|") {
            if (blank++ == 0) out += "\n";
        } else {
            out += line + "\n";
            blank = 0;
        }
        line.clear();
    };
    for (size_t i = 0; i < raw.size(); i++) {
        const char c = raw[i];
        if (c == '\n') flush();
        else if (c == ' ' || (c == '\xC2' && i + 1 < raw.size() && raw[i + 1] == '\xA0' && ++i)) {
            if (!line.empty() && line.back() != ' ') line += ' ';
        } else line += c;
    }
    flush();
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

}  // namespace

std::string html_view(const std::string &html, PageElements &els) {
    // As html_text: the article alone where the page marks one, under its
    // title; a page that is little else is read whole.
    PageElements all;
    const std::string whole = text_of(html, true, &all);
    PageElements part;
    std::string text;
    const size_t m = find_tag(html, "main", 0);
    if (m != std::string::npos) {
        size_t e = std::string::npos;
        for (size_t at = find_tag(html, "main", m, true); at != std::string::npos; at = find_tag(html, "main", at + 1, true)) e = at;
        // The forms above the article go first (a site's search box is in
        // its header): two at most, and none that is a page of its own.
        std::string lead;
        int forms = 0;
        for (size_t f = find_tag(html, "form", 0); f != std::string::npos && f < m && forms < 2;
             f = find_tag(html, "form", f + 1)) {
            const size_t open = tag_end(html, f);
            if (open == std::string::npos) break;
            const size_t fe = element_end(html, open + 1, "form");
            if (fe > m || fe - f > 6000) continue;
            const std::string form = html.substr(f, fe - f);
            if (find_tag(form, "input", 0) == std::string::npos) continue;
            lead += "<div>" + form + "</div>";
            forms++;
        }
        text = text_of(lead + html.substr(m, e == std::string::npos ? std::string::npos : e - m), false, &part);
    } else {
        text = text_of(html, false, &part);
    }
    const bool use_part = text.size() >= 200 || text.size() * 2 >= whole.size();
    els = use_part ? part : all;
    return use_part ? text : whole;
}

std::string html_title(const std::string &html) {
    const size_t t = find_tag(html, "title", 0);
    if (t == std::string::npos) return "";
    const size_t te = find_tag(html, "title", t, true);
    return label_of(text_of(html.substr(t, te == std::string::npos ? 300 : te - t), true));
}

std::string html_text(const std::string &html) {
    // The article alone where the page marks it (<main>), under the page's
    // title, and without menus and sidebars: on a long page those would
    // take the model's context before the text starts (Wikipedia: 800
    // characters of menus first). A page that is little else (a list of
    // links) is read whole.
    const std::string whole = text_of(html, true);
    std::string text;
    const size_t m = find_tag(html, "main", 0);
    if (m != std::string::npos) {
        size_t e = std::string::npos;
        for (size_t at = find_tag(html, "main", m, true); at != std::string::npos; at = find_tag(html, "main", at + 1, true)) e = at;
        text = text_of(html.substr(m, e == std::string::npos ? std::string::npos : e - m), false);
        const size_t t = find_tag(html, "title", 0);
        if (t != std::string::npos && t < m) {
            const size_t te = find_tag(html, "title", t, true);
            const std::string title = text_of(html.substr(t, te == std::string::npos || te > m ? 0 : te - t), true);
            if (!title.empty() && text.compare(0, title.size(), title) != 0) text = title + "\n\n" + text;
        }
    } else {
        text = text_of(html, false);
    }
    return text.size() >= 200 || text.size() * 2 >= whole.size() ? text : whole;
}

std::vector<std::string> page_headers() {
    const char *as = getenv("TS5_BROWSER_AGENT");
    if (as && std::string(as) == "own") return {"Accept: text/html,text/plain;q=0.9,*/*;q=0.5"};
    // Chrome's on Linux (its version numbers after the first are zeros for
    // every Chrome). Raise the number with the years.
    return {"User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 "
            "Safari/537.36",
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language: en-US,en;q=0.9",
            "Upgrade-Insecure-Requests: 1"};
}

std::string web_page(const std::string &url, size_t max_chars, std::string &log, bool *failed) {
    auto fail = [&](const std::string &why, const std::string &detail) {
        if (failed) *failed = true;
        log = "web page " + url + ": " + detail;
        return "The page could not be read: " + why + ".";
    };
    if (url.compare(0, 7, "http://") != 0 && url.compare(0, 8, "https://") != 0)
        return fail("its address has to start with http:// or https://", "not an http(s) address");
    const auto t0 = std::chrono::steady_clock::now();
    HttpResponse resp;
    std::string err;
    const bool ok = http_get(url, page_headers(), resp, err, 2u << 20, 20);
    char took[32];
    snprintf(took, sizeof took, "%.1f s",
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (!ok)
        return fail(err.find("certificate") != std::string::npos || err.find("TLS") != std::string::npos
                        ? "the site's identity couldn't be checked"
                        : "the site couldn't be reached",
                    err + " (" + took + ")");
    if (resp.status < 200 || resp.status >= 300)
        return fail(resp.status == 404   ? "there is no such page (404)"
                    : resp.status == 403 ? "the site refused (403)"
                    : resp.status == 429 ? "the site says too many requests (429)"
                                         : "the site answered with error " + std::to_string(resp.status),
                    "HTTP " + std::to_string(resp.status) + " (" + took + ")");
    const std::string &b = resp.body;
    if (b.find('\0') != std::string::npos) return fail("it is not text (a file, an image or a PDF)", "binary");
    bool html = false;
    for (size_t i = 0; i < b.size() && i < 2048 && !html; i++)
        html = b[i] == '<' && (starts_ci(b, i, "<html") || starts_ci(b, i, "<!doctype") || starts_ci(b, i, "<head") ||
                               starts_ci(b, i, "<body") || starts_ci(b, i, "<div") || starts_ci(b, i, "<p"));
    std::string text = html ? html_text(b) : b;
    while (!text.empty() && isspace((unsigned char)text.back())) text.pop_back();
    const size_t whole = text.size();
    if (max_chars > 0 && text.size() > max_chars) {
        size_t cut = max_chars;
        while (cut > 0 && ((unsigned char)text[cut] & 0xC0) == 0x80) cut--;
        text.resize(cut);
        text += "\n[... the page goes on: " + std::to_string(whole - cut) + " more characters are left out ...]";
    }
    if (text.empty()) return fail("it has no text (a page that needs a browser to show anything)", "no text");
    log = "web page " + url + ": " + std::to_string(whole) + " characters (" + took + ")";
    return text;
}

}  // namespace ts5
