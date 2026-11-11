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

std::string search_failure(const std::string &result) {
    if (result.compare(0, strlen(kFailed), kFailed) != 0) return result;
    return result.substr(strlen(kFailed), result.rfind(kTell) - strlen(kFailed));
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
        if (!starts_ci(s, n, name.c_str())) continue;
        const size_t e = n + name.size();
        if (e < s.size() && isalnum((unsigned char)s[e])) continue;
        if (!closing) {
            depth++;
        } else if (--depth == 0) {
            const size_t gt = s.find('>', i);
            return gt == std::string::npos ? s.size() : gt + 1;
        }
    }
    return s.size();
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
            const size_t e = h.find('>', i);
            i = e == std::string::npos || e >= b ? b : e + 1;
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

// HTML as text. menus false: without the page's menus and sidebars (<nav>,
// <aside>, and the lists that are nothing but links).
std::string text_of(const std::string &html, bool menus) {
    // Tags that start a new line, and the ones whose insides are no text.
    static const char *const kBlock[] = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
                                         "article", "header", "footer", "table", "ul", "ol", "pre", "blockquote",
                                         "title", "dt", "dd", "hr", "form", "nav", "main", "aside", "figure"};
    static const char *const kSkip[] = {"script", "style", "noscript", "svg", "template", "iframe", "select"};
    std::string raw;
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
            const size_t e = html.find('>', i);
            const size_t after = e == std::string::npos ? html.size() : e + 1;
            bool skip = false;
            for (const char *k : kSkip) skip |= !closing && name == k;
            skip |= !menus && !closing && (name == "nav" || name == "aside");
            if (skip) {
                i = skip_element(html, after, name);
                continue;
            }
            if (!menus && !closing && (name == "ul" || name == "ol")) {
                const size_t end = element_end(html, after, name);
                if (link_list(html, after, end)) {
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
    const bool ok = http_get(url, {"User-Agent: Mozilla/5.0 (X11; Linux x86_64) TokenStation",
                                   "Accept: text/html,text/plain;q=0.9,*/*;q=0.5"},
                             resp, err, 2u << 20, 20);
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
