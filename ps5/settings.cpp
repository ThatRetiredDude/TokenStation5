#include "settings.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace app {

namespace {

bool is_name(const std::string &word, const std::string &name) {
    return word.size() > name.size() && word.compare(0, name.size(), name) == 0 && word[name.size()] == '=';
}

// A word as read_words reads it back.
std::string quoted(const std::string &w) {
    return w.find_first_of(" \t") == std::string::npos ? w : "\"" + w + "\"";
}

}  // namespace

std::vector<std::string> line_words(const std::string &line) {
    std::vector<std::string> out;
    size_t i = line.find_first_not_of(" \t");
    if (i == std::string::npos || line[i] == '#') return out;
    std::string word;
    bool in_word = false, quote = false;
    for (; i < line.size(); i++) {
        const char c = line[i];
        if (c == '"') {
            quote = !quote;
            in_word = true;
        } else if (!quote && strchr(" \t\r\n", c)) {
            if (in_word) out.push_back(word);
            word.clear();
            in_word = false;
        } else {
            word += c;
            in_word = true;
        }
    }
    if (in_word) out.push_back(word);
    return out;
}

bool set_setting(const std::string &path, const std::string &name, const std::string &value, std::string &err) {
    std::vector<std::string> lines;
    if (FILE *f = fopen(path.c_str(), "r")) {
        char buf[4096];
        std::string line;
        while (fgets(buf, sizeof buf, f)) {
            line += buf;
            if (line.back() != '\n') continue;
            line.pop_back();
            lines.push_back(line);
            line.clear();
        }
        if (!line.empty()) lines.push_back(line);
        fclose(f);
    }
    // Every NAME=... goes; a line that held others keeps them.
    std::vector<std::string> out;
    for (const std::string &l : lines) {
        const std::vector<std::string> words = line_words(l);
        bool found = false;
        std::string kept;
        for (const std::string &w : words) {
            if (is_name(w, name)) found = true;
            else kept += (kept.empty() ? "" : " ") + quoted(w);
        }
        if (!found) out.push_back(l);
        else if (!kept.empty()) out.push_back(kept);
    }
    if (!value.empty()) out.push_back(quoted(name + "=" + value));

    // Written next to it, then moved over it: a half-written file never
    // takes the settings with it.
    const std::string tmp = path + ".new";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) {
        err = "can't write " + tmp + ": " + strerror(errno);
        return false;
    }
    bool ok = true;
    for (const std::string &l : out) ok = ok && fprintf(f, "%s\n", l.c_str()) >= 0;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        err = "can't write " + path + ": " + strerror(errno);
        remove(tmp.c_str());
        return false;
    }
    if (value.empty()) unsetenv(name.c_str());
    else setenv(name.c_str(), value.c_str(), 1);
    printf("ts5-title: settings: %s\n", (value.empty() ? name + " cleared"
                                         : secret_setting(name) ? name + "=(set)"
                                                                : name + "=" + value)
                                            .c_str());
    return true;
}

std::string setting(const char *name, const std::string &def) {
    const char *v = getenv(name);
    return v && *v ? v : def;
}

bool secret_setting(const std::string &name) {
    auto ends = [&](const char *e) {
        const size_t n = strlen(e);
        return name.size() >= n && name.compare(name.size() - n, n, e) == 0;
    };
    return ends("_KEY") || ends("_TOKEN");
}

}  // namespace app
