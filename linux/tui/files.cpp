// The model's file tools (files.h).
#include "files.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "app.h"
#include "json.h"
#include "store.h"

namespace tui {

namespace {

const size_t kMaxFile = 8u << 20;  // a file read or changed whole: no bigger
const size_t kReadChars = 8000;    // one read_file answer
const size_t kReadLines = 300;
const size_t kShownLines = 30;  // lines out, and lines in, of a change the user is shown
const size_t kLineChars = 200;

std::string tool(const char *name, const char *what, const std::vector<std::vector<std::string>> &params) {
    std::string props, required;
    for (auto &p : params) {
        std::string n = p[0];
        const bool optional = !n.empty() && n.back() == '?';
        if (optional) n.pop_back();
        props += (props.empty() ? "" : ",") + ts5::json_escape(n) + ":{\"type\":" + ts5::json_escape(p[1]) +
                 ",\"description\":" + ts5::json_escape(p[2]) + "}";
        if (!optional) required += (required.empty() ? "" : ",") + ts5::json_escape(n);
    }
    return std::string("{\"type\":\"function\",\"function\":{\"name\":\"") + name + "\",\"description\":" +
           ts5::json_escape(what) + ",\"parameters\":{\"type\":\"object\",\"properties\":{" + props +
           "},\"required\":[" + required + "]}}}";
}

std::string start_dir() {
    char buf[PATH_MAX];
    std::string d = getcwd(buf, sizeof buf) ? buf : "/";
    char real[PATH_MAX];
    return realpath(d.c_str(), real) ? real : d;
}

std::string home() {
    const char *h = getenv("HOME");
    std::string d = h && *h ? h : "";
    char real[PATH_MAX];
    return !d.empty() && realpath(d.c_str(), real) ? real : d;
}

// A path as given by the model, made whole: ~ is the home folder, a
// relative one starts at the start folder, "." and ".." are worked out,
// and links are followed as far as the path exists (a link out of the
// start folder is a path outside it).
std::string whole_path(std::string given) {
    while (!given.empty() && (given.back() == ' ' || given.back() == '\n')) given.pop_back();
    while (!given.empty() && given[0] == ' ') given.erase(0, 1);
    if (given.empty()) return "";
    if (given == "~") given = home();
    else if (given.compare(0, 2, "~/") == 0) given = home() + given.substr(1);
    if (given[0] != '/') given = start_dir() + "/" + given;
    std::vector<std::string> parts;
    for (size_t i = 0; i <= given.size();) {
        const size_t e = std::min(given.find('/', i), given.size());
        const std::string p = given.substr(i, e - i);
        if (p == "..") {
            if (!parts.empty()) parts.pop_back();
        } else if (!p.empty() && p != ".") {
            parts.push_back(p);
        }
        i = e + 1;
    }
    // The longest part of it that exists, by its real name; the rest as written.
    for (size_t keep = parts.size();; keep--) {
        std::string head = "/";
        for (size_t k = 0; k < keep; k++) head += (k ? "/" : "") + parts[k];
        char real[PATH_MAX];
        if (realpath(head.c_str(), real)) {
            std::string out = real;
            for (size_t k = keep; k < parts.size(); k++) out += (out == "/" ? "" : "/") + parts[k];
            return out;
        }
        if (keep == 0) break;
    }
    std::string out;
    for (auto &p : parts) out += "/" + p;
    return out.empty() ? "/" : out;
}

bool hidden_path(const std::string &path) { return path.find("/.") != std::string::npos; }

std::string folder_of(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos || slash == 0 ? "/" : path.substr(0, slash);
}

std::string arg(const ts5::Json &j, const char *name) { return j[name].str(); }

// A file's bytes. false with why: no such file, a folder, too big, unreadable.
bool read_whole(const std::string &path, const std::string &shown, std::string &out, std::string &why) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        why = errno == ENOENT ? "There is no file " + shown + ". list_folder shows what is in a folder."
                              : shown + " can't be read: " + strerror(errno) + ".";
        return false;
    }
    if (S_ISDIR(st.st_mode)) {
        why = shown + " is a folder, not a file. list_folder shows what is in it.";
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        why = shown + " is not a file that can be read.";
        return false;
    }
    if ((size_t)st.st_size > kMaxFile) {
        why = shown + " is too big for this tool (" + fmt("%.1f", st.st_size / 1048576.0) + " MiB; it takes files up to 8 MiB).";
        return false;
    }
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        why = shown + " can't be read: " + strerror(errno) + ".";
        return false;
    }
    out.clear();
    char buf[65536];
    ssize_t n;
    while ((n = read(fd, buf, sizeof buf)) > 0 && out.size() <= kMaxFile) out.append(buf, (size_t)n);
    close(fd);
    if (out.find('\0') != std::string::npos) {
        why = shown + " is not a text file (" + std::to_string(out.size()) + " bytes).";
        return false;
    }
    return true;
}

// Whole lines, each without its newline (a last line with none counts).
std::vector<std::string> lines_of(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const size_t e = std::min(s.find('\n', i), s.size());
        out.push_back(s.substr(i, e - i));
        i = e + 1;
    }
    return out;
}

std::string cut_line(const std::string &l) {
    if (l.size() <= kLineChars) return l;
    size_t n = kLineChars;
    while (n > 0 && ((unsigned char)l[n] & 0xC0) == 0x80) n--;  // not inside a character
    return l.substr(0, n) + "…";
}

std::string count(size_t n, const char *one) { return std::to_string(n) + " " + one + (n == 1 ? "" : "s"); }

// A change worked out: the file before and after.
struct Change {
    std::string path, shown, before, after, error;
    bool existed = false, whole = false;  // whole: write_file
    size_t first = 0, out = 0, in = 0;    // the lines that differ: from line `first` (counted from 0), so many out and in
};

void measure(Change &c) {
    const std::vector<std::string> a = lines_of(c.before), b = lines_of(c.after);
    size_t p = 0;
    while (p < a.size() && p < b.size() && a[p] == b[p]) p++;
    size_t s = 0;
    while (s < a.size() - p && s < b.size() - p && a[a.size() - 1 - s] == b[b.size() - 1 - s]) s++;
    c.first = p;
    c.out = a.size() - p - s;
    c.in = b.size() - p - s;
}

Change change_of(const std::string &name, const ts5::Json &j) {
    Change c;
    c.path = whole_path(arg(j, "path"));
    c.shown = path_shown(c.path);
    c.whole = name == "write_file";
    if (c.path.empty()) {
        c.error = "The call gave no path. Give the file's path.";
        return c;
    }
    struct stat st;
    c.existed = stat(c.path.c_str(), &st) == 0;
    if (c.whole) {
        c.after = arg(j, "content");
        if (c.existed && !read_whole(c.path, c.shown, c.before, c.error)) return c;
        if (c.existed && c.before == c.after) c.error = c.shown + " already has exactly this content. Nothing to change.";
        if (!c.existed) {
            const std::string dir = folder_of(c.path);
            struct stat ds;
            // (A folder that is missing is made with the file.)
            if (stat(dir.c_str(), &ds) == 0 && !S_ISDIR(ds.st_mode)) c.error = path_shown(dir) + " is a file, not a folder.";
        }
        if (c.after.size() > kMaxFile) c.error = "That is too much for one file (8 MiB at most).";
    } else {
        const std::string old_text = arg(j, "old_text"), new_text = arg(j, "new_text");
        if (!read_whole(c.path, c.shown, c.before, c.error)) return c;
        if (old_text.empty()) {
            c.error = "old_text is empty. Give the exact text to replace, copied from the file (read_file shows it). To "
                      "write a whole file use write_file.";
            return c;
        }
        size_t at = c.before.find(old_text), n = 0;
        for (size_t i = at; i != std::string::npos; i = c.before.find(old_text, i + old_text.size())) n++;
        if (n == 0) {
            c.error = "old_text is not in " + c.shown + ". It has to match the file exactly, spaces and line breaks "
                      "too. Read the file with read_file and copy the text from it.";
            return c;
        }
        if (n > 1) {
            c.error = "old_text is in " + c.shown + " " + std::to_string(n) + " times. Give more of the lines around "
                      "the place, so that it matches only once.";
            return c;
        }
        if (old_text == new_text) {
            c.error = "old_text and new_text are the same. Nothing to change.";
            return c;
        }
        c.after = c.before.substr(0, at) + new_text + c.before.substr(at + old_text.size());
    }
    if (c.error.empty()) measure(c);
    return c;
}

// The lines of a change as the user sees them: two as they are before it,
// the ones that go out, the ones that come in, two after it.
std::string preview_of(const Change &c) {
    const std::vector<std::string> a = lines_of(c.before), b = lines_of(c.after);
    std::string out;
    auto put = [&](const char *mark, const std::string &l) { out += std::string(mark) + cut_line(l) + "\n"; };
    for (size_t k = c.first >= 2 ? c.first - 2 : 0; k < c.first && k < a.size(); k++) put("  ", a[k]);
    for (size_t k = 0; k < c.out && k < kShownLines; k++) put("- ", a[c.first + k]);
    if (c.out > kShownLines) out += "  … and " + count(c.out - kShownLines, "more line") + " out\n";
    for (size_t k = 0; k < c.in && k < kShownLines; k++) put("+ ", b[c.first + k]);
    if (c.in > kShownLines) out += "  … and " + count(c.in - kShownLines, "more line") + " in\n";
    for (size_t k = c.first + c.out; k < a.size() && k < c.first + c.out + 2; k++) put("  ", a[k]);
    return out;
}

// What was changed last, to take back (`/undo`): the newest at the end.
struct Done {
    std::string path, before, after;
    bool existed;
};
std::mutex g_mu;
std::vector<Done> g_done;

// The file written beside itself and moved into place, so that a failure
// leaves the old one whole. It keeps the old file's permissions.
bool write_whole(const std::string &path, const std::string &content, std::string &why) {
    struct stat st;
    const bool had = stat(path.c_str(), &st) == 0;
    if (!had && !make_dirs(folder_of(path))) {
        why = std::string("its folder couldn't be made: ") + strerror(errno);
        return false;
    }
    const std::string tmp = path + ".ts5-" + std::to_string(getpid()) + ".tmp";
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, had ? (st.st_mode & 07777) : 0666);
    if (fd < 0) {
        why = strerror(errno);
        return false;
    }
    bool ok = true;
    for (size_t off = 0; ok && off < content.size();) {
        const ssize_t n = write(fd, content.data() + off, content.size() - off);
        if (n <= 0) ok = false;
        else off += (size_t)n;
    }
    if (ok && had) fchmod(fd, st.st_mode & 07777);
    if (!ok) why = strerror(errno);
    if (close(fd) != 0 && ok) ok = false, why = strerror(errno);
    if (ok && rename(tmp.c_str(), path.c_str()) != 0) ok = false, why = strerror(errno);
    if (!ok) unlink(tmp.c_str());
    return ok;
}

ToolResult failed(const std::string &head, const std::string &text) {
    ToolResult t;
    t.ok = false;
    t.head = head;
    t.text = text;
    return t;
}

ToolResult read_file(const ts5::Json &j) {
    const std::string path = whole_path(arg(j, "path")), shown = path_shown(path);
    if (path.empty()) return failed("no path given", "The call gave no path. Give the file's path.");
    std::string all, why;
    if (!read_whole(path, shown, all, why)) return failed("not read", why);
    const std::vector<std::string> lines = lines_of(all);
    size_t from = (size_t)std::max(1.0, j["start_line"].num(1));
    if (lines.empty()) {
        ToolResult t;
        t.head = "an empty file";
        t.text = "File " + shown + " is empty.";
        return t;
    }
    if (from > lines.size())
        return failed("not read", shown + " has " + count(lines.size(), "line") + ": there is no line " + std::to_string(from) + ".");
    std::string body;
    size_t to = from - 1;
    for (; to < lines.size() && to - (from - 1) < kReadLines; to++) {
        if (!body.empty() && body.size() + lines[to].size() > kReadChars) break;
        // (One line longer than a whole answer is cut.)
        body += (lines[to].size() > kReadChars ? lines[to].substr(0, kReadChars) + "… [a very long line, cut]" : lines[to]) + "\n";
    }
    ToolResult t;
    const bool part = from > 1 || to < lines.size();
    t.head = part ? fmt("lines %zu to %zu of %zu", from, to, lines.size()) : count(lines.size(), "line");
    t.text = "File " + shown + (part ? fmt(", lines %zu to %zu of %zu", from, to, lines.size()) : "") + ":\n" + body;
    if (to < lines.size()) t.text += fmt("[The file goes on. read_file with start_line %zu reads the next part.]", to + 1);
    return t;
}

ToolResult list_folder(const ts5::Json &j) {
    const std::string given = arg(j, "path");
    const std::string path = whole_path(given.empty() ? "." : given), shown = path_shown(path);
    DIR *d = opendir(path.c_str());
    if (!d)
        return failed("not listed", errno == ENOENT    ? "There is no folder " + shown + "."
                                    : errno == ENOTDIR ? shown + " is a file, not a folder. read_file reads it."
                                                       : shown + " can't be listed: " + strerror(errno) + ".");
    std::vector<std::string> dirs, files;
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        struct stat st;
        if (stat((path + "/" + n).c_str(), &st) != 0) {
            files.push_back(n);
        } else if (S_ISDIR(st.st_mode)) {
            dirs.push_back(n + "/");
        } else {
            std::string size = std::to_string((long long)st.st_size);
            for (int i = (int)size.size() - 3; i > 0; i -= 3) size.insert(i, ",");
            files.push_back(n + "  (" + size + " bytes)");
        }
    }
    closedir(d);
    std::sort(dirs.begin(), dirs.end());
    std::sort(files.begin(), files.end());
    dirs.insert(dirs.end(), files.begin(), files.end());
    ToolResult t;
    t.head = dirs.empty() ? "an empty folder" : count(dirs.size(), "name");
    t.text = "Folder " + shown + (dirs.empty() ? " is empty." : ":\n");
    const size_t most = 300;
    for (size_t k = 0; k < dirs.size() && k < most; k++) t.text += dirs[k] + "\n";
    if (dirs.size() > most) t.text += "[… and " + count(dirs.size() - most, "more name") + "]";
    return t;
}

}  // namespace

bool is_file_tool(const std::string &name) {
    return name == "read_file" || name == "list_folder" || name == "edit_file" || name == "write_file";
}

std::string files_json() {
    return tool("read_file",
                "Read a text file on the user's computer. You get its lines; a long file comes in parts.",
                {{"path", "string", "The file's path. A path that does not start with / is taken from the folder the user is working in."},
                 {"start_line?", "integer", "The line to start at, to read on in a long file."}}) +
           "," +
           tool("list_folder", "List the files and folders in a folder on the user's computer.",
                {{"path?", "string", "The folder's path. Leave it out for the folder the user is working in."}}) +
           "," +
           tool("edit_file",
                "Change part of a text file on the user's computer: one piece of its text is replaced with new text. "
                "The user is shown the change and asked before it is made. Read the file first, and copy old_text "
                "from it exactly.",
                {{"path", "string", "The file's path."},
                 {"old_text", "string",
                  "The exact text to replace, as it is in the file, with enough of the lines around it that it is "
                  "found only once."},
                 {"new_text", "string", "The text to put in its place."}}) +
           "," +
           tool("write_file",
                "Write a whole file on the user's computer: a new file, or an existing file's whole content "
                "replaced. The user is shown it and asked first. To change part of a file use edit_file.",
                {{"path", "string", "The file's path."}, {"content", "string", "The file's whole content."}});
}

bool path_in(const std::string &path, const std::string &folder) {
    if (folder.empty()) return false;
    if (folder == "/") return true;
    return path == folder || (path.size() > folder.size() && path.compare(0, folder.size(), folder) == 0 &&
                              path[folder.size()] == '/');
}

std::string path_shown(const std::string &path) {
    const std::string start = start_dir(), h = home();
    if (path == start) return ".";
    if (path_in(path, start)) return path.substr(start == "/" ? 1 : start.size() + 1);
    if (!h.empty() && path == h) return "~";
    if (!h.empty() && h != "/" && path_in(path, h)) return "~" + path.substr(h.size());
    return path;
}

FilePlan file_plan(const std::string &name, const std::string &args) {
    FilePlan p;
    ts5::Json j;
    ts5::Json::parse(args, j);
    if (name == "edit_file" || name == "write_file") {
        const Change c = change_of(name, j);
        p.change = true;
        p.path = c.path;
        p.hidden = hidden_path(c.path);
        if (!c.error.empty()) {
            p.ok = false;
            p.error = c.error;
            return p;
        }
        p.line = !c.existed ? "write a new file, " + c.shown + " (" + count(c.in, "line") + ")"
                            : (c.whole ? "replace the whole of " : "change ") + c.shown + ": " + count(c.out, "line") +
                                  " out, " + std::to_string(c.in) + " in";
        p.preview = preview_of(c);
        return p;
    }
    const std::string given = arg(j, "path");
    p.path = whole_path(name == "list_folder" && given.empty() ? "." : given);
    if (p.path.empty()) {
        p.ok = false;
        p.error = "The call gave no path. Give the file's path.";
        return p;
    }
    const std::string start = start_dir();
    const bool inside = path_in(p.path, start);
    // (Below the start folder where it is in it: started in ~/.config,
    // not everything is hidden.)
    p.hidden = hidden_path(inside ? p.path.substr(start.size()) : p.path);
    struct stat st;
    if (stat(p.path.c_str(), &st) != 0) {
        p.ok = false;  // the call itself says what is wrong
        return p;
    }
    // "Every file in this folder": the start folder for a path in it,
    // else the file's own folder (a folder's own self, when it is listed).
    p.folder = inside ? start : name == "list_folder" ? p.path : folder_of(p.path);
    p.line = std::string(name == "list_folder" ? "list the folder " : "read ") + path_shown(p.path) +
             (p.hidden ? "  (hidden)" : inside ? "" : "  (outside the folder ts5-tui was started in)");
    return p;
}

std::string file_call_line(const std::string &name, const std::string &args) {
    ts5::Json j;
    ts5::Json::parse(args, j);
    const std::string given = arg(j, "path");
    const std::string shown = path_shown(whole_path(name == "list_folder" && given.empty() ? "." : given));
    if (name == "read_file") {
        const int from = (int)j["start_line"].num(1);
        return "reads " + shown + (from > 1 ? fmt(" from line %d", from) : "");
    }
    if (name == "list_folder") return "lists the folder " + shown;
    return (name == "edit_file" ? "changes " : "writes ") + shown;
}

ToolResult file_call(const std::string &name, const std::string &args) {
    ts5::Json j;
    ts5::Json::parse(args, j);
    if (name == "read_file") return read_file(j);
    if (name == "list_folder") return list_folder(j);
    const Change c = change_of(name, j);
    if (!c.error.empty()) return failed("not changed", c.error);
    std::string why;
    if (!write_whole(c.path, c.after, why))
        return failed("not changed: " + why, c.shown + " could not be written: " + why + ". Nothing was changed.");
    {
        std::lock_guard<std::mutex> l(g_mu);
        g_done.push_back({c.path, c.before, c.after, c.existed});
        if (g_done.size() > 20) g_done.erase(g_done.begin());
    }
    ToolResult t;
    const std::vector<std::string> b = lines_of(c.after);
    t.head = !c.existed ? "written: " + count(b.size(), "line") : "changed: " + count(c.out, "line") + " out, " + std::to_string(c.in) + " in";
    t.text = (!c.existed ? "Written: " + c.shown + ", " + count(b.size(), "line") + "."
                         : "Changed " + c.shown + " (" + count(c.out, "line") + " out, " + std::to_string(c.in) + " in). It has " +
                               count(b.size(), "line") + " now.");
    if (c.existed && c.in > 0) {
        // The place as it reads now, for the model to check its work.
        t.text += fmt(" From line %zu it reads:\n", (c.first >= 2 ? c.first - 2 : 0) + 1);
        for (size_t k = c.first >= 2 ? c.first - 2 : 0; k < b.size() && k < c.first + std::min(c.in, kShownLines) + 2; k++)
            t.text += cut_line(b[k]) + "\n";
    }
    return t;
}

std::string file_undo() {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_done.empty()) return "There is no change by a file tool to take back in this run of ts5-tui.";
    const Done d = g_done.back();
    const std::string shown = path_shown(d.path);
    std::string now, why;
    struct stat st;
    const bool there = stat(d.path.c_str(), &st) == 0;
    if (!there || !read_whole(d.path, shown, now, why) || now != d.after)
        return shown + " is not as the tool left it (it was changed again, or is gone), so it was left alone.";
    g_done.pop_back();
    if (!d.existed) {
        if (unlink(d.path.c_str()) != 0) return shown + " could not be removed: " + strerror(errno) + ".";
        return "Taken back: " + shown + " was new, and is gone again.";
    }
    if (!write_whole(d.path, d.before, why)) return shown + " could not be put back: " + why + ".";
    return "Taken back: " + shown + " is as it was before the change.";
}

}  // namespace tui
