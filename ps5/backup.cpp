#include "backup.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <set>

#include "log.h"

namespace app {

namespace {

bool read_file(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    const bool ok = !ferror(f);
    fclose(f);
    return ok;
}

// Written next to it, then moved over it, so a copy cut short by the title
// closing never stands in for the file.
bool copy_file(const std::string &from, const std::string &to, std::string &err) {
    std::string data;
    if (!read_file(from, data)) {
        err = "can't read " + from + ": " + strerror(errno);
        return false;
    }
    const std::string tmp = to + ".new";
    FILE *f = fopen(tmp.c_str(), "wb");
    bool ok = f && fwrite(data.data(), 1, data.size(), f) == data.size();
    if (f) ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), to.c_str()) != 0) {
        err = "can't write " + to + ": " + strerror(errno);
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

bool is_file(const std::string &path, struct stat &st) { return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode); }

bool ends_with(const std::string &s, const char *end) {
    const size_t n = strlen(end);
    return s.size() >= n && s.compare(s.size() - n, n, end) == 0;
}

// The files in dir, without hidden ones and the ones being written (.new,
// .tmp), which are moved over the real one when they're done.
std::set<std::string> names_in(const std::string &dir) {
    std::set<std::string> names;
    DIR *d = opendir(dir.c_str());
    if (!d) return names;
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.empty() || n[0] == '.' || ends_with(n, ".new") || ends_with(n, ".tmp")) continue;
        names.insert(n);
    }
    closedir(d);
    return names;
}

}  // namespace

Backup::Backup(std::string home, std::string store, std::vector<std::string> files, std::vector<std::string> dirs)
    : home_(std::move(home)), store_(std::move(store)), files_(std::move(files)), dirs_(std::move(dirs)) {}

int Backup::restore() {
    std::lock_guard<std::mutex> lk(mu_);
    mkdir(store_.c_str(), 0777);
    std::vector<std::string> missing;
    struct stat st;
    for (const std::string &f : files_)
        if (!is_file(home_ + "/" + f, st) && is_file(store_ + "/" + f, st)) missing.push_back(f);
    for (const std::string &d : dirs_) {
        mkdir((home_ + "/" + d).c_str(), 0777);
        for (const std::string &n : names_in(store_ + "/" + d)) {
            const std::string rel = d + "/" + n;
            if (!is_file(home_ + "/" + rel, st) && is_file(store_ + "/" + rel, st)) missing.push_back(rel);
        }
    }
    int copied = 0;
    std::string err;
    for (const std::string &rel : missing) {
        if (copy_file(store_ + "/" + rel, home_ + "/" + rel, err)) copied++;
        else ts5::log_line("ts5-title: backup: %s", err.c_str());
    }
    if (!missing.empty())
        ts5::log_line("ts5-title: backup: put back %d of %zu files from %s", copied, missing.size(), store_.c_str());
    return copied;
}

void Backup::sync_file(const std::string &rel) {
    const std::string from = home_ + "/" + rel, to = store_ + "/" + rel;
    struct stat st, copy;
    if (!is_file(from, st)) {
        // Gone from home since it was copied this run (a chat removed): from
        // the copy too, or restore() would bring it back. One home never had
        // this run stays, in case restore() couldn't write it.
        if (seen_.erase(rel)) unlink(to.c_str());
        return;
    }
    const Seen now{(int64_t)st.st_size, (int64_t)st.st_mtime};
    auto it = seen_.find(rel);
    if (it != seen_.end() && it->second.size == now.size && it->second.mtime == now.mtime) return;
    if (it == seen_.end() && is_file(to, copy) && copy.st_size == st.st_size) {
        // The first look this run: a copy that's already the same stays.
        std::string a, b;
        if (read_file(from, a) && read_file(to, b) && a == b) {
            seen_[rel] = now;
            return;
        }
    }
    std::string err;
    if (copy_file(from, to, err)) {
        seen_[rel] = now;
    } else if (!failed_) {
        failed_ = true;
        ts5::log_line("ts5-title: backup: %s (said once)", err.c_str());
    }
}

void Backup::sync() {
    std::lock_guard<std::mutex> lk(mu_);
    mkdir(store_.c_str(), 0777);
    for (const std::string &f : files_) sync_file(f);
    for (const std::string &d : dirs_) {
        mkdir((store_ + "/" + d).c_str(), 0777);
        std::set<std::string> names = names_in(home_ + "/" + d);
        for (const std::string &n : names_in(store_ + "/" + d)) names.insert(n);
        for (const std::string &n : names) sync_file(d + "/" + n);
    }
}

}  // namespace app
