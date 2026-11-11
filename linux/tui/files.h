// The model's file tools in ts5-tui: read_file, list_folder, edit_file and
// write_file. A change is never made unseen: the user gets the lines that
// go out and the lines that come in, and says yes or no (chat.cpp asks);
// `/undo` takes the last change back. Reading is asked too, until the user
// says "every file in this folder, this chat"; a hidden file (a name that
// starts with a dot) is asked for every time.
//
// A relative path starts at the folder ts5-tui was started in, the same
// one the terminal tool's commands run in.
#pragma once
#include <string>
#include <vector>

#include "tools.h"

namespace tui {

bool is_file_tool(const std::string &name);

// The four tools as the request names them (JSON objects, comma-joined).
std::string files_json();

// What a call would do, worked out without doing it.
struct FilePlan {
    bool ok = true;       // false: it can't be done, and `error` tells the model why
    bool change = false;  // edit_file, write_file
    bool hidden = false;  // a part of the path starts with a dot: asked for every time
    std::string path;     // the whole path
    std::string line;     // in words, for the question: "change notes/plan.md: 2 lines out, 3 in"
    std::string preview;  // a change's lines: "  " as they are, "- " out, "+ " in
    std::string folder;   // a read: the folder a "every file in this folder" would cover
    std::string error;
};
FilePlan file_plan(const std::string &name, const std::string &args);
// Whether `path` is in `folder` or under it.
bool path_in(const std::string &path, const std::string &folder);
// A path as the user reads it: from the start folder where it is inside
// it, with ~ for the home folder, else whole.
std::string path_shown(const std::string &path);

// One line for the screen: "reads notes/plan.md", "changes src/main.c".
std::string file_call_line(const std::string &name, const std::string &args);
ToolResult file_call(const std::string &name, const std::string &args);

// Takes back the last change a file tool made in this run of ts5-tui, if
// the file is still as the tool left it. Says what happened.
std::string file_undo();

}  // namespace tui
