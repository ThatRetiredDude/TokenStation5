// The title's settings file, ts5.env in its folder: NAME=VALUE words, which
// the title puts in its environment at start (title.cpp). The Settings page
// changes one setting at a time through set_setting, which rewrites the file
// keeping everything else in it (other settings, comments, keys) and puts the
// new value in the environment, where whatever reads it next finds it.
#pragma once
#include <string>
#include <vector>

namespace app {

// One line of a command file as words: whitespace separates them, double
// quotes group them. A line starting with # has none.
std::vector<std::string> line_words(const std::string &line);

// Sets NAME=VALUE in the file and the environment; an empty value removes the
// setting from both. False, with err, if the file can't be written.
bool set_setting(const std::string &path, const std::string &name, const std::string &value, std::string &err);

// NAME's value now (the environment), or def.
std::string setting(const char *name, const std::string &def = "");

// Whether NAME holds a secret (..._KEY, ..._TOKEN), which stays out of the
// log and off the screen.
bool secret_setting(const std::string &name);

}  // namespace app
