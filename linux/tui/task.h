// Unattended tasks: a job the model works on for hours with nobody there to
// answer, in the sandbox only, in a program of its own (`ts5-tui
// --task-run ID`, started in its own session), so it keeps going when
// ts5-tui closes, when the phone goes away and when the server starts again
// for another model (it waits for one and goes on).
//
// Each task is a folder, <data>/tasks/<id>/, kept for good (no cleanup of
// ts5's removes it):
//   task.json     the goal, the folder it may change, its limits, where it
//                 stands and how it ended
//   log.jsonl     every model turn (its thinking, words and calls), every
//                 result the model got and every set of notes, a line each,
//                 with the time
//   commands/     every command's whole output, N.txt, beside the shortened
//                 one the model got
//   runner.log    the program's own lines
// The task's chat is saved in the chat list too, so ts5-tui opens it.
//
// Nobody to ask: a command runs in the sandbox without asking (the sandbox
// is required: without it no task starts); anything that would reach out
// of it is refused, with a plain word to the model: a button or typing on a
// web page, a change on GitHub, the file tools (off: the model changes
// files with commands, in the sandbox), outside tools. Reading the web
// stays on. It ends when the model answers (done), when its work goes in
// circles or keeps repeating (stuck), when a limit is reached (steps, hours,
// tokens the model writes), when it is stopped, or when it fails (no
// server for 10 minutes).
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "chat.h"

namespace tui {

struct TaskLimits {
    int steps = 200;        // turns that call tools
    double hours = 4;       // from its start
    long long tokens = 0;   // that the model writes, thinking too (0: no limit)
};

struct TaskInfo {
    std::string id, goal, folder;
    TaskLimits limits;
    // starting, waiting (for the server), running, then how it ended: done,
    // stuck, steps, hours, tokens, stopped, failed.
    std::string state = "starting";
    std::string why;      // how it ended, in words, or what it waits for
    std::string answer;   // the model's last answer
    std::string model, chat_id;
    double created = 0, started = 0, ended = 0, updated = 0;
    int pid = 0, steps = 0, commands = 0, refused = 0, restarts = 0;
    long long tokens_out = 0, tokens_in = 0;
    bool ended_state() const { return state != "starting" && state != "waiting" && state != "running"; }
};

std::string tasks_dir();
std::string task_dir(const std::string &id);
bool task_read(const std::string &id, TaskInfo &t);
// Newest first.
std::vector<TaskInfo> task_list();
// The task whose program still runs, if one does.
bool task_running(TaskInfo *t = nullptr);
// "done", "stuck" ... in words: "done", "stuck: going in circles" ...
std::string task_state_words(const TaskInfo &t);
// One line on where it stands: "running · step 12 of 200 · 0.4 of 4 hours · 31,200 tokens".
std::string task_progress(const TaskInfo &t);
// Makes the task and starts its program (self: this program's path), in
// its own session, in folder. Its id, or "" and why (one runs already, no
// sandbox, no goal).
std::string task_start(const std::string &self, const std::string &goal, const std::string &folder,
                       const TaskLimits &lim, std::string &why);
// Asks its program to stop (it ends the model's turn and its command, and
// says so). False with why.
bool task_stop(const std::string &id, std::string &why);
// The last n lines of its log, as one line each for a screen: "12:04 $ ls".
std::vector<std::string> task_tail(const std::string &id, size_t n);

// The program's side (main.cpp's --task-run). request: the request as
// ts5-tui would make it now (it polls the server first; ok false while no
// model is loaded). ended: called once it has ended, with how.
struct TaskHooks {
    std::function<ChatRequest(bool &ok, std::string &why)> request;
    std::function<void(const TaskInfo &)> ended;
};
int task_run_main(const std::string &id, const TaskHooks &h);

}  // namespace tui
