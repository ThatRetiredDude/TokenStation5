// The tools ts5-tui gives the model, each with its own switch (Settings):
// the web search (the server's own, with its Tavily key), a web page read
// as text, a command run in a shell on this machine (asked first), notes
// that are kept between chats, and a browser the model works by the numbers
// on the page (links freely, buttons and typing asked first), files it
// reads and changes (files.h), and the tools of outside programs that
// speak MCP (mcp.h). The model calls them the way any OpenAI
// client's tools are called ("tools" in the request, "tool_calls" back);
// ts5-tui runs the call and sends the result in a "tool" message.
#pragma once
#include <atomic>
#include <string>
#include <vector>

namespace tui {

struct ToolSwitches {
    bool web = true;       // the server's web search (it needs a Tavily key)
    bool page = true;      // read_web_page
    bool terminal = true;  // run_command, asked first
    // run_command's commands go through the sandbox (sandbox.h): it was
    // chosen and it runs here. sandbox_net: with the network.
    bool sandbox = false, sandbox_net = true;
    bool memory = true;    // remember, and the notes at the top of every chat
    bool browser = false;  // browser_open and the rest (`ts5 browser`, engine/src/browser.h)
    // Settings -> Browser: "auto", "text", "obscura", "chromium", or a program's path.
    std::string browser_program = "auto";
    // A command, and a button or typing on a page, wait for the user's
    // yes; false: they go without asking (the user's choice in Settings).
    bool terminal_ask = true, browser_ask = true;
    // read_file, list_folder, edit_file, write_file (files.h). files_ask
    // false: they read and change without asking.
    bool files = false, files_ask = true;
    // The outside tools (mcp.h): the programs in mcp.json that are not
    // switched off there. Each has its own "asks first".
    bool mcp = false;
    // GitHub's REST API as the user signed in on the Connectors page
    // (`ts5 github`, engine/src/github.h). github_ask: a change (anything
    // but GET) waits for the user's yes.
    bool github = false, github_ask = true;
    bool any() const { return page || terminal || memory || browser || files || mcp || github; }  // tools ts5-tui runs
};

// The request's "tools" for the switches that are on: a JSON array, or "".
std::string tools_json(const ToolSwitches &s);

// The notes (memory.md in the data folder, a line each).
std::string memory_path();
std::vector<std::string> memory_notes();
bool memory_add(const std::string &note, std::string &why);
// Note n, counted from 1 as /memory lists them.
bool memory_forget(size_t n, std::string &why);
bool memory_clear(std::string &why);
// The notes as the model reads them at the top of a chat; "" without any.
std::string memory_prompt();

struct ToolResult {
    std::string text;  // what the model gets back
    std::string head;  // one line for the screen
    bool ok = true;
    bool refused = false;  // the user said no to it
};

// One line for the screen saying what a call does: "$ ls -la", "reads
// https://…", "remembers: …".
std::string tool_call_line(const std::string &name, const std::string &args);
// A run_command call's command, or "".
std::string tool_command(const std::string &name, const std::string &args);
// What a browser call would do that the user is asked about first, in words
// ("press the button "Buy" on shop.example"); "" for a call that needs no
// asking (reading a page, following a link) and for the other tools.
std::string browser_question(const std::string &name, const std::string &args, const ToolSwitches &s,
                             const std::atomic<bool> &cancel);
// Ends the browser and the page it holds (the tool was switched off, a new
// chat, ts5-tui quits). It starts again with the next call.
void browser_stop();
// The browsers installed here that the tool can use, by name ("Chromium",
// "Obscura"), the one Auto takes first; empty: pages are read by the text view.
std::vector<std::string> browsers_found();
// Runs a call (its arguments as JSON text). cancel: set from another thread,
// it stops a command that is running.
ToolResult run_tool(const std::string &name, const std::string &args, const ToolSwitches &s,
                    const std::atomic<bool> &cancel);
// What the model is told about a command the user refused, and about a call
// that didn't run because the answer was stopped.
ToolResult tool_refused();
ToolResult tool_stopped();
// One of the browser's tools (browser_open, browser_click, ...).
bool is_browser_tool(const std::string &name);

// The GitHub connector's token file (`ts5 github` keeps it; this only looks):
// TS5_GITHUB_FILE, else connectors/github.json in the data folder.
std::string github_file();
bool github_signed_in();
// A github call's method, in capitals ("GET" when it names none), and
// whether it changes something on GitHub (anything but GET).
std::string github_method(const std::string &args);
// What a github call would send, a line each: its body, laid out.
std::string github_preview(const std::string &args);
// A file written on GitHub: how much of it changes ("Replaces README.md on main: 60 of its 61 lines go,
// 1 comes in"), from the file as it is now. Empty for any other call.
std::string github_change_note(const std::string &args, const std::atomic<bool> &cancel);

}  // namespace tui
