// Outside tools: programs that speak MCP (the Model Context Protocol), which
// ts5-tui starts and whose tools the model can then call like its own. They
// are listed in mcp.json beside tui.conf, in the layout other MCP clients
// read too:
//
//   {"mcpServers": {"notes": {"command": "notes-server", "args": ["--dir", "/home/me/notes"],
//                             "env": {"NOTES_KEY": "..."}}}}
//
// with three keys of ts5-tui's own beside them: "disabled": true (off),
// "ask": false (its calls go without a question) and "only": ["tool", ...]
// (the model gets just these). A program is started when ts5-tui starts,
// spoken to over its standard input and output, a line a message, and ended
// when ts5-tui quits. Every call is asked first ("Run it", "Every call to
// NAME in this chat", "No") unless the server was set to never ask.
//
// Only programs on this machine ("command"); a "url" server is listed as
// not spoken to yet.
#pragma once
#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include "tools.h"

namespace tui {

struct McpTool {
    std::string name;   // the server's own name for it
    std::string shown;  // as the model calls it: SERVER_NAME
    std::string what;   // its description, as the model gets it
    bool on = true;     // false: left out by "only"
    std::string left;   // why the model doesn't get it ("a tool of ts5-tui's own has this name"), or ""
};

struct McpServer {
    std::string name;
    std::string command;  // the program and its arguments, for the screen (never the env)
    int level = 1;        // 0 off, 1 asks first, 2 never asks
    int state = 0;        // 0 not started, 1 starting, 2 ready, 3 failed
    std::string why;      // failed: why
    std::vector<McpTool> tools;
    size_t chars = 0;     // what its tools add to every request
};

// mcp.json's place.
std::string mcp_path();
// Reads mcp.json and brings the running programs in line with it: starts
// the ones that are new or changed (in the background), ends the ones that
// are gone or off. run false: nothing runs (all tools are off). Returns ""
// or what is wrong with the file.
std::string mcp_load(bool run);
// As they stand now.
std::vector<McpServer> mcp_servers();
// Waits until no server is still starting, at most `seconds`.
void mcp_wait(double seconds);
// Ends every program (ts5-tui quits).
void mcp_stop_all();

// Changes to mcp.json, each followed by mcp_load. false: `why` says what
// went wrong.
bool mcp_add(const std::string &name, const std::string &line, std::string &why);  // line: [KEY=VALUE ...] PROGRAM [ARGUMENTS]
bool mcp_remove(const std::string &name, std::string &why);
bool mcp_set_level(const std::string &name, int level, std::string &why);
// The model gets only these of the server's tools (their own names or as the model calls them); none: all of them.
bool mcp_set_only(const std::string &name, const std::vector<std::string> &tools, std::string &why);
bool mcp_restart(const std::string &name, std::string &why);

// The tools of the servers that are ready and not off, as the request
// names them (JSON objects, comma-joined), or "".
std::string mcp_tools_json();
// Whether the model's name for a tool is one of a server's.
bool is_mcp_tool(const std::string &shown);
// The server a tool is of, and whether its calls are asked first.
std::string mcp_server_of(const std::string &shown);
bool mcp_asks(const std::string &shown);
// One line for the screen ("notes: search {"query": "tax"}"), and the
// arguments a line each, for the question.
std::string mcp_call_line(const std::string &shown, const std::string &args);
std::string mcp_call_preview(const std::string &args);
ToolResult mcp_call(const std::string &shown, const std::string &args, const std::atomic<bool> &cancel);

// `ts5-tui --mcp`: starts the servers, says what each one offers, and ends
// them (json: the tools as a request carries them, nothing else). `ts5-tui
// --mcp-call TOOL JSON`: a call, its result printed; several are made one
// after the other, by the same programs. The exit code is 0 when all is well.
int mcp_check_main(bool json);
int mcp_call_main(const std::vector<std::pair<std::string, std::string>> &calls);

}  // namespace tui
