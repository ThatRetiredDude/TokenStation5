#!/usr/bin/env python3
"""ts5-tui's outside tools (linux/tui/mcp.h) against fake-mcp.py, with no
model and no screen: `ts5-tui --mcp`, `--mcp-tools` and `--mcp-call`.

    linux/tui/test/mcp_test.py PATH/TO/ts5-tui
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake-mcp.py")
failed = 0


def check(ok, what, got=None):
    global failed
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failed += 1
        if got is not None:
            print("      " + str(got)[:1500].replace("\n", "\n      "))


class Home:
    """A home folder with an mcp.json in it."""

    def __init__(self, tui):
        self.tui = tui
        self.dir = tempfile.mkdtemp()
        self.conf = os.path.join(self.dir, ".config", "tokenstation")
        os.makedirs(self.conf)
        self.log = os.path.join(self.dir, "fake.log")

    def servers(self, servers, key="mcpServers"):
        with open(os.path.join(self.conf, "mcp.json"), "w") as f:
            json.dump({key: servers}, f)
        if os.path.exists(self.log):
            os.remove(self.log)

    def fake(self, *flags, **more):
        return {"command": sys.executable, "args": [FAKE, "--log", self.log, *flags], **more}

    def run(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith("XDG_") and not k.startswith("TS5_")}
        e.update(HOME=self.dir, **(env or {}))
        r = subprocess.run([self.tui, *args], env=e, capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout, r.stderr

    def logged(self):
        return open(self.log).read() if os.path.exists(self.log) else ""

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def main():
    h = Home(sys.argv[1])
    try:
        code, out, err = h.run("--mcp")
        check(code == 0 and "No MCP servers are set up" in out, "no mcp.json: nothing is set up, and that is no error", (code, out, err))

        with open(os.path.join(h.conf, "mcp.json"), "w") as f:
            f.write('{"mcpServers": {"fake": ')
        code, out, err = h.run("--mcp")
        check(code == 2 and "is not valid JSON" in err, "an mcp.json that is not JSON is said so", (code, out, err))

        # What a server offers, and under which names.
        h.servers({"fake": h.fake("--extra", "odd_name with.space", env={"FAKE_KEY": "abc"})})
        code, out, err = h.run("--mcp")
        check(code == 0 and "13 tools" in out and "asks first" in out, "a server starts and its tools are counted", (code, out, err))
        check("\n    fake_echo  Gives back the text it is given." in out, "a tool is called SERVER_TOOL, with its description", out)
        check("fake_odd_name_with_space  A name with a dot and spaces.\n" in out,
              "a name the model can't be given is made one it can", out)
        check("fake_odd_name_with_space  One more tool, called odd_name with.space.  [left out: another of its tools has this name]" in out,
              "two tools that come to the same name: the second is left out, and said so", out)
        check("fake_struct  Structured only" in out, "a tool without a description is shown by its title", out)
        check("FAKE_KEY" not in out and "abc" not in out, "a server's environment is never printed", out)
        check(h.logged().split("\n")[:3] == ["initialize", "notifications/initialized", "tools/list"] and "end of input" in h.logged(),
              "the server is greeted as the protocol has it, and ends when its input closes", h.logged())

        # The tools as a request carries them.
        code, out, err = h.run("--mcp-tools")
        tools = {t["function"]["name"]: t["function"] for t in json.loads(out)}
        check(code == 0 and len(tools) == 13 and "fake_echo" in tools, "the request gets every tool once", (code, sorted(tools)))
        nested = tools["fake_nested"]["parameters"]
        text = json.dumps(nested)
        check("$schema" not in text and "$id" not in text and "additionalProperties" not in text and "NestedArguments" not in text
              and '"City"' not in text and '"Tag"' not in text and '"Day"' not in text,
              "a schema loses what only costs context ($schema, titles, additionalProperties)", text)
        check(nested["properties"]["title"] == {"type": "string", "description": "A property named title."} and
              nested["required"] == ["title"] and nested["properties"]["where"]["properties"]["city"] == {"type": "string"},
              "and keeps the rest: a property named title, what is required, the nesting", text)
        big = tools["fake_big"]["description"]
        check(300 < len(big) <= 600 and big.endswith("to be cut."), "a long description is cut at a sentence's end", (len(big), big[-60:]))

        # Calls.
        code, out, err = h.run("--mcp-call", "fake_echo", '{"text": "hello there"}')
        check(code == 0 and out == "echo: hello there\n", "a call's text comes back", (code, out, err))
        code, out, err = h.run("--mcp-call", "fake_add", '{"a": 2, "b": 40}', "--mcp-call", "fake_struct",
                               "--mcp-call", "fake_picture", "--mcp-call", "fake_ping_back")
        check(code == 0 and out.startswith("42\n"), "text wins over structured content", out)
        check('"rows": [' in out, "structured content alone is given as JSON", out)
        check("[a picture (image/png): it can't be shown here]\na caption\n" in out, "a picture is named, its caption kept", out)
        check("the client answered the ping with {} and the folders with -32601" in out,
              "a server's ping is answered; a request ts5-tui doesn't serve is refused, and the call goes on", out)
        code, out, err = h.run("--mcp-call", "fake_fail")
        check(code == 1 and out.startswith("The tool reported an error:\nit broke: the disk is full"), "a tool's own error is told as one", (code, out, err))
        code, out, err = h.run("--mcp-call", "fake_refuse")
        check(code == 1 and "The call was not accepted: Unknown tool: refuse" in out, "the server's refusal is told with its words", (code, out, err))
        code, out, err = h.run("--mcp-call", "fake_echo", "not json")
        check(code == 1 and "were not a JSON object" in out and "tools/call" not in h.logged().split("tools/list")[-1],
              "arguments that are no JSON never reach the server", (code, out, h.logged()))
        code, out, err = h.run("--mcp-call", "fake_big", '{"n": 2000}')
        check(code == 0 and len(out) < 8600 and "characters left out here" in out and out.startswith("line 1 of") and
              "line 2000 of the big answer" in out, "a long result keeps its start and its end", (code, len(out), out[:200]))
        code, out, err = h.run("--mcp-call", "fake_nothere")
        check(code == 2 and "no MCP tool called fake_nothere" in err, "a tool that is not there is said so", (code, out, err))
        code, out, err = h.run("--mcp-call", "fake_grow", "--mcp-call", "fake_late")
        check(code == 0 and "late was called with {}" in out, "a server that says its tools changed is asked again", (code, out, err))

        # The environment: ts5's own keys stay here, the server's own are set.
        env = {"TS5_API_KEY": "secret-1", "TAVILY_API_KEY": "secret-2", "OTHER_THING": "kept"}
        code, out, err = h.run("--mcp-call", "fake_env", '{"name": "TS5_API_KEY"}', "--mcp-call", "fake_env", '{"name": "TAVILY_API_KEY"}',
                               "--mcp-call", "fake_env", '{"name": "OTHER_THING"}', "--mcp-call", "fake_env", '{"name": "FAKE_KEY"}', env=env)
        check(out == "(not set)\n(not set)\nkept\nabc\n", "the server gets the environment without ts5's keys, and its own on top", (code, out, err))
        h.servers({"fake": h.fake(env={"TAVILY_API_KEY": "its-own"})})
        code, out, err = h.run("--mcp-call", "fake_env", '{"name": "TAVILY_API_KEY"}', env=env)
        check(out == "its-own\n", "a key mcp.json gives the server itself is its to have", (code, out, err))

        # Servers that are slow, careless or gone.
        h.servers({"fake": h.fake("--banner", "--pages")})
        code, out, err = h.run("--mcp")
        check(code == 0 and "13 tools" in out and "fake_odd_name_with_space" in out,
              "a line that is no message is passed over; tools that come in pages are all read", (code, out, err))
        h.servers({"fake": h.fake()})
        code, out, err = h.run("--mcp-call", "fake_slow", '{"seconds": 5}', env={"TS5_MCP_CALL_SECONDS": "1"})
        check(code == 1 and "gave no answer in 1 seconds" in out, "a call that takes too long is given up", (code, out, err))
        h.servers({"fake": h.fake("--die-after", "1")})
        code, out, err = h.run("--mcp-call", "fake_echo", '{"text": "one"}', "--mcp-call", "fake_echo", '{"text": "two"}')
        check(code == 0 and out == "echo: one\necho: two\n" and h.logged().count("initialize\n") == 2,
              "a server that left since (exit code 0) is started again for the next call", (code, out, err, h.logged()))
        h.servers({"fake": h.fake("--die-in-call")})
        code, out, err = h.run("--mcp-call", "fake_echo", '{"text": "one"}')
        check(code == 1 and "ended while it worked on the call" in out and "out of memory, giving up" in out and "not known" in out,
              "a server that ends in a call: said so, with its last words", (code, out, err))
        h.servers({"fake": h.fake("--hang")})
        code, out, err = h.run("--mcp", env={"TS5_MCP_START_SECONDS": "1"})
        check(code == 1 and "not running: it said nothing in 1 seconds" in out, "a server that never answers is given up", (code, out, err))
        h.servers({"fake": h.fake("--stderr", "boom: no license")})
        code, out, err = h.run("--mcp")
        check(code == 1 and "not running: the program ended as it started (boom: no license)" in out,
              "a server that ends as it starts: its last words say why", (code, out, err))
        h.servers({"gone": {"command": "no-such-program-here", "args": []}, "web": {"url": "https://example.com/mcp"},
                   "bare": {}, "fake": h.fake()})
        code, out, err = h.run("--mcp")
        check(code == 1 and "there is no program called no-such-program-here on this machine" in out, "a program that is not installed is said so", out)
        check("it is reached by a web address" in out, "a server at a web address is listed as not spoken to yet", out)
        check('gives it no "command"' in out, "an entry without a command is said so", out)
        check("13 tools" in out, "and the others still run", out)

        # ts5-tui's own keys in mcp.json: off, never asks, only.
        h.servers({"fake": h.fake(disabled=True)})
        code, out, err = h.run("--mcp")
        check(code == 0 and "\n  off\n" in out and h.logged() == "", "a server that is off is not started", (code, out, h.logged()))
        code, out, err = h.run("--mcp-call", "fake_echo", '{"text": "x"}')
        check(code == 2, "and its tools are not there to call", (code, out, err))
        h.servers({"fake": h.fake(ask=False, only=["echo", "fake_add"])})
        code, out, err = h.run("--mcp")
        check(code == 0 and "2 tools" in out and "never asks" in out and 'fake_fail  Always fails, as a tool that hit an error.  [left out by "only"]' in out,
              '"only" leaves the model just the tools it names; "ask": false is shown', (code, out, err))
        code, out, err = h.run("--mcp-tools")
        check(sorted(t["function"]["name"] for t in json.loads(out)) == ["fake_add", "fake_echo"], "and the request carries only them", out)
        code, out, err = h.run("--mcp-call", "fake_fail")
        check(code == 1 and "switched this tool off" in out and "tools/call" not in h.logged(), "a tool left out can't be called", (code, out, h.logged()))

        # Names: a server whose tool would take one of ts5-tui's own names; the other layout's key.
        h.servers({"read": h.fake("--extra", "file")}, key="servers")
        code, out, err = h.run("--mcp")
        check(code == 0 and "read_file  One more tool, called file.  [left out: a tool of ts5-tui's own has this name]" in out and "read_echo" in out,
              'a tool that would take the name of one of ts5-tui\'s own is left out ("servers" is read too)', (code, out, err))
    finally:
        h.close()
    if failed:
        sys.exit(f"mcp_test: {failed} failed")
    print("mcp_test: all passed")


if __name__ == "__main__":
    main()
