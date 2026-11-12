#!/usr/bin/env python3
"""A small MCP server for trying ts5-tui's outside tools (linux/tui/mcp.h)
without installing one: it speaks the protocol over its standard input and
output, a line a message, and offers a few tools that show what a client
has to cope with.

    linux/tui/test/fake-mcp.py [--banner] [--pages] [--extra NAME]... [--log FILE]
                               [--hang] [--stderr TEXT] [--die-after N] [--die-in-call]

Tools: echo (text), add (a, b; the sum also as structured content), big (n
lines), fail (the tool's own error), refuse (the protocol's error), slow
(seconds), ping_back (asks the client a ping and its folders, says what came
back), picture (an image and a caption), env (name: that variable here),
struct (structured content only), grow (a tool "late" appears, and the
client is told the list changed), nested (a schema with everything in it),
"odd.name with space" (a name the model can't be given as it is).
--banner: a line that is no message first, as a careless server prints.
--pages: the tools two at a time. --extra NAME: one more tool of that name.
--log FILE: each message's method, a line each. --hang: never answers.
--stderr TEXT: says TEXT on standard error and ends. --die-after N: ends
after its Nth call is answered. --die-in-call: ends in a call, unanswered.
"""
import json
import os
import sys
import time

args = sys.argv[1:]


def flag(name):
    if name in args:
        args.remove(name)
        return True
    return False


def value(name, default=None):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


BANNER, PAGES, HANG, DIE_IN_CALL = flag("--banner"), flag("--pages"), flag("--hang"), flag("--die-in-call")
LOG, STDERR, DIE_AFTER = value("--log"), value("--stderr"), int(value("--die-after", "0"))
EXTRA = []
while "--extra" in args:
    EXTRA.append(value("--extra"))


def schema(props, required=()):
    return {"type": "object", "properties": props, "required": list(required)}


TOOLS = [
    {"name": "echo", "description": "Gives back the text it is given.",
     "inputSchema": schema({"text": {"type": "string", "description": "The text."}}, ["text"])},
    {"name": "add", "description": "Adds two numbers.",
     "inputSchema": schema({"a": {"type": "number"}, "b": {"type": "number"}}, ["a", "b"])},
    {"name": "big", "description": "Gives back n numbered lines. " + "It is a long description, to be cut. " * 40,
     "inputSchema": schema({"n": {"type": "integer"}}, ["n"])},
    {"name": "fail", "description": "Always fails, as a tool that hit an error.", "inputSchema": schema({})},
    {"name": "refuse", "description": "Is refused by the server itself.", "inputSchema": schema({})},
    {"name": "slow", "description": "Answers after so many seconds.",
     "inputSchema": schema({"seconds": {"type": "number"}}, ["seconds"])},
    {"name": "ping_back", "description": "Asks the client a ping and its folders first.", "inputSchema": schema({})},
    {"name": "picture", "description": "Gives back a picture and a caption.", "inputSchema": schema({})},
    {"name": "env", "description": "Says what an environment variable is here.",
     "inputSchema": schema({"name": {"type": "string"}}, ["name"])},
    {"name": "struct", "title": "Structured only", "inputSchema": schema({})},
    {"name": "grow", "description": "Makes one more tool appear.", "inputSchema": schema({})},
    {"name": "nested", "description": "A schema with everything in it.",
     "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "$id": "nested", "title": "NestedArguments",
                     "type": "object", "additionalProperties": False,
                     "properties": {"title": {"type": "string", "title": "Title", "description": "A property named title."},
                                    "where": {"type": "object", "title": "Where", "additionalProperties": False,
                                              "properties": {"city": {"type": "string", "title": "City"}}},
                                    "tags": {"type": "array", "items": {"type": "string", "title": "Tag"}},
                                    "when": {"anyOf": [{"type": "string", "title": "Day"}, {"type": "null"}]}},
                     "required": ["title"]}},
    {"name": "odd.name with space", "description": "A name with a dot and spaces.", "inputSchema": schema({})},
] + [{"name": n, "description": "One more tool, called %s." % n, "inputSchema": schema({})} for n in EXTRA]

calls = 0
next_id = 0


def log(text):
    if LOG:
        with open(LOG, "a") as f:
            f.write(text + "\n")


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    if not line:
        log("end of input")
        sys.exit(0)
    try:
        return json.loads(line)
    except ValueError:
        log("not JSON: " + line.strip())
        return {}


def ask(method, params=None):
    """A request of the server's own to the client; what comes back."""
    global next_id
    next_id += 1
    rid = "srv-%d" % next_id
    send({"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params is not None else {})})
    while True:
        m = read()
        if m.get("id") == rid:
            return m
        log("while waiting: " + json.dumps(m))


def text(t, **more):
    return {"content": [{"type": "text", "text": t}], **more}


def call(name, a):
    if name == "echo":
        return text("echo: " + str(a.get("text")))
    if name == "add":
        s = a.get("a", 0) + a.get("b", 0)
        return text(str(s), structuredContent={"sum": s})
    if name == "big":
        return text("\n".join("line %d of the big answer, with some words after it" % i for i in range(1, int(a.get("n", 10)) + 1)))
    if name == "fail":
        return text("it broke: the disk is full", isError=True)
    if name == "slow":
        time.sleep(float(a.get("seconds", 1)))
        return text("slept %s seconds" % a.get("seconds", 1))
    if name == "ping_back":
        ping = ask("ping")
        roots = ask("roots/list")
        return text("the client answered the ping with %s and the folders with %s" % (
            json.dumps(ping.get("result")), (roots.get("error") or {}).get("code", json.dumps(roots.get("result")))))
    if name == "picture":
        return {"content": [{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
                            {"type": "text", "text": "a caption"}]}
    if name == "env":
        return text(os.environ.get(str(a.get("name")), "(not set)"))
    if name == "struct":
        return {"content": [], "structuredContent": {"rows": [1, 2, 3]}}
    if name == "grow":
        if not any(t["name"] == "late" for t in TOOLS):
            TOOLS.append({"name": "late", "description": "The tool that came later.", "inputSchema": schema({})})
        send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        return text("one more tool is there now")
    if name == "nested":
        return text("nested: " + json.dumps(a, sort_keys=True))
    if any(t["name"] == name for t in TOOLS):
        return text("%s was called with %s" % (name, json.dumps(a, sort_keys=True)))
    return None


def main():
    global calls
    if STDERR:
        sys.stderr.write(STDERR + "\n")
        sys.exit(1)
    if BANNER:
        print("Fake MCP server 1.0 starting (this line is no message)")
        sys.stdout.flush()
    while True:
        m = read()
        method, rid = m.get("method"), m.get("id")
        log(str(method) + ("" if method != "tools/call" else " " + str(m.get("params", {}).get("name"))))
        if HANG:
            continue
        if rid is None:
            continue  # a notification: initialized, cancelled
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": m.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "fake-mcp", "version": "1.0"}}})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif method == "tools/list":
            at = int((m.get("params") or {}).get("cursor") or 0)
            if PAGES:
                page = {"tools": TOOLS[at:at + 2]}
                if at + 2 < len(TOOLS):
                    page["nextCursor"] = str(at + 2)
            else:
                page = {"tools": TOOLS}
            send({"jsonrpc": "2.0", "id": rid, "result": page})
        elif method == "tools/call":
            p = m.get("params") or {}
            if DIE_IN_CALL:
                sys.stderr.write("fake-mcp: out of memory, giving up\n")
                sys.exit(3)
            r = None if p.get("name") == "refuse" else call(p.get("name"), p.get("arguments") or {})
            if r is None:
                send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "Unknown tool: %s" % p.get("name")}})
            else:
                send({"jsonrpc": "2.0", "id": rid, "result": r})
            calls += 1
            if DIE_AFTER and calls >= DIE_AFTER:
                log("ending after %d calls" % calls)
                sys.exit(0)
        else:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}})


if __name__ == "__main__":
    main()
