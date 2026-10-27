#!/usr/bin/env python3
"""Checks that the model the console is serving answers the way apps expect,
through the OpenAI API: chat, streaming, a system message, several turns,
limits and stop strings, tool calls and their results, a small coding-agent
task, a long prompt and the error for one that's too long.

    python3 scripts/model-check.py                       # PS5_HOST from the env
    python3 scripts/model-check.py --host 10.35.12.121 --only chat,tools
    TS5_KEY=ts5-... python3 scripts/model-check.py      # when a key is needed
    python3 scripts/model-check.py --repeat 3 --sampling '{"temperature": 0.6, "top_p": 0.9}'

Every check prints PASS or FAIL with what came back, and --out FILE saves
them all as JSON, with each check's last reply whole. --sampling gives every
request settings it doesn't set itself, and --repeat runs the checks that many
times and counts each one's passes, to compare settings for a model. Each request sends
"web_search": false, so no check spends a search. The agent check never runs what the model writes: it compiles the
file and looks at it. Standard library only.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

FILLER = ("The quick brown fox jumps over the lazy dog while the console fan "
          "hums and the GPU streams weights from GDDR6. ")  # ~25 tokens

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "The current weather in a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "The city, such as Paris"}},
        "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "The current local time in a time zone.",
    "parameters": {"type": "object", "properties": {
        "zone": {"type": "string", "description": "An IANA zone, such as Europe/Paris"}},
        "required": ["zone"]}}}


class Api:
    def __init__(self, base, key, timeout, sampling=None):
        self.base, self.key, self.timeout = base, key, timeout
        self.sampling = sampling or {}  # what every request gets unless it sets its own
        self.last = None  # the last chat reply, whole, for --out

    def _req(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.Request(self.base + path, data=data, headers=headers)

    def get(self, path):
        with urllib.request.urlopen(self._req(path), timeout=self.timeout) as r:
            return json.load(r)

    def chat(self, body):
        """The reply as JSON, or (status, error JSON) for an HTTP error."""
        body = dict(self.sampling, **body, web_search=False)
        try:
            with urllib.request.urlopen(self._req("/v1/chat/completions", body), timeout=self.timeout) as r:
                self.last = json.load(r)
        except urllib.error.HTTPError as e:
            try:
                self.last = {"_status": e.code, **json.load(e)}
            except ValueError:
                self.last = {"_status": e.code}
        return self.last

    def stream(self, body):
        """The stream's chunks, in order, and whether it ended with [DONE]."""
        body = dict(self.sampling, **body, web_search=False, stream=True)
        chunks, done = [], False
        with urllib.request.urlopen(self._req("/v1/chat/completions", body), timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunks.append(json.loads(data))
        return chunks, done

    def completion(self, body):
        body = dict(self.sampling, **body)
        with urllib.request.urlopen(self._req("/v1/completions", body), timeout=self.timeout) as r:
            return json.load(r)


def message(reply):
    return (reply.get("choices") or [{}])[0].get("message") or {}


def finish(reply):
    return (reply.get("choices") or [{}])[0].get("finish_reason")


def text(reply):
    return (message(reply).get("content") or "").strip()


def short(s, n=160):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 3] + "..."


# Each check returns (passed, what came back).

def check_models(api, ctx):
    m = api.get("/v1/models")
    data = m.get("data") or []
    if not data:
        return False, "no model listed"
    ctx["model"] = data[0]["id"]
    ctx["tools"] = (data[0].get("capabilities") or {}).get("tools", False)
    ctx["n_ctx"] = data[0].get("context_length") or 0
    return True, "%s, context %s, tools %s" % (ctx["model"], ctx["n_ctx"], ctx["tools"])


def check_chat(api, ctx):
    r = api.chat({"model": ctx["model"], "messages": [
        {"role": "user", "content": "What is 17 times 23? Answer with just the number."}]})
    return "391" in text(r), "%s (finish %s, %s tokens)" % (short(text(r)), finish(r),
                                                             (r.get("usage") or {}).get("completion_tokens"))


def check_short_limit(api, ctx):
    # Apps often cap a reply (titles, commit messages): the answer must fit.
    r = api.chat({"model": ctx["model"], "max_tokens": 64, "messages": [
        {"role": "user", "content": "Name the capital of France in one word."}]})
    return "paris" in text(r).lower(), "%r (finish %s, reasoning %d chars)" % (
        short(text(r)), finish(r), len(message(r).get("reasoning_content") or ""))


def check_no_thinking(api, ctx):
    # vLLM's and llama.cpp's switch for thinking models; others ignore it.
    r = api.chat({"model": ctx["model"], "chat_template_kwargs": {"enable_thinking": False}, "messages": [
        {"role": "user", "content": "Name the capital of Italy in one word."}]})
    reasoning = message(r).get("reasoning_content") or ""
    return "rome" in text(r).lower() and not reasoning.strip(), "%r, reasoning %d chars" % (
        short(text(r)), len(reasoning))


def check_stream(api, ctx):
    chunks, done = api.stream({"model": ctx["model"], "stream_options": {"include_usage": True}, "messages": [
        {"role": "user", "content": "Count from 1 to 5, separated by spaces."}]})
    content = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in chunks
                      if c.get("choices"))
    finishes = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
    usage = any(c.get("usage") for c in chunks)
    ok = done and all(str(i) in content for i in range(1, 6)) and "stop" in finishes and usage
    return ok, "%d chunks, [DONE] %s, finish %s, usage %s: %r" % (
        len(chunks), done, [f for f in finishes if f], usage, short(content))


def check_system(api, ctx):
    r = api.chat({"model": ctx["model"], "messages": [
        {"role": "system", "content": "You always answer in French, never in English."},
        {"role": "user", "content": "What colour is the sky on a clear day? One short sentence."}]})
    t = text(r).lower()
    return any(w in t for w in ("ciel", "bleu", "est")), short(text(r))


def check_turns(api, ctx):
    r = api.chat({"model": ctx["model"], "messages": [
        {"role": "user", "content": "My name is Ravi and my favourite number is 42."},
        {"role": "assistant", "content": "Nice to meet you, Ravi! 42 is a great number."},
        {"role": "user", "content": "What is my name, and what is my favourite number?"}]})
    t = text(r).lower()
    return "ravi" in t and "42" in t, short(text(r))


def check_parts(api, ctx):
    # Content as a list of parts, as Kilo Code, Cline and the OpenAI SDKs send it.
    r = api.chat({"model": ctx["model"], "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Reply with the word"}, {"type": "text", "text": " banana, and nothing else."}]}]})
    return "banana" in text(r).lower(), short(text(r))


def check_stop(api, ctx):
    r = api.chat({"model": ctx["model"], "stop": ["7"], "messages": [
        {"role": "user", "content": "Count from 1 to 10, separated by spaces, nothing else."}]})
    t = text(r)
    return "7" not in t and "6" in t, "%r (finish %s)" % (short(t), finish(r))


def check_json(api, ctx):
    r = api.chat({"model": ctx["model"], "response_format": {"type": "json_object"}, "messages": [
        {"role": "user", "content": 'Give me a JSON object with keys "city" and "country" for the Eiffel Tower. '
                                    "Only the JSON."}]})
    t = re.sub(r"^```(json)?|```$", "", text(r).strip()).strip()
    try:
        j = json.loads(t)
        return isinstance(j, dict) and "paris" in json.dumps(j).lower(), short(t)
    except ValueError:
        return False, "not JSON: " + short(text(r))


def check_completion(api, ctx):
    r = api.completion({"model": ctx["model"], "prompt": "The capital of Japan is", "max_tokens": 8,
                        "temperature": 0})
    t = (r.get("choices") or [{}])[0].get("text") or ""
    return "tokyo" in t.lower(), repr(short(t))


def tool_call(r):
    calls = message(r).get("tool_calls") or []
    return calls[0] if calls else None


def check_tool_call(api, ctx):
    if not ctx.get("tools"):
        return None, "the model takes no tools"
    r = api.chat({"model": ctx["model"], "tools": [WEATHER, TIME], "messages": [
        {"role": "user", "content": "What's the weather like in Paris right now?"}]})
    c = tool_call(r)
    if not c:
        return False, "no call: %r (finish %s)" % (short(text(r)), finish(r))
    args = json.loads(c["function"]["arguments"] or "{}")
    ok = c["function"]["name"] == "get_weather" and "paris" in json.dumps(args).lower() and finish(r) == "tool_calls"
    ctx["call"] = c
    return ok, "%s(%s), finish %s" % (c["function"]["name"], c["function"]["arguments"], finish(r))


def check_tool_result(api, ctx):
    c = ctx.get("call")
    if not c:
        return None, "needs the tool call check"
    r = api.chat({"model": ctx["model"], "tools": [WEATHER, TIME], "messages": [
        {"role": "user", "content": "What's the weather like in Paris right now?"},
        {"role": "assistant", "content": None, "tool_calls": [c]},
        {"role": "tool", "tool_call_id": c["id"], "content": '{"city": "Paris", "sky": "light rain", "celsius": 11}'}]})
    t = text(r).lower()
    return ("11" in t or "rain" in t) and not tool_call(r), "%r (finish %s)" % (short(text(r)), finish(r))


def check_tool_choice(api, ctx):
    if not ctx.get("tools"):
        return None, "the model takes no tools"
    r1 = api.chat({"model": ctx["model"], "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Say hello in one word."}]})
    r2 = api.chat({"model": ctx["model"], "tools": [WEATHER], "tool_choice": "none", "messages": [
        {"role": "user", "content": "What's the weather in Rome?"}]})
    # A call written out as text (a made-up tool) is no answer either.
    t1 = text(r1).strip()
    ok = not tool_call(r1) and "hello" in t1.lower() and not t1.startswith("{") and not tool_call(r2) and text(r2)
    return ok, "no-need: %r; tool_choice none: %r" % (short(text(r1), 60), short(text(r2), 60))


def check_no_tool(api, ctx):
    """Questions the one tool doesn't fit: a plain answer, no call."""
    if not ctx.get("tools"):
        return None, "the model takes no tools"
    got = []
    for q, want in [("What is 7 times 6?", "42"), ("Write a two-line poem about rain.", "")]:
        r = api.chat({"model": ctx["model"], "tools": [WEATHER], "messages": [{"role": "user", "content": q}]})
        t = text(r).strip()
        got.append((not tool_call(r) and want in t and len(t) > 1 and not t.startswith("{"), t, tool_call(r)))
    ok = all(g[0] for g in got)
    return ok, "; ".join("%s" % (("call %s" % g[2]["function"]["name"]) if g[2] else repr(short(g[1], 50)))
                         for g in got)


def check_two_calls(api, ctx):
    if not ctx.get("tools"):
        return None, "the model takes no tools"
    r = api.chat({"model": ctx["model"], "tools": [WEATHER, TIME], "messages": [
        {"role": "user", "content": "What's the weather in Paris, and what time is it in Tokyo?"}]})
    names = [c["function"]["name"] for c in message(r).get("tool_calls") or []]
    return sorted(names) == ["get_time", "get_weather"], "calls %s, text %r" % (names, short(text(r), 60))


def check_tool_stream(api, ctx):
    if not ctx.get("tools"):
        return None, "the model takes no tools"
    chunks, done = api.stream({"model": ctx["model"], "tools": [WEATHER], "messages": [
        {"role": "user", "content": "What's the weather in Berlin?"}]})
    calls = [tc for c in chunks if c.get("choices") for tc in c["choices"][0].get("delta", {}).get("tool_calls") or []]
    finishes = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
    ok = done and calls and calls[0]["function"]["name"] == "get_weather" and "tool_calls" in finishes
    return ok, "%d call chunks, finish %s" % (len(calls), [f for f in finishes if f])


AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List the files in the project folder.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file in the project folder.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File name, such as main.py"},
            "content": {"type": "string", "description": "The whole file"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file in the project folder.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]


def check_agent(api, ctx):
    """A coding agent's loop: the model calls tools until it says it's done."""
    if not ctx.get("tools"):
        return None, "the model takes no tools"
    files = {"README.md": "# fizz\nA tiny project.\n"}
    msgs = [{"role": "system", "content": "You are a coding agent. Work only through the tools. When the task is "
                                          "done, reply with a one-sentence summary and no tool call."},
            {"role": "user", "content": "Create fizzbuzz.py: it prints the numbers 1 to 15, but Fizz for multiples "
                                        "of 3, Buzz for multiples of 5 and FizzBuzz for both."}]
    steps = []
    for _ in range(6):
        r = api.chat({"model": ctx["model"], "tools": AGENT_TOOLS, "messages": msgs})
        calls = message(r).get("tool_calls") or []
        if not calls:
            break
        msgs.append({"role": "assistant", "content": message(r).get("content"), "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except ValueError:
                args, out = {}, "error: the arguments aren't JSON"
            if name == "list_files":
                out = "\n".join(sorted(files))
            elif name == "write_file" and "path" in args:
                files[args["path"]] = args.get("content", "")
                out = "wrote %s (%d bytes)" % (args["path"], len(files[args["path"]]))
            elif name == "read_file":
                out = files.get(args.get("path", ""), "error: no such file")
            else:
                out = "error: unknown tool or arguments"
            steps.append(name)
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    code = files.get("fizzbuzz.py", "")
    try:
        compile(code, "fizzbuzz.py", "exec")  # compiled, never run
        compiles = bool(code)
    except SyntaxError:
        compiles = False
    ok = compiles and "Fizz" in code and "Buzz" in code and text(r) and not (message(r).get("tool_calls"))
    detail = "steps %s, fizzbuzz.py %d bytes, compiles %s, summary %r" % (steps, len(code), compiles,
                                                                          short(text(r), 80))
    if code and not compiles:
        detail += ", file %r" % code[:300]
    return ok, detail


def check_long(api, ctx):
    # About the size of a coding app's system prompt and tools (Kilo Code's is ~15,700 tokens).
    n = min(14000, int((ctx.get("n_ctx") or 8192) * 0.6))
    filler = FILLER * (n // 50)  # twice: before the word and after it
    t0 = time.time()
    r = api.chat({"model": ctx["model"], "max_tokens": 400, "messages": [
        {"role": "system", "content": "Notes follow.\n" + filler + "\nThe secret word is HELIOTROPE.\n" + filler},
        {"role": "user", "content": "What is the secret word in the notes? Answer with just the word."}]})
    s = time.time() - t0
    u = r.get("usage") or {}
    return "heliotrope" in text(r).lower(), "%s prompt tokens in %.1f s: %r" % (
        u.get("prompt_tokens"), s, short(text(r), 60))


def check_too_long(api, ctx):
    n = (ctx.get("n_ctx") or 8192) + 2000
    r = api.chat({"model": ctx["model"], "messages": [{"role": "user", "content": FILLER * (n // 20)}]})
    code = (r.get("error") or {}).get("code")
    return r.get("_status") == 400 and code == "context_length_exceeded", "HTTP %s, code %s" % (r.get("_status"), code)


CHECKS = [("models", check_models), ("chat", check_chat), ("stream", check_stream), ("system", check_system),
          ("turns", check_turns), ("parts", check_parts), ("limit", check_short_limit),
          ("nothink", check_no_thinking), ("stop", check_stop), ("json", check_json),
          ("completion", check_completion), ("tools", check_tool_call), ("toolresult", check_tool_result),
          ("toolchoice", check_tool_choice), ("notool", check_no_tool), ("twocalls", check_two_calls),
          ("toolstream", check_tool_stream), ("agent", check_agent), ("long", check_long),
          ("toolong", check_too_long)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("PS5_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--key", default=os.environ.get("TS5_KEY", ""))
    ap.add_argument("--only", default="", help="comma-separated check names (models always runs)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default="", help="save the results as JSON")
    ap.add_argument("--sampling", default="{}",
                    help='JSON settings for every request that has none of its own, e.g. {"temperature": 0.6}')
    ap.add_argument("--repeat", type=int, default=1, help="run the checks this many times")
    a = ap.parse_args()
    api = Api("http://%s:%d" % (a.host, a.port), a.key, a.timeout, json.loads(a.sampling))
    only = {x for x in a.only.split(",") if x}
    ctx, runs = {}, []
    for run in range(a.repeat):
        if a.repeat > 1:
            print("== run %d of %d" % (run + 1, a.repeat), flush=True)
        results = []
        for name, fn in CHECKS:
            if only and name not in only and name != "models":
                continue
            t0 = time.time()
            api.last = None
            try:
                ok, detail = fn(api, ctx)
            except Exception as e:  # a check that throws is a failed check
                ok, detail = False, "%s: %s" % (type(e).__name__, e)
            s = time.time() - t0
            mark = "SKIP" if ok is None else "PASS" if ok else "FAIL"
            print("%-4s %-10s %6.1f s  %s" % (mark, name, s, detail), flush=True)
            results.append({"check": name, "result": mark, "seconds": round(s, 1), "detail": detail,
                            "last_reply": api.last})
            if name == "models" and not ok:
                break
        runs.append(results)
        if results and results[0]["result"] == "FAIL":
            break
    if a.out:
        with open(a.out, "w") as f:
            out = {"model": ctx.get("model"), "sampling": api.sampling, "results": runs[0]}
            if a.repeat > 1:
                out["runs"] = runs
            json.dump(out, f, indent=1)
    passes, failed = 0, []
    for name in [r["check"] for r in runs[0]]:
        marks = [r["result"] for results in runs for r in results if r["check"] == name]
        passes += marks.count("PASS")
        if "FAIL" in marks:
            failed.append(name if len(runs) == 1 else "%s %d/%d" % (name, marks.count("PASS"), len(marks)))
    print("%s: %d passed, %d failed%s%s" % (
        ctx.get("model"), passes, sum(r["result"] == "FAIL" for results in runs for r in results),
        " in %d runs" % len(runs) if len(runs) > 1 else "", (" (" + ", ".join(failed) + ")") if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
