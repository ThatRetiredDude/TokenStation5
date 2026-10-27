# Web search

ts5 can search the web for the model. Give it a [Tavily](https://tavily.com)
API key and a chat that looks like it needs the web gets a `web_search` tool:
one that asks to look something up or names a website, or asks about what
changes (news, weather, prices, scores, elections, "today", "latest", a year
after 2023). Other chats get no tool, so the model answers them itself, with
no credit spent and a shorter prompt. When the model searches, ts5 calls
Tavily over HTTPS from the console, gives the results back to the model, and
the model answers from them with the URLs it used. No PC, Pi or proxy is
involved, and it works from any chat client, including ones that know nothing
about tools.

## Turn it on

1. Get a key at [app.tavily.com](https://app.tavily.com) (it starts with `tvly-`).
2. On the console: on the title's Settings page, choose **Tavily key**, press
   cross and type it on the console's keyboard (with the controller, or a USB
   or Bluetooth keyboard). It counts from the next request.

   Or from the PC:

   ```sh
   scripts/ts5-ps5.sh websearch tvly-YOUR-KEY   # on, then the title restarts
   scripts/ts5-ps5.sh websearch                 # is it on?
   scripts/ts5-ps5.sh websearch off             # off
   ```

   This adds `TAVILY_API_KEY=...` to `ts5.env` in the title's folder and keeps
   the other settings there. On Linux, set `TAVILY_API_KEY` in the
   environment of `ts5 serve`.

`GET /v1/models` shows what the loaded model can do:
`"capabilities": {"tools": true, "tool_format": "hermes", "web_search": true}`.

## Which models

ts5 reads the tool format from each model's own chat template when it loads it.

| Model | Tool calls | Web search |
| --- | --- | --- |
| Qwen3 8B | yes (Hermes format) | best of the four at deciding when to search |
| Llama 3.1 8B | yes (Llama JSON format) | good |
| Qwen 2.5 Coder 7B | yes (Hermes format) | works; less sure about when to search |
| Llama 3.2 3B | yes (Llama JSON format) | simple single searches |
| Mistral Nemo 12B | yes (Mistral format) | not tried yet |

A model whose template has no tool format gets no tools and no search;
everything else works as before.

## How it behaves

- Searches cost prompt tokens: each result is cut to about 1,200 characters,
  five results per search (`TS5_SEARCH_RESULTS` changes that). At today's
  console speeds that adds a few seconds per search before the answer.
- Only a chat whose last message looks like it needs the web is offered the
  tool. With it in the prompt (about 250 tokens), Llama 3's template tells
  the model to answer with a call, and on the PS5 Llama 3.1 8B searched for
  "how does a refrigerator keep food cold" and a Fibonacci function; without
  it, those cost nothing. The check is a list of words and phrases in
  `needs_web` (`engine/src/websearch.cpp`); a question it misses gets an
  answer from what the model knows. `"web_search": true` in a request offers
  the tool anyway, and `TS5_WEB_SEARCH=always` in `ts5.env` offers it on every
  chat as before.
- At most three rounds of searches per answer. A model that still wants to
  search after that answers from what the searches found, without tools, so
  there is always an answer.
- A failed search (no network, a refused key, Tavily's limit) goes back to the
  model as text saying why, so it can tell you instead of stalling. ts5 does
  not call Tavily again in that answer; if the model tries to search again, it
  answers without tools from what it knows and says the search failed. The
  log (`scripts/ts5-ps5.sh log`) has a line per search with the query, the
  number of results and the time.
- A request can turn it off with `"web_search": false`, or with
  `"tool_choice": "none"`, and turn it on with `"web_search": true`.

## Clients with their own tools

ts5 also speaks OpenAI tool calling for the client's own tools (`tools` in the
request, `tool_calls` in the answer, `role: "tool"` messages back), streaming
included, so opencode, Continue, Open WebUI's native function calling and
similar clients can use their tools with the console's models. A request
with tools of its own gets ts5's `web_search` only when it says
`"web_search": true` (or with `TS5_WEB_SEARCH=always`): a coding agent's
messages are full of URLs, dates and file text that would pass the "needs the
web" check on every turn, and each search costs a credit. With it, ts5
answers `web_search` calls itself and hands every other call back to the
client. A client that defines its own `web_search` tool keeps it.

## Security

- The key sits in `ts5.env`, which anyone on your network can read over the
  console's FTP server (it has no password). The title's log shows it as
  `TAVILY_API_KEY=(set)`, never the key itself. Use a key you can revoke, on
  a network you trust.
- Search results are text from the web, and a page can try to talk the model
  into something. With only a search tool the worst case is a wrong answer;
  be careful giving the same conversation tools that change things.
- HTTPS is checked against the Mozilla CA bundle built into ts5
  (`TS5_CA_FILE` points at another one).
