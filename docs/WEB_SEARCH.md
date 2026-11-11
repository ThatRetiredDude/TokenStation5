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
   environment of `ts5 serve`: ts5-tui's Settings → Web search key does it
   for a model ts5-tui loads, and a line `TAVILY_API_KEY=tvly-...` in
   `/etc/default/tokenstation` for the API-only service.

`GET /v1/models` shows what the loaded model can do:
`"capabilities": {"tools": true, "tool_format": "hermes", "web_search": true,
"web_search_source": "tavily"}`.

## Without a key: Wikipedia

With no Tavily key the search is Wikipedia's own (its public API, no key, no
account), and only for a request that asks for a search: `"web_search":
true`, or `"web_search": "auto"` on a chat that looks like it needs the web,
names Wikipedia or asks to search. ts5-tui asks that way while its search
tool is switched on, so a model in ts5-tui can look things up out of the box.
A plain request that says nothing gets no search, as before.

It is an encyclopedia: no news, prices, weather or anything live. The tool's
description tells the model so, and to say when the articles don't hold the
answer. The model gets the top articles (five, `TS5_SEARCH_RESULTS`), each
with its title, link and opening text; in ts5-tui it can then read a whole
article with the web page tool. A query no article matches word for word
goes once more as any of its words. Wikipedia's search splits and stems
words, so a name it doesn't know can still bring articles on something else
("Gated DeltaNet": airlines and a river): when the articles were found by
any of the words, or none has one of the query's own words in its title or
opening, the model is told that they are only the closest found. Every
result ends with a line that says to search once more with only the name of
the thing when the articles miss the subject. A search takes about half a
second on the PS5.

`TS5_WIKIPEDIA` picks the language (`en` by default; `sk`, `de` ...) or turns
it off (`off`). `/v1/models` says `"web_search": false, "web_search_source":
"wikipedia"` then: `web_search` stays "the whole web, by itself".

## Which models

ts5 reads the tool format from each model's own chat template when it loads it.

| Model | Tool calls | Web search |
| --- | --- | --- |
| Qwen3 8B | yes (Hermes format) | best of the four at deciding when to search |
| Llama 3.1 8B | yes (Llama JSON format) | good |
| Qwen 2.5 Coder 7B | yes (Hermes format) | works; less sure about when to search |
| Llama 3.2 3B | yes (Llama JSON format) | simple single searches |
| Mistral Nemo 12B | yes (Mistral format) | not tried yet |
| Gemma 4 12B | yes (Gemma 4 format) | not tried yet |
| DeepSeek R1 distills | yes (asked for in words) | not tried yet |

A model whose template has no tool format its file can write (DeepSeek R1's
distills, Llama 3.0, an early Mistral) still gets tools and search: ts5 asks
for a call in words, as `<tool_call>` JSON, in the model's own turns, and
gives the result back in a user turn, in plain words. `tool_format` says `generic` for these.
How well such a model calls tools depends on the model.

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
- A search ts5 runs is told to the client. A streamed answer gets a chunk
  with no text and a field `"ts5_web_search": {"query": "…", "state":
  "running"}` when the search starts, and one with `"state": "done"` (or
  `"failed"`) and a `"note"` such as `5 results (0.5 s, Wikipedia)` when it
  ends; a client that does not know the field sees two empty chunks. An
  answer that is not streamed has them as `"ts5_web_searches": [...]` beside
  `"usage"`. ts5-tui shows a line for each (`⚙ searches the web: …`).

## Clients with their own tools

ts5 also speaks OpenAI tool calling for the client's own tools (`tools` in the
request, `tool_calls` in the answer, `role: "tool"` messages back), streaming
included, so opencode, Continue, Open WebUI's native function calling and
similar clients can use their tools with the console's models. A request
with tools of its own gets ts5's `web_search` only when it says
`"web_search": true` (or `"web_search": "auto"`, which offers it when the
chat looks like it needs the web, as for a plain chat; or with
`TS5_WEB_SEARCH=always`): a coding agent's
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
