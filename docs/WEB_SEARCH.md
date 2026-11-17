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

## Without a key: the web through DuckDuckGo (Linux)

With no Tavily key a search on Linux is still a regular web search, and only
for a request that asks for a search: `"web_search": true`, or `"web_search":
"auto"` on a chat that looks like it needs the web or asks to search (with
`"auto"`, once a chat needed the search it stays offered, so the chat's
prompt starts the same from turn to turn). ts5-tui asks that way while its
search tool is switched on, so a model in ts5-tui can search out of the box.
A plain request that says nothing gets no search, as before.

ts5 asks for DuckDuckGo's results page that needs no scripts
(`html.duckduckgo.com/html`), the way a browser asks for a page, and gives
the model the results as text: title, the site's own address and the line
or two under it, eight of them (`TS5_SEARCH_RESULTS`), without the
advertisements. The passages are short, so the tool's description tells the
model to open a result's page when it needs more. Of eight search sites
tried from a PS5 on 2026-11-17 it was the one that answered such a request
with real results, in about 0.7 s: Google's page wants scripts, Bing sent
unrelated results, Brave, Mojeek and Startpage a check for robots.

When DuckDuckGo can't be searched (it sends its own check for robots, which
ts5 does not answer, or it can't be reached), the search goes to Grokipedia,
and when that can't be searched either, to Wikipedia. The model is told
which it got and that an encyclopedia has no news or live data.

`TS5_SEARCH_SOURCE` picks what is searched without a key: `web` (the
default), `google` (below), `grokipedia`, `wikipedia` or `both` (the two
encyclopedias, below);
a request picks with `"web_search_source"`, and `/v1/models` lists the
choices under `capabilities.search_sources`. `/v1/models` says
`"web_search": false, "web_search_source": "web"`: `web_search` stays "a
key is set, and the search is offered by itself". On the PS5 title the
search is the key's alone.

## Without a key: Google through Chromium (Linux)

Google's results page wants scripts, so a plain request gets none of it. A
browser does: with `TS5_SEARCH_SOURCE=google` (or `"web_search_source":
"google"`; in ts5-tui Settings → Search source, or `/tools search google`)
ts5 opens the results page in a Chromium installed on the machine, the one
the browser tool uses, and gives the model what the page shows between its
"Search Results" heading and its page numbers, without the page's buttons
and boxes and without its advertisements ("Sponsored results"): the answer
Google's AI writes at the top when it shows one ("AI Overview"), the results
with their passages, and under them a list of where the numbered links lead
(the addresses of other sites only, each once). A result's number stands at
the end of the line that names its site.
The model is told that the answer at the top is written by an AI and is to
be checked against the results.

Since November 2026 Google's page no longer has its results' addresses: a
link to a result is an address of Google's own that sends on
(`/goto?url=` and a token that says nothing), and the page shows only the
site's name with a path it cuts short. ts5 asks Google where the first two
results lead: from the page, the way a click on the link asks, without
following the answer, so the site a result is on hears nothing of it until
the model opens the page. Two, because Google sends on for two a results
page: asked for more, at once or one after another, it answers each with
a page that says to wait (3 seconds, then longer each time, up to 28
seconds on a PS5 on 17 November 2026). ts5 does not sit that wait out and
does not go around it. A result whose line under the title is its site's
address and nothing more (a front page) needs no asking. (An address
Google still writes into a link, as `/url?q=` did, is read from the link.)

The other results keep their passages and the line with their site's
name, and their addresses come from elsewhere: while the browser reads
Google's page, ts5 makes the same search on DuckDuckGo (as `web` does),
and adds its results under Google's page as a list of titles with their
addresses, without those Google's list already has. The model is told
which list is whose. When DuckDuckGo can't be searched, Google's page
stands alone, with a line that says the other addresses are not known.
That one DuckDuckGo search is also the one that answers when Google can't
be searched (below), so no search is made twice.

On a PS5 (17 November 2026, six searches) a search took 4.6 to 6.6
seconds and Chromium about 500 MiB of memory while it ran; four of the
six had an answer by Google's AI.

- It is there where a Chromium is: `/v1/models` lists `google` under
  `capabilities.search_sources` only then. `TS5_SEARCH_BROWSER` names
  another program than the one found (a path, or a name on the `PATH`).
- A browser starts for each search and ends with it. That takes seconds
  where DuckDuckGo's page takes under one, and memory: with less than
  900 MiB free the browser is not started. A search is one page and at
  most two short requests to Google, and one page of DuckDuckGo's.
- Google's question about cookies, where it asks one, is answered with its
  button for no ("Reject all", in the page's language).
- Its check for robots ("unusual traffic", a captcha) is not answered.
  Then, as when no Chromium is installed, the memory is short or the page
  lists nothing, the search goes on as `web` does (DuckDuckGo, Grokipedia,
  Wikipedia), and the result starts with a line that says why Google
  could not be searched.
- The page is asked for the way every page in the browser tool is: as
  Chrome on a Mac, in English (`hl=en`), with nothing that names ts5.

## Without a key: Wikipedia

With `TS5_SEARCH_SOURCE=wikipedia` (or `"web_search_source": "wikipedia"`)
the search is Wikipedia's own (its public API, no key, no account), for the
same requests as above, and also for one that names Wikipedia.

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

### Grokipedia (Linux)

On Linux the encyclopedia can also be [Grokipedia](https://grokipedia.com)
(xAI's, written by its AI), alone or together with Wikipedia:

- In ts5-tui: Settings → Search source, or `/tools search grokipedia`
  (`wikipedia`, `both`).
- For `ts5 serve` and the service: `TS5_SEARCH_SOURCE=grokipedia` or `both`
  (in `/etc/default/tokenstation` for the service).
- In a request: `"web_search_source": "grokipedia"` (or `"wikipedia"`,
  `"both"`) beside `"web_search"`. `/v1/models` lists what can be picked
  under `capabilities.encyclopedias`.

ts5 uses the site's own search (`/api/full-text-search`), under ts5's own
name, as with Wikipedia's API. The model gets the top articles with title,
link and text: the first with the opening of the article (one more
request, of which only the first 64 KiB are read), the others with the line
the search gives. The tool's description tells the model that this is
xAI's encyclopedia, written by an AI, so it can weigh it. **Both** asks the
two and gives half the results from each, Wikipedia's first, numbered on.

When Grokipedia can't be searched (the site is down, or answers with a
check for robots), ts5 does not try to get around it. If the search was a
name of a few words it opens the article of that name as an ordinary web
page, once. Failing that, the search goes to Wikipedia and the model is
told so in the first line of the results. With `TS5_WIKIPEDIA=off` it is a
failed search instead. `TS5_GROKIPEDIA_URL` points at another address with
the same API (the tests use it).

Note that the site's `robots.txt` asks robots to stay out of `/api/`. ts5
asks it only when a person's model searches, a request or two per search.

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
