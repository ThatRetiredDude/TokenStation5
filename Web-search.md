# Web search

Give TokenStation5 a [Tavily](https://tavily.com) API key and a chat that looks like it needs the web gets a `web_search` tool: one that asks to look something up, names a website, or asks about what changes (news, weather, prices, scores, "today", "latest"). Other chats get no tool, so the model answers them itself with no credit spent. When the model searches, ts5 calls Tavily over HTTPS from the console, gives the results to the model, and the model answers from them with the URLs it used. No PC or proxy is involved, and it works from any chat client, including ones that know nothing about tools. Only the search words leave the house.

## Turn it on

1. Get a key at [app.tavily.com](https://app.tavily.com) (it starts with `tvly-`).
2. Settings > **Tavily key**: press cross and type it with the controller or a USB or Bluetooth keyboard. It counts from the next request. Square twice removes it.

From a computer, `scripts/ts5-ps5.sh websearch tvly-YOUR-KEY` does the same (`websearch off` turns it off). On Linux, the terminal app has a field for it: Settings → **Web search key** (see [Using the terminal app](Using-the-terminal-app#settings-f4)); it counts for a model ts5-tui loads after that. For the always-on service, put `TAVILY_API_KEY=tvly-YOUR-KEY` in `/etc/default/tokenstation` and restart it; for `ts5 serve` run by hand, set `TAVILY_API_KEY` in its environment.

`GET /v1/models` shows what the loaded model can do: `"capabilities": {"tools": true, "tool_format": "hermes", "web_search": true}`.

## Without a key: an encyclopedia (Linux edition)

With no Tavily key the Linux edition searches an encyclopedia instead: Wikipedia unless you choose another, through its public API, with no key and no account. It does so only for a request that asks for a search. The terminal app asks while its search tool is on; over the API a request says `"web_search": true`, or `"web_search": "auto"` to leave it to the message.

- The model gets the top articles: title, link and opening text.
- It is told that this is an encyclopedia, with no news and no live data, and it is told when the articles are only a loose match for the question, so it says that it could not look the thing up instead of answering from the wrong article.
- A search takes about half a second on the PS5.
- `TS5_WIKIPEDIA=sk` (or any language code) picks another Wikipedia; `TS5_WIKIPEDIA=off` turns it off.
- Wikipedia refuses searches for a while from an address that has sent it very many (it did after some hundreds of searches and pages in two hours of our tests). ts5 then says that Wikipedia has had too many searches from this address, and the model tells you so. It lifts by itself.
- `/v1/models` says `"web_search_source": "wikipedia"` (`"tavily"` with a key).

**Grokipedia** (xAI's encyclopedia, written by its AI) can be searched instead of Wikipedia, or beside it:

- In the terminal app: Settings → **Search source**, or `/tools search wikipedia`, `grokipedia` or `both`. For the always-on service: `TS5_ENCYCLOPEDIA=grokipedia` (or `both`) in `/etc/default/tokenstation`. Over the API a request can say `"web_search_source": "grokipedia"`, `"both"` or `"wikipedia"`.
- The model gets the top articles as with Wikipedia: title, link, a line from each, and the opening of the first. It is told which encyclopedia it searched and that Grokipedia's articles are written by an AI.
- **Both** gives half the results from each, and the result says that two encyclopedias were searched and which results are whose.
- When Grokipedia doesn't answer (it is down, or it turns the request away), the search goes to Wikipedia and the result's first line says so. With `TS5_WIKIPEDIA=off` it fails instead, and the model is told why.
- Grokipedia's `robots.txt` asks programs to stay away from its search address. ts5 asks under its own name, one search at a time, only when a person's chat calls for it.
- `/v1/models` lists `"encyclopedias": ["wikipedia", "grokipedia", "both"]` under `capabilities` on a build that has the choice.

The Prospero Store title searches only with a key.

## Which models

| Model | Tool calls | Web search |
| --- | --- | --- |
| Qwen 3 8B | yes (Hermes format) | best at deciding when to search |
| Llama 3.1 8B | yes (Llama JSON format) | good |
| Qwen 2.5 Coder 7B | yes (Hermes format) | works; less sure about when to search |
| Llama 3.2 3B | yes (Llama JSON format) | simple single searches |
| Mistral Nemo 12B | yes (Mistral format) | not tried yet |
| Gemma 3 12B | yes (Hermes format in Gemma turns) | works |
| Gemma 4 12B (Linux edition) | yes (its own format) | not tried yet |
| DeepSeek R1 Distill 14B (Linux edition) | yes (asked for in words) | not tried yet |

ts5 reads the tool format from each model's own chat template. On the Prospero Store title a model whose template has no tool format gets no tools and no search. On the Linux edition such a model (DeepSeek R1, an early Llama 3 or Mistral) is asked in words to write its calls in Qwen 3's tags, and `/v1/models` says `"tool_format": "generic"`; how well it keeps to that depends on the model.

## How it behaves

- Each result is cut to about 1,200 characters, five results per search (`TS5_SEARCH_RESULTS` changes that). A search adds a few seconds before the answer.
- Only a chat whose last message looks like it needs the web is offered the tool. `"web_search": true` in a request offers it anyway; `TS5_WEB_SEARCH=always` in `ts5.env` offers it on every chat. A request turns it off with `"web_search": false` or `"tool_choice": "none"`.
- At most three rounds of searches per answer; after that the model answers from what it found.
- A failed search (no network, a refused key, Tavily's limit) goes back to the model as text saying why, so it tells you instead of stalling.
- On the Linux edition a search ts5 runs is told to the client. A streamed answer gets a chunk with no text and `"ts5_web_search": {"query": "…", "state": "running"}` when the search starts, and one with `"state": "done"` (or `"failed"`) and a `"note"` such as `5 results (0.5 s, Wikipedia)` when it ends. A client that doesn't know the field sees two empty chunks. The terminal app shows a line for each.

## Clients with their own tools

ts5 also speaks OpenAI tool calling for the client's own tools (`tools` in the request, `tool_calls` in the answer, `role: "tool"` messages back), streaming included, so opencode, Continue, Open WebUI and similar clients can use their tools with the console's models. A request with tools of its own gets ts5's `web_search` only when it says `"web_search": true`: a coding agent's messages are full of URLs and dates that would trip the "needs the web" check on every turn.

## Security

- The key sits in `ts5.env`, which anyone on your network can read over the jailbreak's FTP server. Use a key you can revoke, on a network you trust. The log never shows the key itself.
- Search results are text from the web, and a page can try to talk the model into something. With only a search tool the worst case is a wrong answer; be careful giving the same conversation tools that change things.
- HTTPS is checked against the Mozilla CA bundle built into ts5 (`TS5_CA_FILE` points at another one).
