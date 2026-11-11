# Web search

Give TokenStation5 a [Tavily](https://tavily.com) API key and a chat that looks like it needs the web gets a `web_search` tool: one that asks to look something up, names a website, or asks about what changes (news, weather, prices, scores, "today", "latest"). Other chats get no tool, so the model answers them itself with no credit spent. When the model searches, ts5 calls Tavily over HTTPS from the console, gives the results to the model, and the model answers from them with the URLs it used. No PC or proxy is involved, and it works from any chat client, including ones that know nothing about tools. Only the search words leave the house.

## Turn it on

1. Get a key at [app.tavily.com](https://app.tavily.com) (it starts with `tvly-`).
2. Settings > **Tavily key**: press cross and type it with the controller or a USB or Bluetooth keyboard. It counts from the next request. Square twice removes it.

From a computer, `scripts/ts5-ps5.sh websearch tvly-YOUR-KEY` does the same (`websearch off` turns it off). On Linux, the terminal app has a field for it: Settings → **Web search key** (see [Using the terminal app](Using-the-terminal-app#settings-f4)); it counts for a model ts5-tui loads after that. For the always-on service, put `TAVILY_API_KEY=tvly-YOUR-KEY` in `/etc/default/tokenstation` and restart it; for `ts5 serve` run by hand, set `TAVILY_API_KEY` in its environment.

`GET /v1/models` shows what the loaded model can do: `"capabilities": {"tools": true, "tool_format": "hermes", "web_search": true}`.

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

## Clients with their own tools

ts5 also speaks OpenAI tool calling for the client's own tools (`tools` in the request, `tool_calls` in the answer, `role: "tool"` messages back), streaming included, so opencode, Continue, Open WebUI and similar clients can use their tools with the console's models. A request with tools of its own gets ts5's `web_search` only when it says `"web_search": true`: a coding agent's messages are full of URLs and dates that would trip the "needs the web" check on every turn.

## Security

- The key sits in `ts5.env`, which anyone on your network can read over the jailbreak's FTP server. Use a key you can revoke, on a network you trust. The log never shows the key itself.
- Search results are text from the web, and a page can try to talk the model into something. With only a search tool the worst case is a wrong answer; be careful giving the same conversation tools that change things.
- HTTPS is checked against the Mozilla CA bundle built into ts5 (`TS5_CA_FILE` points at another one).
