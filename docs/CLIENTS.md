# Connecting an app

TokenStation5 speaks the OpenAI API on port 8081, over plain HTTP. Any app
that takes an "OpenAI compatible" provider works with three settings:

| Setting | Value |
| --- | --- |
| Base URL | `http://<your PS5's address>:8081/v1` (http, not https, and ending in `/v1`) |
| Model | the id that `http://<your PS5's address>:8081/v1/models` shows, such as `Qwen2.5-Coder-7B-Instruct-Q4_K_M` (the server answers with the model it has loaded whatever id is sent) |
| API key | one of your keys (Settings on the PS5, under Apps: New key). With "Require a key" off, any text works, and a real key still shows the app by name on Home |

`/v1/models` also gives the loaded model's `context_length`: tell your app
that number as the context window, so it shortens the conversation before
the PS5 has to refuse it. Coding tools send a long first prompt (10,000 to
15,000 tokens of instructions and tool descriptions), so serve the model with
`-c auto` (the default: as much context as fits) rather than a small fixed
context.

All of it is on the TV too: Settings, under Apps, "Set up an app" shows the
base URL, the model, the key to use and the context window. A DHCP
reservation in the router keeps the address from changing.

## API keys

Settings, under Apps, makes a key per app: "New key", name it after the app,
and the key shows once on its row, such as `ts5-k7qp2-wx9md-3hf8r-tz6na`
(no i, l, o or u, so it reads off the TV without mix-ups). Type it into the
app then; the PS5 keeps only its SHA-256 (in `keys.tsv` in the title's
folder). Cross turns a key off and on, square twice removes it. With
"Require a key" on, a request without a key that's on gets HTTP 401
(`invalid_api_key`); the Chat page on the PS5 never needs one. `ts5 serve`
on a computer takes one key from `TS5_API_KEY`.

## Kilo Code

Kilo Code reads custom providers from `~/.config/kilo/kilo.jsonc`. Add
TokenStation5 under `provider`:

```jsonc
{
  "provider": {
    "tokenstation5": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TokenStation5",
      "options": { "baseURL": "http://192.168.1.50:8081/v1", "apiKey": "ts5-k7qp2-wx9md-3hf8r-tz6na" },
      "models": {
        "Qwen2.5-Coder-7B-Instruct-Q4_K_M": {
          "name": "Qwen 2.5 Coder 7B",
          "tool_call": true,
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

Then pick `tokenstation5/Qwen2.5-Coder-7B-Instruct-Q4_K_M` as the model.
Put the model's `context_length` from `/v1/models` in `limit.context`. Leave
`reasoning` out for the Llama and Qwen 2.5 models (only Qwen 3 thinks).

## Continue, Cline, Roo Code and others

Choose the "OpenAI compatible" (or "OpenAI", with a custom base URL)
provider and fill in the three settings above. Where the app asks for a
context window, give it the `context_length`; where it asks for max output
tokens, 4,096 to 8,192 is plenty. Tool calling ("native" tools) works with
every model of the set.

## When it doesn't connect

- The Logs page on the TV has a line for each request. A request that never
  shows there never reached the PS5: check the address and that the app uses
  `http://`. An app that tries `https://` is named in the log
  ("tried https:// on the API port").
- "maximum context length" errors: the conversation is longer than the
  model's context. Serve with `-c auto`, or start a new task in the app.
