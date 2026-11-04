# Connecting apps

TokenStation5 speaks the OpenAI API on port 8081, over plain HTTP. Any app with an "OpenAI compatible" provider works with three settings:

| Setting | Value |
| --- | --- |
| Base URL | `http://<the PS5's address>:8081/v1` (http, not https, ending in `/v1`) |
| Model | the id that `http://<the PS5's address>:8081/v1/models` shows, such as `Qwen2.5-Coder-7B-Instruct-Q4_K_M` (the server answers with the model it has loaded whatever id is sent) |
| API key | one of your keys (Settings > Apps > New key). With "Require a key" off any text works, and a real key still shows the app by name on Home |

Settings > Apps > Instructions > "Set up an app" shows these on the TV, filled in. A DHCP reservation in your router keeps the address from changing.

`/v1/models` also gives the loaded model's `context_length`: tell your app that number as the context window, so it trims the conversation before the PS5 has to refuse it. Coding tools send a long first prompt (10,000 to 15,000 tokens of instructions and tool descriptions), so serve with as much context as fits (the default). It lists the model's sampling settings under `settings` too: the `temperature`, `top_p` and `top_k` a request gets when it sends none, which for a checked model are the ones that did best on the PS5. Leave them out of your app's config to get them.

Endpoints: `/v1/chat/completions` (streaming, tool calling), `/v1/completions`, `/v1/models`.

## API keys

Settings > Apps > **New key**: name it after the app, and the key shows once on its row, like `ts5-k7qp2-wx9md-3hf8r-tz6na` (no i, l, o or u, so it reads off the TV without mix-ups). Type it into the app then; the PS5 keeps only its SHA-256. Cross turns a key off and on, square twice removes it. With **Require a key** on, a request without a key that's on gets HTTP 401; the Chat page on the TV never needs one. `ts5 serve` on a computer takes one key from `TS5_API_KEY`.

## Kilo Code

Kilo Code reads custom providers from `~/.config/kilo/kilo.jsonc`:

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

Then pick `tokenstation5/Qwen2.5-Coder-7B-Instruct-Q4_K_M` as the model. Put the `context_length` from `/v1/models` in `limit.context`. Leave `reasoning` out for the Llama and Qwen 2.5 models; only Qwen 3 thinks.

## Continue, Cline, Roo Code, Open WebUI, aider, opencode and others

Choose the "OpenAI compatible" provider (or "OpenAI" with a custom base URL) and fill in the three settings. Where the app asks for a context window, give it the `context_length`; for max output tokens, 4,096 to 8,192 is plenty. Native tool calling works with every checked model.

## Which model for what

- **Agent work** (editing files, then running them): Qwen 3 8B. On the PS5 it took every step of a write-then-run task; Qwen 2.5 Coder 7B wrote the file and stopped in 4 of 4 runs.
- **Qwen 3** thinks before it answers; the thinking comes back apart from the answer as `reasoning_content`. Skip it with `"chat_template_kwargs": {"enable_thinking": false}` or `/no_think` at the end of the message. A reply capped under 1,024 tokens never thinks, so a request for a short title still gets an answer.
- **Llama 3.1 8B and 3.2 3B** sometimes call a tool the question doesn't need (asked for a poem about rain with only a weather tool, Llama 3.1 8B asks for the weather). For plain chat with a Llama, leave the tools out, or use a Qwen model.
- **Mistral Nemo 12B** takes tools in its own format and passed every tool check on the PS5, with about 15k of context beside its weights.
- **Gemma 3 12B** takes tools in Qwen 3's format inside Gemma's own turns; 59 of 60 checks passed over three runs on the PS5, with about 29,700 tokens of context.

## When it doesn't connect

- The Logs page has a line for each request. A request that never shows there never reached the PS5: check the address, and that the app uses `http://`. An app that tries `https://` is named in the log.
- "maximum context length" errors: the conversation is longer than the model's context. Serve with as much context as fits, or start a new task in the app.
- Away from home, see [Tailscale](Away-from-home-with-Tailscale). Don't forward port 8081 on your router.
