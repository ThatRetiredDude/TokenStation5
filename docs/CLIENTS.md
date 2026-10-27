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

`/v1/models` lists the model's sampling settings under `settings` too: what
a request gets for the `temperature`, `top_p` and `top_k` it doesn't send.
For a model checked on a PS5 they are the ones that did best in its checks
(`known_good` says whose: `file` for this file, `model` for another download
of it, `family` for a model like it, `none` for the standard 0.7, 0.95 and
40), with the day of that check and how many of its checks passed. Leave
these settings out of your app's config to get them.

All of it is on the TV too: Settings, under Apps, Instructions, "Set up an app" shows the
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

## Away from home

Don't forward port 8081 on your router. The API is plain HTTP, so a key
would cross the internet unprotected, and anyone who finds the port can try
it. Use Tailscale, which runs on the PS5 itself.

### Tailscale

[Tailscale](https://tailscale.com) (free for personal use) joins your
devices into a private network of their own, a tailnet, wherever each one
is. TokenStation5 joins it as a device of its own: nothing to set up on your
router, and it works on networks where no port can be forwarded at all (a
building's shared internet, CGNAT).

1. Install Tailscale on your phone, tablet or computer and sign in.
2. On the PS5, in Settings, Tailscale, turn it on (make an API key first:
   over Tailscale a key is always needed). The TV shows a code: scan it with
   your phone, sign in to the same Tailscale account, and approve
   TokenStation5.
3. With Tailscale on, apps use `http://100.x.y.z:8081/v1` (the PS5's tailnet
   address, shown in Settings) and one of your keys, at home and away. The
   name, `http://tokenstation5:8081/v1`, works too on devices with "Use
   Tailscale DNS" on.

The PS5 can't run Tailscale's own app, so TokenStation5 has a small client of
its own: it signs in to Tailscale's control server and finds a direct path
to each device where it can (on the home network, over UDP port 9031),
otherwise it goes through Tailscale's relay servers (DERP), which work on any
network. Settings shows which devices are connected directly. Only devices on
your tailnet can reach it, and your tailnet's access rules apply. Signing out
(square twice on the Tailscale row) signs the PS5 out of the tailnet; it then
shows as expired in Tailscale's admin console, where you can remove it.

From a computer, `ts5.env` can also set `TS5_TS_SERVER` (another control
server, such as a [Headscale](https://headscale.net)), `TS5_TS_NAME` (the
PS5's name on the tailnet, `tokenstation5` by default) and `TS5_TS_AUTH_KEY`
(an auth key from the admin console, which signs it in without the code).

Settings on the TV has the steps too, under Apps, Instructions: "At home and
away with Tailscale".

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

Qwen 3 thinks before it answers, and the thinking comes back apart from the
answer, as `reasoning_content`. To skip it for a request, send
`"chat_template_kwargs": {"enable_thinking": false}` (as vLLM and llama.cpp
take it) or end the message with `/no_think`. A reply capped under 1,024
tokens (`max_tokens`) never thinks, so an app asking for a short title or a
line of code still gets an answer. Stop strings end the answer, never the
thinking.

For agent work (editing files, then running them), use Qwen 3 8B. On the PS5
it took every step of a write-then-run task; Qwen 2.5 Coder 7B wrote the file
and then stopped in 4 of 4 runs, without running it.

Given tools, Llama 3.1 8B and 3.2 3B sometimes call one the question doesn't
need: asked for a poem about rain with only a weather tool, Llama 3.1 8B asks
for the weather in London. Meta says the same of its 8B model. For plain chat with a
Llama, leave the tools out, or use a Qwen model.

Mistral Nemo 12B takes tools in its own format. On the PS5 it passed every
tool check of the model check, the write-a-file agent task too, with 15,360
tokens of context (what fits beside its weights).

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
