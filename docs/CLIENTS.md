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
it. Use Tailscale or WireGuard, which both run on the PS5 itself.

### WireGuard

The PS5 runs a [WireGuard](https://www.wireguard.com) tunnel of its own:
devices you add reach the API through it, encrypted, and nothing else on
your network. WireGuard answers nothing without a key it knows, so the port
it opens gives nothing away.

What your home needs: a router you can set up, and a public address of its
own. Where the internet provider shares one address among many homes (CGNAT,
common with mobile, 5G and satellite internet), or the internet comes through
a building's network you can't set up, nothing outside can reach the PS5, and
WireGuard can't work there. Settings, WireGuard, What you need says the same
on the TV.

First, on the PS5: make an API key (Settings, Apps, New key) if you have
none, and turn on WireGuard in Settings, WireGuard. On your router, forward
UDP port 9030 (Settings, WireGuard, Port) to the PS5, and keep the PS5's
address fixed with a DHCP reservation. Then add each device one of two ways:

| | Phone or tablet: the TV's code | Computer: the join page |
|---|---|---|
| Best for | iPhone, iPad, Android | Mac, Windows, Linux; any device whose private key should never leave it |
| Steps | Settings, WireGuard, Add a phone or tablet: name it, then in its WireGuard app tap +, Create from QR code, and scan the TV | Settings, WireGuard, Add a computer shows an address: open it in the computer's browser, press Ask to join, let it in on the TV, then Download the tunnel and import it (Import tunnel(s) from file; on Linux `sudo wg-quick up ./TokenStation5.conf`) |
| Its key pair | Made on the PS5, which keeps only the public half; the private one shows in the code and is then forgotten | Made in the device's browser; the PS5 only ever gets the public half |
| Watch out | Anyone who sees the TV while the code is up could copy that tunnel | The device must reach the PS5 at home: a blank page usually means its VPN is on |

Both make a tunnel that reaches the PS5 alone (`AllowedIPs = 10.55.0.1/32`).
Away from home, with the tunnel on, apps use `http://10.55.0.1:8081/v1` and
one of your keys (none needed with "Require a key" off). A phone runs one VPN at a time, so the tunnel takes over from
another VPN while it's on. Delete a downloaded tunnel file once WireGuard has
it: it holds the device's private key.

The join page's code comes from the PS5, so a PS5 someone had tampered with
could keep a key it makes. If you'd rather it never could, open "Rather have
your WireGuard app make the key?" on the page: the app makes the key pair, you
paste its public key, and copy the tunnel's settings in by hand. Keep the
threat in proportion, though: the tunnel leads only to the PS5, and the PS5's
FTP server lets anything on your home network read its folder, keys and all.
This is a sound setup for a home, not a hardened one.

The endpoint is your home's public address, which the PS5 looks up, or a
dynamic DNS name for it: type that in Settings, WireGuard, Home address.
To check it works, put a phone on mobile data and turn its tunnel on: within
seconds its row under WireGuard says Connected. If it never does, nothing
from outside is reaching the PS5's port: the forward is missing, or the home
has no public address of its own (see above).
The PS5's own private key is kept in `wireguard.tsv` in the title's folder,
which FTP reaches from your home network: if you think it got out, make a new
key pair in Settings, WireGuard, This PS5's key, and give each device the new
public key.

### Tailscale

[Tailscale](https://tailscale.com) (free for personal use) joins your
devices into a private network of their own, a tailnet, wherever each one
is. TokenStation5 joins it as a device of its own: nothing to set up on your
router, and it works on networks where no port can be forwarded at all (a
building's shared internet, CGNAT).

1. Install Tailscale on your phone, tablet or computer and sign in.
2. On the PS5, in Settings, Tailscale, turn it on. The TV shows a code: scan it with
   your phone, sign in to the same Tailscale account, and approve
   TokenStation5.
3. With Tailscale on, apps use `http://tokenstation5:8081/v1` (or the PS5's
   100.x address, shown in Settings) and one of your keys, at home and away
   (none needed with "Require a key" off: only your tailnet's devices can
   reach it).

The PS5 can't run Tailscale's own app, so TokenStation5 has a small client of
its own: it signs in to Tailscale's control server and sends its traffic
through Tailscale's relay servers (DERP), always, without direct connections
yet. That's plenty for streamed replies. Only devices on your tailnet can
reach it, and your tailnet's access rules apply. Signing out (square twice
on the Tailscale row) removes the PS5 from the tailnet.

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

For agent work (editing files, then running them), use Qwen 3 8B. On the PS5
it took every step of a write-then-run task; Qwen 2.5 Coder 7B wrote the file
and then stopped in 4 of 4 runs, without running it.

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
