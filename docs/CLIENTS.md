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
it. Use WireGuard, which runs on the PS5 itself, or Tailscale, through a
computer at home.

### WireGuard

The PS5 runs a [WireGuard](https://www.wireguard.com) tunnel of its own:
devices you let in reach the API through it, encrypted, and nothing else on
your network. WireGuard answers nothing without a key it knows, so the port
it opens gives nothing away. Each device makes its own key pair, and the PS5
only ever gets the public half.

1. On the PS5, make an API key (Settings, Apps, New key) if you have none,
   and turn on Settings, WireGuard. On your router, forward UDP port 9030
   (Settings, WireGuard, Port) to the PS5, and keep the PS5's address fixed
   with a DHCP reservation.
2. Install WireGuard on your computer, phone or tablet. At home, scan the
   code in Settings, WireGuard, Add a device with its camera, or open
   `http://192.168.1.50:8081/wireguard` (the PS5's address) in its browser.
   Give the device a name and press Ask to join. The page makes the
   device's key pair there, in the browser, and sends the PS5 only the public
   half. If the page stays blank, the device's VPN is probably sending the
   PS5's address away from home: turn it off while you add the device, or let
   it reach the local network.
3. The TV says the device asks to join: in Settings, WireGuard, press cross
   on it to let it in (square refuses it).
4. The page then has the tunnel ready: check your home's address on the
   internet (it looks it up), press Download the tunnel, and open the file in
   WireGuard (Import tunnel(s) from file on a computer; +, Create from file or
   archive on a phone or tablet). The tunnel reaches the PS5 alone
   (`AllowedIPs = 10.55.0.1/32`). Delete the downloaded file once WireGuard
   has it: it holds the device's private key.
5. Away from home, with the tunnel on, apps use `http://10.55.0.1:8081/v1`.
   Over the tunnel a key is always needed, whatever "Require a key" says.

The page's code comes from the PS5, so a PS5 someone had tampered with
could keep a key it makes. If you'd rather it never could, open "Rather have
your WireGuard app make the key?" on the page: the app makes the key pair, you
paste its public key, and copy the tunnel's settings in by hand.

The endpoint is your home's public address, or a dynamic DNS name for it.
Some internet providers share one public address among many homes (CGNAT),
and then a forwarded port can't be reached from outside: use Tailscale then.
The PS5's own private key is kept in `wireguard.tsv` in the title's folder,
which FTP reaches from your home network: if you think it got out, make a new
key pair in Settings, WireGuard, This PS5's key, and give each device the new
public key.

### Tailscale

[Tailscale](https://tailscale.com) (free for personal use) needs no port
opened on your router. Tailscale doesn't run on the PS5 itself, so a computer
that stays on at home shares the PS5's address with your devices as a
[subnet route](https://tailscale.com/kb/1019/subnets):

1. Install Tailscale on a computer that stays on at home (a PC, a Mac, a
   Raspberry Pi, a NAS) and on the computer, phone or tablet you'll use away, signed in
   to the same account.
2. On the home computer, advertise the PS5's address, for example
   `sudo tailscale up --advertise-routes=192.168.1.50/32` on Linux (which
   also needs IP forwarding turned on, see the link above). The Windows and
   Mac apps have the same setting.
3. In the Tailscale admin console, under Machines, approve the route for the
   home computer.
4. Away from home, with Tailscale on, apps use the same base URL as at home,
   `http://192.168.1.50:8081/v1`. Phones, Windows and Macs use approved
   routes on their own; another Linux machine needs
   `tailscale up --accept-routes`.

Instead of a subnet route, the home computer can also pass the API on with
[Tailscale Serve](https://tailscale.com/kb/1312/serve):
`tailscale serve --bg http://192.168.1.50:8081`. Your devices then use
`https://<home computer's name>.<your tailnet>.ts.net/v1` as the base URL,
with no route to approve and https from Tailscale.

Turn on "Require a key" in Settings as well, and keep the PS5's address fixed
with a DHCP reservation, since the route names one address. Settings on the
TV has the same steps for both, with this PS5's address filled in, under
Apps, Instructions: "Away from home with WireGuard" and "Away from home with
Tailscale".

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
