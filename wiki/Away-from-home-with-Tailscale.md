# Away from home with Tailscale

Don't forward port 8081 on your router. The API is plain HTTP, so a key would cross the internet unprotected, and anyone who finds the port can try it. Use Tailscale, which TokenStation5 runs on the PS5 itself.

[Tailscale](https://tailscale.com) (free for personal use) joins your devices into a private network of their own, a tailnet, wherever each one is. TokenStation5 joins it as a device of its own: nothing to set up on your router, and it works on networks where no port can be forwarded at all (a building's shared internet, CGNAT).

## Set it up

1. Install Tailscale on your phone, tablet or computer and sign in.
2. Make an API key first (Settings > Apps > New key): over Tailscale a key is always needed.
3. On the PS5, Settings > Away from home > Tailscale: turn it on. The TV shows a QR code. Scan it with your phone, sign in to the same Tailscale account, and approve TokenStation5.
4. Apps now use `http://100.x.y.z:8081/v1` (the PS5's tailnet address, shown in Settings) and one of your keys, at home and away. On devices with "Use Tailscale DNS" on, `http://tokenstation5:8081/v1` works too.

## How it works

The PS5 can't run Tailscale's own app, so TokenStation5 has a small client of its own. It signs in to Tailscale's control server and finds a direct path to each device where it can (on the home network, over UDP port 9031); otherwise it goes through Tailscale's relay servers (DERP), which work on any network. Settings shows which devices are connected directly.

Only devices on your tailnet can reach it, and your tailnet's access rules apply: the PS5 checks them itself, as Tailscale's own apps do. Over Tailscale only the API's port is open; FTP and everything else on the PS5 stay on the home network. Signing out (square twice on the Tailscale row) signs the PS5 out of the tailnet; it then shows as expired in Tailscale's admin console, where you can remove it.

## Options in `ts5.env`

| Variable | What it does |
| --- | --- |
| `TS5_TS_SERVER` | another control server, such as a [Headscale](https://headscale.net) |
| `TS5_TS_NAME` | the PS5's name on the tailnet (`tokenstation5` by default) |
| `TS5_TS_AUTH_KEY` | an auth key from the admin console, which signs in without the QR code. Make it a one-off key: the PS5 removes it from `ts5.env` once signed in |

## Keep in mind

The PS5's Tailscale keys are kept in `tailscale.tsv` in the title's folder, which anyone on the home network can read over the jailbreak's FTP server. Someone who copies it could pose as the PS5 on your tailnet until you remove the PS5 in the admin console. Keep the console on a network you trust.

Tailscale is a trademark of Tailscale Inc.; TokenStation5 is not affiliated with or endorsed by Tailscale. Signing in uses your own Tailscale account under Tailscale's terms.
