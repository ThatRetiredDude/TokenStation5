# Away from home with Tailscale

Don't forward port 8081 on your router. The API is plain HTTP, so a key would cross the internet unprotected, and anyone who finds the port can try it. Use Tailscale, which TokenStation5 runs itself: in the TV app on the PS5, and in `ts5` on Linux.

[Tailscale](https://tailscale.com) (free for personal use) joins your devices into a private network of their own, a tailnet, wherever each one is. TokenStation5 joins it as a device of its own: nothing to set up on your router, and it works on networks where no port can be forwarded at all (a building's shared internet, CGNAT).

## Set it up on the PS5 (TV app)

1. Install Tailscale on your phone, tablet or computer and sign in.
2. Make an API key first (Settings > Apps > New key): over Tailscale a key is always needed.
3. On the PS5, Settings > Away from home > Tailscale: turn it on. The TV shows a QR code. Scan it with your phone, sign in to the same Tailscale account, and approve TokenStation5.
4. Apps now use `http://100.x.y.z:8081/v1` (the PS5's tailnet address, shown in Settings) and one of your keys, at home and away. On devices with "Use Tailscale DNS" on, `http://tokenstation5:8081/v1` works too.

## Set it up on Linux (ts5-tui)

The Linux packages have the same client, built into `ts5 serve`: no `tailscale` package, no root, no network interface, nothing changed on the system.

1. Install Tailscale on your phone, tablet or computer and sign in.
2. In the terminal app, open **Access** (F6) and turn **Tailscale** on. A QR code of the sign-in link opens: scan it with your phone, sign in to the same Tailscale account, and it closes by itself once the machine is connected. The row then shows its name, its 100.x address and how many other devices your tailnet has.
3. **Connect a phone** (a row further down) makes a new key and puts it inside a QR code. Scan it with the phone's camera: the chat page opens, signed in, with nothing to type. The codes after it give the browser terminal (when it is on) and the address and key for an app, such as `http://100.101.102.103:8081/v1`.

```
     Tailscale         ‹ on ›
                       connected as tokenstation5 · 100.101.102.103 · 4 other devices · D signs out
     Connect a phone   a new key, as QR codes
                       Enter: a key as a QR code; the phone opens the chat page signed in, by HTTPS
```

- From the tailnet a key is **always** needed, even with "They need a key" off. Only the API's port is reachable through it.
- The QR code gives the 100.x address because it always works. The name (`http://tokenstation5.tailXXXX.ts.net:8081/v1`) needs MagicDNS on in your tailnet (admin console, DNS) and "Use Tailscale DNS" on in the phone's Tailscale app; without them a browser says "server cannot be found".
- **D** on the row signs out. Off leaves it signed in, so on again needs no QR code.
- The background service has no screen for the QR code: `curl -d '{"on":true}' http://127.0.0.1:8081/ts5/tailnet`, then `curl -s http://127.0.0.1:8081/ts5/tailnet` shows `sign_in_url` to open on your phone. `TS5_TS_NAME`, `TS5_TS_SERVER` and `TS5_TS_AUTH_KEY` go in `/etc/default/tokenstation`.
- Its files sit beside the keys (`~/.local/share/tokenstation/access/`, or `/var/lib/tokenstation/access/` for the service): `tailscale.conf` and `tailscale.tsv`, readable by their owner only. `tailscale.tsv` belongs to that machine. Copied to another one, Tailscale refuses it as copied keys: ts5 then starts over as a new device and shows a new sign-in QR code (remove the old one in the admin console).

### HTTPS: secure pages on the phone

With HTTPS the phone treats the chat page and the browser terminal as secure pages: no "not secure" warning, and the phone's passwords (iCloud Keychain, Google Password Manager) can keep the key and fill it in. TokenStation5 gets the certificate by itself, for its own name on your tailnet, from Let's Encrypt, the same way Tailscale's own `tailscale cert` does. Each machine gets its own; nothing is shared or shipped in the package. It needs three switches, all on your side:

1. **Tailscale on** in the terminal app (Access, F6), connected.
2. **MagicDNS** and **HTTPS Certificates** on in your Tailscale admin console, on its **DNS** page.
3. **Use Tailscale DNS** on in the phone's Tailscale app (on by default), so the phone can find the name.

A minute or two after the admin console's switch, the Connect a phone row says "by HTTPS" and its codes open `https://tokenstation5.YOUR-TAILNET.ts.net/`. Until then the row says which switch is missing and the codes use the http address.

- It answers HTTPS on the name at port 443, and on the API's own port too (`https://tokenstation5.YOUR-TAILNET.ts.net:8081/v1`), so an app can be given either, and a browser set to try HTTPS first still opens an old `http://…:8081` link by name.
- The certificate lasts 90 days and is renewed by itself 30 days before it ends. It and its key sit beside the keys (`tailscale-https.crt`, `tailscale-https.key`, and the Let's Encrypt account's key in `tailscale-acme.key`), readable by their owner only.
- **A certificate's name is public.** Every certificate goes into the public certificate transparency logs, with its name: the machine's name and your tailnet's name (`tokenstation5.tail1234.ts.net`). Nothing else: no address, no port, no key. If you'd rather the machine's name say nothing, rename it **before** you turn HTTPS Certificates on: admin console, Machines, the machine's menu, Edit machine name (switch "Auto-generate from OS hostname" off first). TokenStation5 gets a certificate for the new name by itself.
- An address by number (`https://100.101.102.103:8081/`) gets no certificate: a certificate is for a name. A browser trying HTTPS there goes on with http.
- `TS5_TS_HTTPS=0` leaves HTTPS off (in the terminal app's `ts5.env`, or `/etc/default/tokenstation` for the background service).

### SSH from away

**SSH via Tailscale**, the row under Tailscale on the Access page, passes the tailnet's port 22 to the machine's own SSH server. Then `ssh you@100.101.102.103` and an editor's remote SSH (VS Code Remote-SSH, Zed, JetBrains Gateway) work from anywhere, the same as at home.

- Off by default. It needs an SSH server on the machine (`sudo apt install openssh-server`); the row says so when there is none.
- SSH's own login applies (your SSH keys or password), not the API's keys, and your tailnet's access rules still decide which devices get through. Nothing else on the machine is opened.
- The SSH server sees these logins as coming from `127.0.0.2` (in `who` and its log), so they stand apart from logins on the machine itself; ts5's log names the device each came from. To allow only SSH keys from the tailnet, add to `/etc/ssh/sshd_config`:

```
Match Address 127.0.0.2
    PasswordAuthentication no
```

## In a browser: the chat page and the terminal

`ts5 serve` answers a web browser too, at the address apps use, without `/v1`.

**The chat page.** Scan Connect a phone's code, or open `https://tokenstation5.YOUR-TAILNET.ts.net/` (with HTTPS on) or `http://100.101.102.103:8081/` on your phone (or `http://192.168.1.20:8081/` at home with Other devices on) and chat with the loaded model, its answers streamed, with its thinking when the model thinks. It asks once for one of your keys and keeps it, and your chats, in that browser only. Nothing to install on the phone.

**The browser terminal.** ts5-tui itself in a web page, for a laptop or tablet with no SSH client, or a computer that isn't yours.

1. On the Access page turn **Browser terminal** on. Once Tailscale is connected the row shows the address, such as `http://100.101.102.103:8081/ts5/terminal`, and **Enter** shows it as a QR code.
2. Open it on a device on your tailnet and give it one of your keys. ts5-tui opens, the same as in a terminal.

- Off by default. It opens only for your tailnet's devices and the machine itself, each time with a key that is on, never for the rest of the home network, and for at most four at once.
- It runs ts5-tui in tmux, so a closed tab or a lost signal leaves an answer running, and the page connects again by itself.
- A bar of keys under it gives a phone what its keyboard lacks: Esc, Tab, the arrows, F1 to F7.
- ts5's log names each device that opens one and how long it stayed. Turning it off closes the ones that are open.

The pages themselves need no key; everything they ask of the model does. The terminal is drawn by [xterm.js](https://xtermjs.org) (MIT), built into `ts5`.

## How it works

The PS5 can't run Tailscale's own app, so TokenStation5 has a small client of its own (on Linux too, so that it needs no root). It signs in to Tailscale's control server and finds a direct path to each device where it can (on the home network, over UDP port 9031); otherwise it goes through Tailscale's relay servers (DERP), which work on any network. Settings shows which devices are connected directly.

Only devices on your tailnet can reach it, and your tailnet's access rules apply: the PS5 checks them itself, as Tailscale's own apps do. Over Tailscale only the API's port is open; FTP and everything else on the PS5 stay on the home network. Signing out (square twice on the Tailscale row) signs the PS5 out of the tailnet; it then shows as expired in Tailscale's admin console, where you can remove it.

## Options in `ts5.env`

| Variable | What it does |
| --- | --- |
| `TS5_TS_SERVER` | another control server, such as a [Headscale](https://headscale.net) |
| `TS5_TS_NAME` | the PS5's name on the tailnet (`tokenstation5` by default) |
| `TS5_TS_HTTPS` | `0` leaves HTTPS off (Linux) |
| `TS5_TS_AUTH_KEY` | an auth key from the admin console, which signs in without the QR code. Make it a one-off key: the PS5 removes it from `ts5.env` once signed in |

## Keep in mind

The PS5's Tailscale keys are kept in `tailscale.tsv` in the title's folder, which anyone on the home network can read over the jailbreak's FTP server. Someone who copies it could pose as the PS5 on your tailnet until you remove the PS5 in the admin console. Keep the console on a network you trust.

Tailscale is a trademark of Tailscale Inc.; TokenStation5 is not affiliated with or endorsed by Tailscale. Signing in uses your own Tailscale account under Tailscale's terms.
