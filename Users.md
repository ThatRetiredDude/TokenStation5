# Users

*Linux edition, from 1.1.0.*

Give someone else their own account on your TokenStation, such as a family member or a friend. They get the chat page and the API with keys of their own, and only what you allow. They never get the terminal app (ts5-tui) or the browser terminal.

You are the **Owner**. You have everything, and every key made before users is yours, so upgrading changes nothing.

## Add a user

On the terminal app's [Access page](Using-the-terminal-app#access-f6) (F6), the list under the rows now shows users, each with its keys under it.

1. Press **U** and type their name, such as `Sam`.
2. Their page opens. Choose what they may do (below). Each change holds from their next request.
3. On the last row, **A key for Sam**, press **Enter** and name the device, such as `Sam's phone`. The key shows once, as QR codes: the phone's camera opens the chat page already signed in. Sam's codes never include the browser terminal.

On the list: **Enter** on a user opens their page again, **N** on a user makes another key of theirs (one per device), **P** pauses them (none of their keys gets in, their chats stay), and **D** removes them, their keys and their chats. A single key is turned off or removed as before.

Adding a user switches **They need a key** on, and it stays on while there are users: without it, a device on your network could get in with no key and be treated as you. To switch it off again, remove the users first.

## What a user may do

| Setting | What it does |
| --- | --- |
| Preset | **Chat only**, or **Chat + web search** |
| Web search | TokenStation's own web search, when a chat needs it |
| Apps' own tools | The tools an app sends, such as an editor's file and command tools, which the app runs on the user's own machine. On by default, so an editor or any OpenAI-compatible app works with the user's key |
| Models | From 1.2.0: the models they may use. **All** by default. Enter picks some. With another model loaded, their chat is refused and told which models are theirs |
| May switch model | From 1.2.0: whether they may load one of their models themselves: the model's name on their chat page becomes a menu. Everyone's chats then use it |
| You read the chats | Whether you can read their chats. Choose it when you make the user. The user sees it under **Controls** on the chat page, so it is never a secret |
| Tokens a day | Prompts read plus replies, counted per day. Once it is used up, apps get HTTP 429 until tomorrow (a request already running finishes) |
| Longest reply | A cap on every reply, whatever the app asks for |
| Last day | From 1.2.0: after this date none of their keys gets in; their chats stay. ←→ moves it a week at a time |
| Paused | None of their keys gets in |

## When a key is refused

When a switched-off key, a paused user's key, or a key of someone past their last day tries to get in, your log says which key and whose, once a minute per key. For example: `ts5: access: 192.168.1.20: the key Sam's phone is Sam's, whose last day was 2026-12-01: refused`.

Everything on a user's page can also be set from your phone: see [Phone controls](Phone-controls).

## Kept apart

Users can't reach each other. Each has only their own chats and settings, and nobody but you learns who else has an account. The model's prompt cache is never shared between users: when a request comes from a different user than the last one, it starts fresh, so no speed or usage number an app sees can tell one user what another asked. The cost is that the first request after someone else's reads its whole prompt again.

The browser terminal and the terminal app are yours alone, and so are the model's own tools there (files, the terminal, the browser, GitHub). A user's key gets the chat page and the API only.

Two things to know:

- Anyone using the PS5 itself, or with a shell on it, counts as you.
- On a computer someone else uses, don't stay signed in to the chat page with your own key: the browser keeps it, so they would be signed in as you.

## Their chats

A user's chats from the chat page are kept on the TokenStation under that user, so the same chats are on every device with one of their keys. Nobody else sees them. You can read them only if **You read the chats** is on: the chat page's **Chats** list then shows "Other users who let you read their chats", and their chats open read-only. Each time you read them, your log notes it.

All of a user's chats together take at most 256 MB on the machine; past that, the chat page says the chat wasn't saved and asks them to delete some.

The first time a browser signs in with **your** key, any chats it kept in the browser from before users move to your account. Another user's key never takes or shows them, so a shared family computer stays apart.

What a user searches the web for is theirs too: your Logs say that a search was made for them, not what it was for (unless **You read the chats** is on for them).

## Reaching it from away: share by Tailscale

*From 1.1.1.*

At home, a user's phone reaches the TokenStation on your network. From anywhere else it needs [Tailscale](Away-from-home-with-Tailscale), and there are two ways to let them in. Pick one per person:

| | **Only this PS5** (recommended) | **Join my tailnet** |
| --- | --- | --- |
| What it is | Tailscale's *machine sharing*: the PS5 is shared into their own Tailscale account | A plain invite: they become a member of your tailnet |
| What they reach | This PS5 only | Every device on your tailnet: your computers, NAS, cameras, unless you write rules in Tailscale yourself |
| Their account | Their own Tailscale account (free) | They sign in to yours as a member |
| Your devices see theirs | No | Yes |
| Choose it for | Family and friends who chat or connect an editor | Someone you'd trust with every device, such as your own second account or a partner who shares your machines |

When unsure, choose **Only this PS5**. You can always invite them to the tailnet later.

### With the terminal app

1. **Once:** make a Tailscale access token. Sign in at [login.tailscale.com](https://login.tailscale.com), go to **Settings**, **Keys**, **Generate access token**. You must be an Owner or Admin of the tailnet. Tokens last up to 90 days; when one runs out, the next invite asks for a new one.
2. Turn **Tailscale** on on the Access page and sign in, if it isn't already.
3. Open the user's page (Enter on them). On **Share by Tailscale**, choose the way with ←→. The hint under it says what each one means.
4. Press **Enter**. The first time, paste the token; it is kept on the PS5 (beside the keys, readable by your user only) and never shown again.
5. A QR code of the invite appears. Sam scans it, installs Tailscale if needed, signs in and accepts.
6. Then the last row, **A key for Sam**, makes their key. Its QR code opens the chat page, signed in.

Removing the user (**D**) takes back their machine share and any invite not yet accepted. Someone who already joined your tailnet stays on it: remove them in Tailscale's admin console, **Users**. The terminal app says so when it happens.

From a script, the same:

```sh
ts5 share token set < token.txt             # once: the token on stdin, never on the command line
ts5 share new Sam ps5 NODE_ID               # only this PS5 (NODE_ID: the machine's id, in GET /ts5/tailnet as node_id)
ts5 share new Sam tailnet                   # join the whole tailnet
ts5 share list Sam
ts5 share revoke Sam                        # also done by ts5 users remove
```

### By hand, without giving ts5 a token

- **Only this PS5:** in Tailscale's admin console, **Machines**, the PS5's **⋯** menu, **Share…**, then copy the invite link and send it. They open it, sign in with their own account and accept. To end it later: the same menu, **Share…**, remove them.
- **Join my tailnet:** admin console, **Users**, **Invite users**, copy the link and send it. To end it: **Users**, their entry, **Remove user**.

Either way, make their key on the Access page as above and send them the chat page's QR code or link.

## Editors and other apps

A user connects an app as in [Connecting apps](Connecting-apps), with their own key. Their permissions hold there too: no web search unless allowed, the longest reply, tokens a day, and the app's own tools only if **Apps' own tools** is on.

## From a script

```sh
ts5 users new Sam                          # chat only
ts5 users new Ann --preset chat-web --owner-reads
ts5 keys new "Sam's phone" --user Sam      # a key of Sam's
ts5 users set Sam tokens_day=50000 max_reply=2000 web_search=on
ts5 users set Sam on=off                   # paused
ts5 users set Sam models=Qwen3-8B,Llama-3.1-8B   # only these (models=* for all), from 1.2.0
ts5 users set Sam may_switch=yes
ts5 users set Sam until=2026-12-01         # until=none for no last day
ts5 users                                  # the users, their permissions, tokens today, their keys
ts5 users remove Sam --forget              # Sam, Sam's keys and Sam's chats (from 1.1.1, first Sam's Tailscale shares)
```

For the always-on service add `--file /var/lib/tokenstation/access/keys.tsv`, as with `ts5 keys`.
