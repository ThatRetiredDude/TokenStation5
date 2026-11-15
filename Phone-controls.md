# Phone controls

*Linux edition, from 1.2.0.*

Everything the terminal app (ts5-tui) sets, from your phone: models, the fan, temperatures and memory, every setting and tool switch, users and their keys, and feedback. It works the running ts5-tui itself, so the TV and the phone always show the same thing.

## Open it

On the chat page, signed in with **your** key, tap **Controls** at the top (or **Controls** in the browser terminal's bar). A user's key never gets the machine's controls: their **Controls** button opens only what they may set for themselves, such as notifications and thinking.

It works from your home network and from your tailnet ([Away from home](Away-from-home-with-Tailscale)), never from the open internet. It always needs your key, even on the PS5 itself, so someone else using that computer isn't treated as you.

If ts5-tui isn't running, the page starts it in its tmux session, where the browser terminal finds it too.

## The chat page's home screen

The chat page opens on its home screen: the logo, the model (tap it to switch, if your key may), **New chat**, **Controls**, **TUI** (yours only: ts5-tui in the browser terminal) and your five latest chats. **New** brings it back. A notification's tap opens its own chat instead. The **TUI** button is also beside **Controls** at the top.

Whichever page you add to the Home Screen from (the chat page, Controls, or the browser terminal), the icon is the logo. Added from the chat page or Controls, it opens the chat page; added from `/ts5/terminal`, it opens ts5-tui.

## Models

- **Loaded:** the model serving now, its context length and speed. **Unload** frees the memory.
- **Downloaded:** load one, delete one (ts5-tui asks first, and the loaded model is never deleted).
- **To download:** the catalog's models, with their size. A download shows its progress and can be stopped.
- The context length and speed-or-length are the same rows as on ts5-tui's Settings page.

When ts5-tui asks something (load a model this small? delete this one?), the question comes to the phone and you answer it there.

## Machine

Temperatures, the fan's speed and its mode (the same four modes as ts5-tui), how busy the GPU is and its clock, and memory: GTT, VRAM and RAM in GiB.

## Settings

First this device's own rows: **Notifications**, whether a thinking model thinks first and shows its thinking, whether the key goes in the chat page's link, and saving or forgetting the key. Then every row of ts5-tui's Settings page, tool switches included. **‹** and **›** do what ts5-tui's arrows do. Rows that take words get a box. A key's row never shows its words on the phone: type a new one to replace it.

### Feedback

At the bottom of Settings. Write what went wrong (add an email if you'd like a reply) and choose whether the log's last lines and the chat open in ts5-tui go in. **Make the report** has ts5-tui make it exactly as `/feedback` does: keys, addresses, names and paths are taken out on the PS5 first. The phone then shows the whole report. **Send it** sends that report and nothing else, and the page shows its report id. **Save on the machine** keeps it in ts5-tui's feedback folder instead. **Cancel** drops it. Reports go only to TokenStation's developers, never anywhere public.

## Users

The users as ts5-tui's [Access page](Users) has them, and each user's page with the same rows: preset, web search, apps' own tools, models, may switch model, you read the chats, tokens a day, longest reply, last day and paused.

- Add a user by name, remove one (ts5-tui asks first).
- Turn a key off or on, or remove it.
- **New key:** name the device. Its QR codes show on the phone once: scan them with the other device. They are never shown on the TV or put on ts5-tui's clipboard, and closing them leaves no trace on the page.

Share by Tailscale stays on ts5-tui's Access page for now.

## One chat list

Your chats on the chat page are ts5-tui's chats. A chat begun on the phone opens in ts5-tui (**Ctrl+O**, or `/open`), and one begun in ts5-tui opens on the phone. While a chat is open in ts5-tui, what the phone adds to it shows there within a second. The chat page fetches the list again when it comes back into view, and every 15 seconds.

A tool's results and summaries stay in ts5-tui's chat and aren't shown on the phone. Users' chats stay theirs.

With the always-on service, your chat page chats stay apart from ts5-tui's for now.

## Notifications

When an answer finishes while the app is away (the iPhone's Home Screen app sent to the background, or the phone locked), you get a notification and a count on the app's icon. That works for the chat page and for ts5-tui in the browser terminal's app. The PS5 finishes a chat page answer even after the phone has left, and keeps it in the chat; a tap on the notification opens that chat, or ts5-tui.

- **Turn them on** once on each device: **Controls**, **Settings**, **Notifications**, **Turn on**, and allow them when asked. A test notification comes at once.
- **On an iPhone** they work in the Home Screen app (iOS 16.4 or later), not in Safari's tab: Share, **Add to Home Screen**, then open it from there. After updating to 1.2.0, remove the icon and add it again, so iOS takes the app's new settings. Added from `/ts5/terminal`, the app opens at ts5-tui.
- Each user gets notifications for their own answers only, on the devices they turned them on for. **Turn off** in the same place stops them for that device.
- They go through the push service the device uses (Apple's for an iPhone), encrypted for that device alone: the push service can't read them. The PS5 signs each one with its own key, kept beside the users.
- ts5-tui tells only this machine's own server that an answer finished, never a server elsewhere.
- **Stop** still stops an answer at once. Only an answer whose page went away goes on.

## The browser terminal on a phone

Its buttons (Esc, Tab, the arrows, the F keys, **A−** and **A+**) don't bring up the phone's keyboard, so it never covers them. **⌨** shows or hides the keyboard; while it's up, the button bar stays just above it.

## With the always-on service

The service can't see your home folder, so ts5-tui also listens in `/run/tokenstation`, which the package sets up for the `tokenstation` group. Your user must be in that group: the installer adds you (sign in again once after installing). If Controls says ts5-tui doesn't answer, start ts5-tui once on the TV or over SSH.

## What the log says

The server's log notes each change made from the phone and to which model, user or key, for example `ts5: controls: 192.168.1.20: load Qwen3-8B`. It never notes what a setting was set to, a key's words or what a report said.
