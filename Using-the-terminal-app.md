# Using the terminal app (ts5-tui)

`ts5-tui` is TokenStation5 in a terminal: a colorful, keyboard-driven app for when you SSH into the PS5, or sit at its text console. It chats with the loaded model and keeps every chat, downloads, switches, checks and tunes models, and shows the PS5's GPU memory, temperatures and fan live. Linux edition only.

![Chat in ts5-tui](images/terminal-app-chat.png)

It is a client of TokenStation's own API (port 8081), like any other app. It loads nothing itself and uses no GPU memory, so it can run beside the TV app or the API-only service.

> **Availability:** ts5-tui ships in the Linux packages from the [linux-latest](https://github.com/ThatRetiredDude/TokenStation5/releases/tag/linux-latest) build of 2026-11-10 on: `/usr/bin/ts5-tui` with the `.deb`, `.rpm` and Arch packages, and `~/.local/bin/ts5-tui` with the portable build. Already installed an older build? Run the [install command](Install-on-Linux#2-install-the-package) again to update.

## Start it

From your computer, SSH into the PS5, then run `ts5-tui`:

```sh
ssh ps5@<your PS5's address>
ts5-tui
```

The TokenStation title sweeps in for about a second (any key skips it), and you land in Chat.

| Command | What it does |
| --- | --- |
| `ts5-tui` | Talks to the server on this machine (`http://127.0.0.1:8081`). |
| `ts5-tui --url http://ps5.local:8081` | Talks to another TokenStation (`ps5.local` alone works too: http and port 8081 are assumed). |
| `ts5-tui --key KEY` | The key ts5-tui itself sends, when the server it talks to is on another machine and asks for one. On the machine it runs on, no key is needed. |
| `ts5-tui --theme Neon` | Colors: `Ocean` (default), `Neon`, `Sunset`, `Forest`, `Mono`. |
| `ts5-tui --no-intro` | Skips the opening title. |
| `ts5-tui --no-tools` | The model gets no tools at all in this run, whatever Settings says (see [Tools](#tools)). |
| `ts5-tui --mcp` | Starts your outside tools' programs, prints what each one offers, and ends them (see [Outside tools](#outside-tools-mcp)). |
| `ts5-tui --install-browser chromium` | Installs a browser for the model's browser tool without opening the app: `chromium` from your system's packages, or `obscura` into your home folder (see [Browser](#browser)). With `--dry-run` it only says what it would run. |
| `ts5-tui --tmux` / `--no-tmux` | Forces its own tmux session on or off (see [tmux](#tmux-ssh-that-can-drop)). |

`TS5_URL` and `TS5_API_KEY` in the environment do the same as `--url` and `--key`.

Run it **on the PS5** (over SSH) for the whole experience. Pointed at a PS5 from another machine with `--url`, chat works, but the sensors, models and logs it shows are that machine's, and loading a model is refused: models are switched where they are served.

## The screen

```
 ◆ TokenStation5   F1 Chat  F2 Models  F3 Status  F4 Settings  F5 Logs  F6 Access  F7 Connectors   09:52   ← pages
 ● Qwen3-8B-Q4_K_M · helper Qwen3-0.6B to 8k · chat 909/20k · 94.4 tok/s · GPU 5.7/13.6 GiB · die 80°C   ← live bar

   (chat, or the page you picked)

 Qwen3-8B-Q4_K_M  chat 909 of 20,480 tokens
╭────────────────────────────────────────────────────────────────────────╮
│ ts5> your message                                                      │   ← the prompt
╰────────────────────────────────────────────────────────────────────────╯
 Enter send  Ctrl+J new line  Ctrl+N new chat  …                             ← keys for this page
```

**The top bar stays put** whatever you do below it:

| Item | Meaning | Colors |
| --- | --- | --- |
| ● model | What the server has loaded. | Green: ready. Yellow: no model yet. Red: the server is offline. |
| helper | The helper model that is drafting for it right now, as the server reports it, and how far into a chat it helps (**to 8k**: the chat's first 8,192 tokens). **helper none**: this model has none. **helper not running**: it has one, but it isn't loaded; the model's details on the Models page say why. | Yellow: not running. |
| chat | Tokens this chat uses, of the most it can hold (the model's context as loaded). | Yellow from 75%, red from 90%. |
| tok/s | The last answer's speed: any app's when ts5-tui can read the server's log, else its own chats'. | |
| GPU | GPU memory used / total, in GiB. | Yellow under 2 GiB free, red under 1 GiB (the engine keeps 1 GiB back). |
| RAM | System memory used / total. | Same as GPU. |
| die | The chip's temperature (Tctl), the one the PS5's limiter watches. | Yellow from 78 °C, red from 83 °C. |
| fan | Fan speed. | |

On a narrow window the least needed items drop off first (fan, then RAM). When the die reaches **85 °C**, where the PS5 starts slowing its GPU, a red warning line flashes under the bar: `/fan max` helps.

**Seven pages** fill the middle. Switch with **F1** to **F7**, **Tab** (next) and **Shift+Tab** (previous), or **Alt+1** to **Alt+7**. On every page but Chat, **1** to **7** switch too, **q** quits, and **/** jumps to the prompt with a command started. From the prompt, `/chat`, `/models`, `/status`, `/settings`, `/logs`, `/access` and `/connectors` go to a page. In a window narrower than 99 columns the name in the corner shortens to TS5, and under 89 the pages are listed without their F keys, so all seven stay in view.

> **On a Mac**, macOS keeps the F keys for itself (F3 is Mission Control). Use **Tab** / **Shift+Tab**, **1** to **7** on the pages, the commands above, or **fn+F1** to **fn+F7**. **Option+1** to **Option+7** work when the terminal sends Option as Meta: in Terminal, Settings → Profiles → Keyboard → *Use Option as Meta key*; in iTerm2, Profiles → Keys → *Left Option key: Esc+*.

## Chat (F1)

Type at `ts5>` and press **Enter**. The answer streams in with its formatting: headings, lists, quotes, and code blocks with highlighting. A thinking model's reasoning (Qwen 3, Qwen 3.5, DeepSeek R1, Gemma 4) is folded away: while it thinks you see a pulsing **thinking…** line with its last words, then **▸ Thought for 4.1 s**. **Click that line** to read what it thought, and click it again to fold it; **Ctrl+T** unfolds every answer's thinking at once (and folds it again). A thinking that doesn't end is ended for the model, which then answers from what it has: one that goes round in circles, one that would leave no room for the answer, and one past 8,192 tokens. Its last line then says so in square brackets. Under each answer: its speed, tokens, prompt speed and time, and a **[copy]** mark (see [The mouse, and copying](#the-mouse-and-copying)).

| Key | What it does |
| --- | --- |
| Enter | Send. While an answer is coming, Enter **steers** it: the model reads your note as soon as it can and goes on from there (see [Messages while it answers](#messages-while-it-answers)). |
| Tab | While an answer is coming: **queues** the message instead, sent by itself when that answer is done. |
| Ctrl+J or Alt+Enter | A new line in the message. Pasting several lines works too. |
| ↑ / ↓ | The messages and commands you typed before. With messages queued and nothing typed, ↑ takes the last queued one back to change it. |
| Esc | Stops an answer (Ctrl+C too). With no answer running, clears what you typed; pressed again, drops queued messages. |
| Ctrl+N | A new chat. The one you leave is kept (see [Saved chats](#saved-chats)). |
| Ctrl+T | Show or hide the thinking, of every answer. A click on one answer's **Thought for** line does it for that one. |
| Ctrl+Y | Copy the last answer (see [The mouse, and copying](#the-mouse-and-copying)). |
| PgUp / PgDn | Scroll the chat a page. Ctrl+↑ / Ctrl+↓ scroll a line, and so does the mouse wheel in tmux. End (with nothing typed) goes back to the bottom. |
| ← → Home End | Move in what you're typing. Ctrl+← / Ctrl+→ by word. |
| Home / End | Start / end of the line. Ctrl+E is End too. Ctrl+A is Home outside ts5's tmux session; inside it Ctrl+A is tmux's own key, so press it twice, or use Home. |
| Ctrl+U / Ctrl+K / Ctrl+W | Delete to the start / to the end / the word before. |
| Ctrl+Q | Quit (Ctrl+C too, when no answer is running). A model ts5-tui loaded is unloaded and its GPU memory is free again. While a tuning, a check or a download runs, it asks first. |

### Messages while it answers

You don't have to wait for an answer to finish. Type while the model is still at work, then choose:

- **Enter steers it.** The model reads your note as soon as it can and goes on from there: "make it shorter", "use Python instead", "stop, the file is in src/". While it writes, it stops where it is: what it wrote stays, marked **your note came in here**, and it goes on from your note. While a tool it called runs (a command, a page), the note waits over the prompt as **your note** and goes in once that step is done, so the model sees the tool's result and your note together. Your note shows in the chat with **while it worked** beside it.
- **Tab queues it** instead: shown over the prompt as **queued**, and sent by itself the moment the answer before it is done. Queue as many as you like; they go in order, one answer each.
- **↑** with nothing typed takes the last queued message back into the prompt, to change it or (Esc) throw it away.
- **Esc** stops the answer that is coming. A note that hadn't gone in yet moves to the front of the queue. The queue then waits instead of firing: **Enter** sends it on, **Esc** drops it. The same when an answer fails (the server went away).
- Commands (`/think off`, `/model …`) still run at once. A new chat empties the queue.
- Steering works with every model: the note is simply your next message, put where the model will read it next. Saved chats keep the notes where they came in.

### A chat that fills up

The top bar always shows how full the chat is (**chat 7.8k/32k**), and the line over the prompt reads "chat N of M tokens": yellow from 75%, red from 90%.

A chat grows until the model's memory for it (its context) is nearly full. Then, before your next message goes out, the model writes **a detailed summary of the chat so far**, and the chat goes on from that summary and its last messages, with room again. Nothing is lost from the screen or the saved chat: the summary is a folded block in the chat, and a click opens it. An answer that ran into the end of the context gets a note and a summary too; ask the model to go on.

- `/compact` writes the summary now. `/compact keep the file names and the error text` says what must not be lost.
- **Esc** stops a summary and leaves the chat as it was.
- Settings → **Full chat**: **stops** turns this off. A full chat then says so under the answer that was cut, with the numbers, and **Ctrl+N** starts a new one (the full one stays saved). For a model loaded at a fixed context, the note adds that Settings → Context → auto gives it more.
- A single message that is longer than the model's whole context can't be summarized away: it comes back to the prompt with a note.

If an answer hit your **Answer length** cap instead, the note under it says that.

### Tools

The model in a chat has tools it can reach for by itself:

| Tool | What the model can do with it | When you get ts5-tui |
| --- | --- | --- |
| **search** | Search the web (the whole web with a key; without one an encyclopedia: Wikipedia, Grokipedia or both). | on |
| **pages** | Read a web page you give it a link to. | on |
| **terminal** | Run a command on the machine ts5-tui runs on. | on, asks first |
| **memory** | Keep notes between chats. | on |
| **files** | Read files and folders, and change files. | off |
| **browser** | Open web pages and click through them. | off |
| **GitHub** | Read and change your repositories: files, branches, pull requests, issues, CI runs. | on once you connect GitHub ([Connectors](#connectors-f7)), changes ask first |
| **outside tools** | Use the tools of other programs you add (MCP). | none added |

You decide what it may do:

- **Each tool has its own switch**: Settings → **Tool: search**, **Tool: pages** and so on, or `/tools terminal off`. `/tools` lists them with what is on.
- **One switch turns them all off**: Settings → **Tools**, or `/tools off`. The model then gets nothing but the chat. `/tools on` brings back the ones you had on. `ts5-tui --no-tools` does the same for one run.
- **The terminal, the files, the browser, GitHub and each outside program have three steps**: **off**, **on, asks first** (you see what the model wants to do and say yes or no) and **on, never asks**. Change them in Settings, or with `/tools terminal allow` and `/tools terminal ask`. `/tools ask` brings every question back at once.

In the chat a tool call is a line of its own (**⚙ $ ls -la**, **⚙ reads https://…**, **⚙ remembers: …**, **⚙ searches the web: …**) with a line under it saying how it went (**▸ exit 0 · 12 lines · 0.3 s**). Click that line to see exactly what the model got back.

**A question looks the same for every tool.** Nothing has happened yet; you see what the model wants to do and three answers:

`Run it?  ‹ Run it ›  Every command in this chat  No`

- **Enter** says yes. **Esc** says no: the model is told you said no, and answers in words instead of trying again another way.
- **→** moves the answer to the middle one (no more questions of that kind until a new chat; `/tools ask` brings them back) or to **No**, then Enter.
- Letters never answer the question. You can keep typing your next message while it shows; it waits in the queue as always.

**Web search.** Ask for something to be looked up: "look up on Wikipedia which river Bratislava lies on". With no key the model searches Wikipedia: it gets the top articles' titles, links and opening text, and it is told that this is an encyclopedia, with no news and no live data, and when the articles it got are only a loose match. Settings → **Search source** (or `/tools search wikipedia`, `grokipedia`, `both`) picks the encyclopedia: **Wikipedia** (the default), **Grokipedia** (xAI's encyclopedia, written by its AI), or **Both**, which gives the model half its results from each and tells it that two were searched. When Grokipedia can't be searched, that search goes to Wikipedia and the result says so. With a [Tavily](https://tavily.com) key (Settings → **Web search key**) it searches the whole web. Each search is a line in the chat, **⚙ searches the web: Bratislava river**, with what came of it under it (**5 results (0.5 s, Wikipedia)**).

**Web pages.** Give it a link: "read https://example.com/post and sum it up". It gets the page as plain text: the article where the page marks one, without the site's menus and sidebars, the first 8,000 characters. Pages that only show something in a browser (built by scripts), files, images and PDFs can't be read, and the model is told why. (With the browser tool on, the browser reads pages instead.)

**Terminal.** When the model wants to run a command you see the command first, with the question above.

- A command runs in bash (in sh only where the system has no bash), in the sandbox (below), in the folder ts5-tui was started in, with no keyboard to read from (a program that waits for typing gets nothing), for 2 minutes at most. **Esc** stops one that is running.
- The model gets the output and the exit code. Of long output it gets the start and the end. When a command with `grep` in it ends with code 1 and prints nothing, the model is also told that grep found no match, so it doesn't take that for a broken command before it. When `curl` piped into grep, awk or sed prints nothing, it is told that HTTP/2 header names come in lower case (`grep -i` matches either).
- Commands, tool calls and code are written at the model's own steady temperature, however high **Settings → Temperature** is (see [Settings](#settings-f4)).

**The sandbox.** The model's commands run in [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`, your system's own package; the TokenStation packages recommend it). It is on by default: **Settings → Sandbox**, or `/sandbox`. In there a command:

- sees the system **read-only**, so nothing in it can be installed or broken, and `sudo` can't work;
- sees **none of your home folder** (SSH keys, browsers, ts5's keys, models and chats), no other home folder, no mounted drive, none of the sockets in `/run` (your SSH and desktop agents), and none of your environment variables;
- can change **one folder of yours**: the one ts5-tui was started in. So start ts5-tui in the project you want the model to work on. Started in your home folder itself it gets none, and works in its own home;
- has a **home folder of its own, the toolbox** (`~/.local/share/tokenstation/sandbox/home`), kept between chats. `pip install`, `npm install -g` and a program put in `~/bin` go there, so the model fetches a tool once and uses it again. `/sandbox folder DIR` (or **Settings → Sandbox folder**) moves it to `DIR/tokenstation-sandbox`, on a faster or roomier drive say: one with Unix permissions (ext4, btrfs, xfs; not exFAT or NTFS). `/sandbox folder default` brings it back. While that drive isn't mounted, no command runs;
- has the network, unless you say `/sandbox no-network`;
- sees only its own processes, and whatever it starts ends with it;
- is the first thing the kernel ends when memory runs out, before the model's server (this one also with the sandbox off).

The question before a command says which it is: "the model wants to run a command **in the sandbox**" or "… **as you (no sandbox)**". The sandbox uses no memory of its own and starts in milliseconds. It shares the kernel with everything else, so it is a strong fence, not a second machine.

`/sandbox` says where commands run and how big the toolbox is. `/sandbox reset` empties the toolbox, and `/sandbox off` runs commands as you again, with all your files. Without bubblewrap (`sudo apt install bubblewrap`), or where the system doesn't let it run, Settings says why and commands run as you. Ubuntu 24.04 and later stop it with AppArmor; the [Linux README](https://github.com/ThatRetiredDude/TokenStation5/blob/linux-package/linux/README.md) has the four-line profile that lets bubblewrap in. Debian (the PS5) and Fedora need nothing.

**Memory.** Say "remember that I write Rust and like short answers" and the model saves a note. It reads its notes at the start of every chat.

- `/memory` lists the notes, `/memory forget 2` removes one, `/memory clear` removes all, `/memory TEXT` adds one yourself.
- They are plain lines in `~/.local/share/tokenstation/memory.md`; edit the file if you like.

#### Files

Off until you switch it on: Settings → **Tool: files**, or `/tools files on`. The model can then read a file, list a folder, change part of a file, or write a whole one: "the greeting in src/main.c is misspelled, fix it".

- **No change is made unseen.** Before anything is written you see the lines that go out (red, `-`) and the lines that come in (green, `+`), with the lines around them, and the question **Change it?** (or **Write it?** for a new file). The middle answer is **Every change in this chat**.
- **`/undo` takes the last change back** (a new file is removed again), as long as the file is still as the model left it.
- **Reading is asked too** (**Read it?**, or **List it?** for a folder), until you choose **Every file in this folder, this chat**. A hidden file (a name that starts with a dot) and a file outside the folder ts5-tui was started in are asked for every time.
- Paths start at the folder ts5-tui was started in, the same one the terminal's commands run in. So start ts5-tui in the project you want the model to work on.
- Text files up to 8 MiB. A long file is read in parts.
- A change has to name the text it replaces exactly. When the model gets it wrong it is told the line of the file that comes closest, and its most common slip (a line break sent where the file's code has `\n` written out) is read as it was meant; you see the lines that change either way.

#### Browser

Off until you switch it on: Settings → **Tool: browser**, or `/tools browser on`. The model can then open a web page, follow its links, type into its search box and press its buttons: "open en.wikipedia.org, search it for Bratislava and tell me the city's population".

- The model gets the page as text with every link, text box and button numbered (`Danube[3]`, `[4: text box "Search"]`, `[5: button "Go"]`) and works by the numbers. A long page comes in parts.
- **Reading a page and following a link need no question. Pressing a button, typing into a page, and opening an address on this machine or your home network are asked first** (**Do it?**, with **Everything on pages in this chat** in the middle). Password boxes are never offered to the model.
- **What shows the pages.** If Chromium (or Chrome, Brave, Edge) is installed, ts5 starts one without a window, with a fresh profile that is deleted afterwards, so the model never sees your own browser's logins. With no browser installed, ts5 reads pages itself: quick and light, but a page built by scripts shows little and only simple forms work. Settings → **Browser**, or `/browser auto`, `chromium`, `obscura`, `text` or a program's path, chooses; `/browser` alone says what is installed and in use.
- **What sites see.** Chrome as it is now on a MacBook (Chrome 158 on macOS 27.2.1, November 2026), from the browser and from ts5's own page reader alike: that Chrome's User-Agent and its other headers, and Chromium tells a page's scripts the same (Obscura tells them Windows when they ask for the platform). Nothing names ts5 or a model. The fonts, the graphics and the connection are the machine's own. `TS5_BROWSER_AGENT=linux` makes it Chrome on Linux, `TS5_BROWSER_AGENT=own` leaves each its own words. The searches name ts5, as services made for programs ask. A site that tests for a person (a captcha, a "checking your browser" page) stays closed, and the model says so.
- **No browser installed?** The first time you switch the browser tool on without one (none found on this machine, and none named with `/browser` and its path), ts5-tui asks which to install (arrows and Enter; the choice starts on **Neither**, so an Enter meant for something else installs nothing): **Chromium** from your system's packages (`apt-get`, `dnf`, `pacman` or `zypper`; sudo asks for your password; on Debian 13 it is the small `chromium-headless-shell`), **Obscura** into your home folder (a 72 MB download from its makers' GitHub release, 150 MB once unpacked, checked against a SHA-256 built into ts5, no password), or **Neither**. ts5-tui steps aside while the install runs and comes back when you press Enter. It asks once; `/browser install` asks again, and `ts5-tui --install-browser chromium` or `obscura` does it without the app.
- **Obscura** ([its page](https://github.com/h4ckf0r0day/obscura)) is a lighter browser, used only when you choose it (or install it from the question above). It is young. It runs a page's scripts on the thread that answers ts5, so it says nothing while a heavy page's scripts are at work. ts5 reads a page before those scripts start, and waits for a busy Obscura before it opens the next page. On a PS5, 12 of the 13 pages we tried opened in Obscura in 2 to 20 seconds, most of them under 10 (Chromium: 1 to 5); one, Wikipedia's article on the Black Sea, leaves Obscura 0.2.6 without an answer for good. A page that keeps Obscura silent for 20 seconds (`TS5_BROWSER_PATIENCE_S` sets the seconds) is shown in ts5's own text view, with a line that says so, and the next page is Obscura's again. What the model types or presses on a page that left Obscura stuck goes through the text view too. It does not open addresses on this machine or your home network.
- On the PS5, Chromium took 330 to 400 MiB of RAM beside a loaded model (650 MiB on the heaviest page we tried), and no GPU memory.
- While the browser is on it stands in for the **pages** tool.
- A site that tests for a person (a captcha, a "checking your browser" page) stays closed, and the model says so.

#### GitHub

Once GitHub is connected on the [Connectors](#connectors-f7) page, the model can work in the repositories you picked there: "read the README of my website repo and open an issue for the broken link", "why did the last CI run fail?", "make a branch, fix the typo in index.html and open a pull request".

- It calls GitHub's API as you, through the TokenStation5 app, so it reaches only those repositories, and only what the app may do: code, pull requests and issues (read and write), and CI results (read). It can't change workflow files or a repository's settings.
- **A read goes at once** (a file comes as its text, a list of pull requests, a CI run's jobs). **A change waits for your yes**: making a branch, writing a file, opening a pull request or an issue, a comment. You see what it would send, a file's new text a line at a time, and **Do it**, **Every GitHub change, this chat** or **No**.
- The chat shows each call as a line (**⚙ GitHub: GET /repos/you/website/readme**).
- Settings → **Tool: GitHub**, or `/tools github off`, `ask`, `allow`. The model never sees the sign-in itself.

#### Outside tools (MCP)

Other programs can give the model tools of their own: a calendar, a database, a code host, whatever the program is for. Such programs speak MCP (the Model Context Protocol), and there are many of them. Add one with the command line its own instructions give:

```
/mcp add NAME PROGRAM [ARGUMENTS]
```

so if a program's instructions say to run `npx -y ITS-PACKAGE ~/notes`, that is `/mcp add notes npx -y ITS-PACKAGE ~/notes` (`notes` is the name you give it; `KEY=VALUE` words before the program are passed to it as its environment). ts5-tui starts the program, asks what it offers, and the model can call those tools from then on.

- **Every call is shown to you first**: whose tool it is, what the tool says it does, and what it is called with (`⚙ notes: add  (outside tool: Adds a note.)`, then a line for each value), and the question **Run it?**, with **Every call to NAME, this chat** in the middle.
- `/mcp` lists your programs, whether each is running, and how many tokens its tools take out of every chat. `/mcp NAME` lists one program's tools.
- `/mcp NAME off`, `ask` and `allow` are its three steps (also a row each in Settings). `/mcp NAME only TOOL TOOL …` gives the model just those tools, `/mcp NAME all` gives them all back, `/mcp NAME restart` starts the program again, `/mcp remove NAME` takes it out.
- **Tools cost context.** Every tool's description is sent with every message, so a program with fifty tools makes every chat shorter. On the PS5 a program with 13 small tools took 850 to 980 tokens out of every message, and 2 of them about 200. `/mcp` shows the cost; `only` cuts it down.
- The list is the file `~/.config/tokenstation/mcp.json`, in the layout other MCP apps use, so a list made for one of them can be pasted in (`/mcp reload` reads it again).
- A program runs as you for as long as ts5-tui does, and is ended when ts5-tui quits or the tools are switched off. It is not given ts5's API key or web search key. What it prints for itself is kept in `~/.local/share/tokenstation/mcp-NAME.log`.
- Only programs on this machine so far, not ones reached by a web address.
- `ts5-tui --mcp` checks your programs without a chat: it starts them, prints what each offers, and ends them.

Good to know:

- Every model can call tools, and how well is up to the model. This is what the PS5 runs showed, each model given the browser, the files and an outside tool:

  | Model | With tools |
  | --- | --- |
  | Qwen 3 8B and 14B, Qwen 3.5 9B, Qwen 3.8 9B | Did what was asked, every time. |
  | Bonsai 4B, 9B and 27B | Did what was asked. |
  | Gemma 4 12B, Gemma 3 12B, Qwen 2.5 Coder 14B | Did what was asked. |
  | Llama 3.1 8B | Files and outside tools: yes. On a web page it clicks the wrong links. |
  | DeepSeek R1 14B | Right about two times in three. It is asked in words, because its file has no tool format of its own, and sometimes answers from memory or only describes what it would do. |
  | Gemma 3 4B, Llama 3.2 3B, Qwen 2.5 Coder 7B | Mostly not: a wrong link, an edit that never matches, a number cut short. |

- **With a small model, keep the questions on.** In one test run a 4B model, asked to fix a typo, wrote a command that emptied the file instead. With the question on, that is a command you read and say no to.
- A web page, and what an outside program gives back, is text from a stranger and can try to talk the model into something. With the questions on, the worst case is something you say no to. With "never asks" or "every … in this chat" chosen, leave pages and programs you don't trust alone. An outside program itself can do whatever you can on this machine: add only ones you trust.
- A model that keeps calling tools gets twelve rounds for one message, then has to answer with what it has. Reading on down a long page or file is not counted. If it asks for more after that, the chat says so, and "go on" gives it twelve more.
- In a long run of tools the chat can fill up before the model is done. The earlier long results (pages, files) then give way: the model keeps their first lines, and the chat marks those calls **left out later, for room**. The model can ask for one again if it needs it.
- An answer that comes back empty is asked for once more by itself.
- The tools are ts5-tui's own. Other apps bring theirs (see [Connecting apps](Connecting-apps)).

### The mouse, and copying

| Do this | And |
| --- | --- |
| Click **▸ Thought for …** | That answer's thinking opens. Click **▾ Thinking** to fold it. |
| Click the line under a tool call (**▸ exit 0 · …**) | What the model got back from the tool opens. Click again to fold it. |
| Click **[copy]** under an answer | The answer is copied, as the model wrote it. |
| Click **[copy]** on a code block | That block's code is copied, without the box or the indent. |
| Press **Ctrl+Y**, or type `/copy` | The last answer is copied. `/copy code` copies its last code block, `/copy think` its thinking, `/copy all` the whole chat. |
| Drag over the chat | The whole chat opens as plain text, at the message you dragged on, to select from. On Logs, the log does. |
| Click a page's name in the top bar | Goes to that page. The wheel scrolls. |

**Where a copy goes.** A program in a terminal can't see your computer's clipboard, least of all over SSH, so ts5-tui does everything that can work and then shows you the text:

1. It asks the terminal to put the text on the clipboard. iTerm2 (with *Applications in terminal may access clipboard* on), kitty, WezTerm, Ghostty, Windows Terminal and VS Code do that. Apple's Terminal does not.
2. On a Linux desktop it also uses `wl-copy`, `xclip` or `xsel`.
3. It saves the text in `~/.cache/tokenstation/copied.txt`.
4. It shows the text alone on the screen, plain, and gives the mouse back to the terminal. **Select it with the mouse and copy as you always do** (Cmd+C on a Mac). You get exactly what the model wrote: long lines whole, code without bars or indent. The arrows and PgUp/PgDn move through a long text; **any other key goes back** to the chat.

With **Settings → Mouse: off**, the terminal keeps the mouse all the time: dragging selects what is on the screen, and Ctrl+T and Ctrl+Y do what the clicks did. With the mouse on, holding **Fn** (Apple's Terminal), **Option** (iTerm2) or **Shift** (most others) while dragging also selects what is on the screen.

### Saved chats

Every chat is saved after each answer, the way the TV app saves them: one file per chat in `~/.local/share/tokenstation/chats/` (or `$TS5_DATA_DIR/chats`). Nothing to do: quit, come back, and they're there.

- `/chats` lists them, newest first, with when and how many messages. The TV service's chats (`/var/lib/tokenstation/chats`) are listed too when your user can read them.
- `/chats TEXT` lists only the chats that have every word of TEXT.
- `/open N` opens number N from the list, and you go on where it ended. `/open TEXT` opens the newest chat that has TEXT. Going on with a TV chat makes a copy of your own.

### Commands

Type `/` and the matching commands show above the prompt. **Tab** completes the command, the model name after `/model`, `/tune`, `/check` or `/bench`, and the catalog name after `/download` (press Tab again to step through the matches).

![Commands as you type](images/terminal-app-commands.png)

| Command | What it does |
| --- | --- |
| `/model NAME` | Loads another model (see [Models](#models-f2)). Part of the name is enough. `/model` alone opens the Models page. |
| `/models` | The Models page. `/chat`, `/status`, `/settings`, `/logs` and `/access` go to the others. |
| `/download NAME` | Downloads one of the models tested on the PS5 (see [Downloading models](#downloading-models)): `qwen3-14b`, `qwen35-9b`, `qwen38-9b`, `deepseek-r1-14b`, `qwen3-8b`, `llama31-8b`, `qwen25-coder-7b`, `qwen25-coder-14b`, `gemma3-12b`, `gemma4-12b`, `bonsai2-27b`, `bonsai2-9b`, `bonsai2-4b`, `llama32-3b`, and the helpers `qwen3-0.6b`, `llama32-1b`, `gemma3-1b`. Tab completes the name. `/download` alone opens the list on the Models page. |
| `/download URL [FILE] [--sha256 HEX]` | Downloads any other `.gguf`. A `ts5 fetch URL FILE --sha256 HEX` line, as on [Models tested on PS5](Models-tested-on-PS5), works pasted at the prompt as it is. |
| `/search WORDS` | Looks on Hugging Face for other models that fit this GPU: `/search mistral 7b` (see [Searching Hugging Face](#searching-hugging-face)). |
| `/new` | A new chat (`/clear` too). The one you leave is kept. |
| `/chats [TEXT]` | Your saved chats, or the ones with TEXT (see [Saved chats](#saved-chats)). |
| `/open N` | Opens saved chat N from `/chats` (or `/open TEXT`: the newest with TEXT). |
| `/save [FILE]` | Saves this chat as Markdown, thinking included. Without a file: `~/tokenstation-chat-YYYYMMDD-HHMM.md`. |
| `/copy` | Copies the last answer (Ctrl+Y). `/copy code`: its last code block. `/copy think`: its thinking. `/copy all`: the whole chat. See [The mouse, and copying](#the-mouse-and-copying). |
| `/compact [WHAT TO KEEP]` | Has the model summarize the chat so far and go on from the summary, with room again (see [A chat that fills up](#a-chat-that-fills-up)). `/summarize` too. |
| `/tools` | Lists the model's tools and which are on (see [Tools](#tools)). `/tools search`, `pages`, `terminal`, `files`, `browser`, `github` or `memory` followed by `on` or `off` sets one. `/tools off` takes them all away, `/tools on` gives them back. `/tools terminal allow` stops the questions for that tool (`files`, `browser` and `github` too); `/tools ask` brings every question back. `/tools search wikipedia`, `grokipedia` or `both` picks the encyclopedia searched without a web search key. |
| `/browser` | Says which program shows web pages to the model. `/browser auto`, `chromium`, `obscura`, `text` or a program's path chooses one. `/browser install` offers to install Chromium or Obscura. |
| `/undo` | Takes back the last change the model made to a file. |
| `/sandbox` | Says where the model's commands run (see [Tools](#tools)). `/sandbox on`, `off`, `network`, `no-network` set it; `/sandbox reset` empties the toolbox; `/sandbox folder DIR` moves the toolbox to another folder or drive (`default` brings it back). |
| `/mcp` | Your outside tools' programs (see [Outside tools](#outside-tools-mcp)). `/mcp add NAME PROGRAM …` adds one, `/mcp remove NAME` takes one out, `/mcp NAME` lists its tools, `/mcp NAME off`, `ask`, `allow`, `restart`, `only TOOL …`, `all`; `/mcp reload` reads the list again. |
| `/memory` | The notes the model keeps between chats. `/memory forget N` removes one, `/memory clear` all, `/memory TEXT` adds one. |
| `/find TEXT` | Scrolls to the newest message containing TEXT. |
| `/tune [NAME]` | Runs `ts5-tune` on the loaded model (or NAME), live on the Models page. |
| `/check [NAME]` | The GPU check on the loaded model (or NAME): `ts5 check`, the GPU's answers against the CPU's. |
| `/bench [NAME]` | The speed test on the loaded model (or NAME): `ts5 bench`, prompt and answer tok/s. |
| `/usage` | The API's use over the last day and week, by app (its API key's name, else its address) and by model: requests, prompt and generated tokens, tok/s. From the TV or desktop app's `usage.tsv`. |
| `/access` | The [Access](#access-f6) page: who can use this TokenStation5 from another machine, and the API keys (`/keys` goes there too). |
| `/connectors` | The [Connectors](#connectors-f7) page: GitHub (`/github` goes there too). |
| `/think on`, `off`, `auto` | Thinking before answering, for models that can switch it (Qwen 3, Qwen 3.5, Gemma 4). `auto` leaves it to the model. |
| `/fan auto`, `cool`, `early`, `max` | The PS5's fan (see [PS5 hardware on Linux](PS5-hardware-on-Linux)). `early` ramps from 40 °C and is at full speed by 55 °C. |
| `/theme NAME` | Colors: Ocean, Neon, Sunset, Forest, Mono. Without a name, the next one. |
| `/status`, `/settings`, `/logs`, `/access` | Those pages. |
| `/help` | Every command and key, in the chat. |
| `/unload` | Stops the model ts5-tui loaded, so its GPU memory is free, without quitting. |
| `/quit` | Leaves (`/exit` and `/q` too, and plain `exit`, `quit` or `q`). A model ts5-tui loaded is unloaded (see [When you quit](#when-you-quit)). |

## Models (F2)

![The Models page](images/terminal-app-models.png)

Every `.gguf` on this machine, biggest first, with its size, its helper, and where its settings come from: **tested** (a model tested on the PS5: its settings are built in, and it needs no tuning), a date (tuned on this machine with `ts5-tune`), or **—** (the defaults). **●** marks the one loaded. Under them, **⇣ Download**: the tested models you don't have yet, and **⇣ Search**: what a Hugging Face search found. Below the list, the selected model's settings, each with what it costs: 16- or 32-bit prompt math, the helper, the chat memory (automatic: full precision, or 8-bit where that fits a longer chat) and, for the loaded one, its context.

| Key | What it does |
| --- | --- |
| ↑ ↓, PgUp PgDn | Choose. |
| Enter | Load it. On the model that's loaded: load it again, which is how a changed setting or a helper downloaded since takes hold (it asks first). On a catalog row: download it. |
| T | Tune it with `ts5-tune --models` (10 to 20 minutes). A tested model doesn't need it. T asks before it starts. |
| H | Download this model's helper, when it isn't here yet. |
| C | The GPU check (`ts5 check`): the model's answers on the GPU against the CPU's. C asks before it starts. |
| B | The speed test (`ts5 bench`): a 512-token prompt and 128 tokens of answer. B asks before it starts. |
| D | On a catalog row or a search result: download it. |
| S | Search Hugging Face for other models: type the words right there, Enter searches, Esc cancels. |
| X | Stop the download (it goes on from there next time). |
| R | Look for model files again. |

It looks in `/var/lib/tokenstation/models` (or the service's `MODELS_DIR`), `~/.local/share/tokenstation/models` (the desktop app's), `~/models`, and `$TS5_MODELS` if set. Files under 1.5 GiB are listed as helper size and aren't tuned.

**What Enter does** depends on what serves the API:

| Running | Enter |
| --- | --- |
| The API-only service (`tokenstation`) | Switches its model and restarts it, through the package's hardware helper. Your user must be in the `tokenstation` group (`sudo adduser $USER tokenstation`, then log in again). The model must be in the service's models folder. |
| Nothing | Starts `ts5 serve` with that model on `127.0.0.1`, with the context and the **Speed or length** choice from Settings. By default (auto context, **faster**) it keeps GPU memory for the model's helper: Qwen3 14B on the PS5 then gets a 20,480-token chat and writes about 58 tok/s. With **longest chat** it gets 32,768 tokens and no helper, about 42 tok/s. Quitting ts5-tui stops it and frees its GPU memory (Settings → **When you quit** keeps it serving instead), and `/unload` stops it without quitting. Its log is `~/.cache/tokenstation/tui-serve.log`. |
| The TV app (`tokenstation-tv`) | Nothing: pick the model on the TV's Models page. |
| Another program on port 8081 | Nothing: switch the model there. |

The top bar shows **loading…** with the seconds until the server reports the new model. Switching models stops the one that was loaded first, so its memory is free for the new one.

- **A model that doesn't load** (not enough memory, a file that isn't a model) says why in plain words, on the Models page and in the chat, and the model that was loaded before it is loaded again, so a failed switch doesn't leave you with none.
- **A Context set to a number** (Settings) never stops a load. Where the full-size chat memory for that many tokens doesn't fit beside the model, the engine uses its smaller chat memory instead (8-bit, as Context on auto does): a 14B at 32,768 tokens needs 6.0 GiB the full-size way and 2.4 GiB the smaller way. Where even that doesn't fit, the model comes up with the longest chat that does, and a note says so.

**Helpers** aren't loaded by hand. A helper is a small model that drafts tokens for a big one (the same text, faster), and it loads along with it.

- **The models tested on the PS5 know their helper**: Qwen3 0.6B for the Qwen3, DeepSeek R1 and Qwen2.5 Coder models, Llama 3.2 1B for the Llama ones. It downloads right after the model, into the same folder, and **H** fetches it for a model that's already here. Qwen3.5 9B, Qwen3.8 9B and the Bonsai models can't use a helper, and the Gemma 12B models run without one (Gemma 3 1B speeds up short answers and slows long prompts; T measures it on your machine).
- **For any other model**, **T** tunes it and picks the fastest helper among the small files of its own family here. Files from another family are left out, with the reason.
- **The top bar shows what is really running**, as the server reports it: **helper NAME to 8k**, **helper none**, or **helper not running** in yellow. The selected model's details say why it isn't running: not downloaded yet (H gets it), no GPU memory left for it at this context, or downloaded after the model was loaded (Enter loads the model again).

Enter on a helper-size file (under 1.5 GiB) asks first, because it would replace the main model.

### Downloading models

The list under **⇣ Download** is every model on [Models tested on PS5](Models-tested-on-PS5), best first, with the helpers last. Each is pinned to the exact upload that was tested, with its size and SHA-256. Models already on this machine aren't listed. Choose one and press **Enter** (or `/download NAME` from the prompt). A tested model's helper downloads right after it, by itself. The row shows a bar and the speed; the box below shows where the file goes, how much space is free there, what it's doing (connecting, downloading with the time so far and left, checking it's the exact file), and its steps ticked off. The top bar shows the download's percent and time left on every page.

- It downloads with `ts5 fetch` into a `.part` file and keeps the model only when its size and SHA-256 match, so a broken download never shows up as a model.
- It checks the free space first and won't start when the drive can't hold the file.
- **X** stops it. The `.part` stays, and downloading it again goes on from there. Quitting ts5-tui stops it the same way.
- It goes to the service's models folder when your user may write there, else to `~/.local/share/tokenstation/models` (or `$TS5_MODELS`). The API-only service only reads its own folder, so ts5-tui then shows the `sudo mv` command that moves the file there.

**Any other model**: `/download URL [FILE] [--sha256 HEX]`, or paste a `ts5 fetch` line from [Models tested on PS5](Models-tested-on-PS5) at the prompt as it is:

```
ts5> ts5 fetch https://huggingface.co/…/resolve/…/Model.Q4_K_M.gguf Model.Q4_K_M.gguf --sha256 5bc6f9…
```

It downloads the same way and shows under the catalog while it runs. FILE is a name only (it goes where downloads go); without it, the name at the end of the URL. Hugging Face page links (`/blob/`) work too. Without `--sha256` only the size is checked.

### Searching Hugging Face

For a model that isn't on the tested list, press **S** on the Models page and type what to look for, or type `/search mistral 7b` at the prompt. It asks Hugging Face for the most downloaded GGUF repos that match and keeps the files ts5 can run (the Llama, Mistral, Qwen and Gemma families, in the usual 4- to 8-bit file types) that fit this GPU's memory with room for a chat. Chat models come first, two files per repo at most.

- The line after **⇣ Search** says how it's going ("looking at model 3 of 8…"), then how many fit, and how many were left out because they're too big for this GPU or of a kind ts5 can't run.
- Left out without a count: repos that match your words only by their uploader's name, and files that aren't the model itself (a vision model's image encoder, `mtp-` side files). Embedding and reranking models count as a kind ts5 can't run: they don't chat.
- The results are listed under it, each with its size and who made the file. The box below shows the repo, the file, how long a chat it gets on this machine and where it goes.
- **Enter** downloads the selected one the same way as a tested model, pinned to the repo's commit and checked against the SHA-256 Hugging Face lists for it. A file that's already here says so, and Enter goes to it in the list.
- Two-bit files (Q2_0, the Bonsai models) are found too. Their other packings (PQ2_0, PTQ1_0) are left out: ts5 doesn't read those.
- A model found this way hasn't been tested on the PS5. Once it's here, **C** checks its answers on this GPU and **T** tunes it.

It's the same search as the TV app's Models page, so it needs the `tokenstation` app from the same package (`tokenstation --hf-search WORDS` prints what it finds), and the internet.

### Tuning, the GPU check and the speed test

**The models tested on the PS5 need no tuning.** Their settings (the helper, and how far ahead it drafts) were measured on a PS5 and are built in: their row says **tested**. Tuning is for other models: one from a `/download URL`, or a new model of a family ts5 runs.

**Tuning** (T), the **GPU check** (C) and the **speed test** (B) open a progress panel:

- **Working:** what it's doing right now, in plain words, with a spinner, the test answer it's on, the time so far and about how long is left (the estimate shows after the first test answer).
- A **checklist** of every step it will go through, ticked off as each one finishes, with its result and how long it took: for tuning, checking the model gives correct answers, measuring its speed on its own, trying each helper model of its own family, saving the best settings. Skipped steps say why (for example, a helper whose vocabulary doesn't match).
- At the end, **what it found**: for tuning, the helper it picked and the speed before and after ("It writes 46% faster with the helper …: 58.9 tok/s, up from 40.3 on its own, with the same answers."); for the check, whether the GPU's answers match the CPU's; for the speed test, how fast it reads prompts and writes. The same line goes into the chat.
- Under it, the tool's own lines, for a bug report.

While a job runs, the top bar says so in yellow (**tuning MODEL · step 3 of 7 · 2:13**) instead of **offline**, on every page. **Esc** hides the panel and the job goes on (T shows it again), **X** stops it (nothing is changed then), PgUp/PgDn scroll. The GPU check runs the CPU side too, which takes a few minutes the first time for a model and is saved after. They need the GPU's memory: a `ts5 serve` that ts5-tui started is stopped for them, and its model is loaded again when the job ends. A model can't be loaded while one runs. If the service holds a model, stop it first (`sudo systemctl stop tokenstation`). ts5-tune saves the results in your own `~/.config/tokenstation/tune/`. The API-only service runs as its own user and reads `/etc/tokenstation/tune/`, so copy the file there for the service (`sudo mkdir -p /etc/tokenstation/tune && sudo cp ~/.config/tokenstation/tune/<model>.gguf.env /etc/tokenstation/tune/`).

## Status (F3)

![The Status page](images/terminal-app-status.png)

btop-style bars for this machine: GPU memory (with GiB free), RAM, the die (Tctl), board and GPU edge temperatures, fan (percent, rpm and mode), GPU load and clock. Below them, the die's temperature over the last minutes, and the generation and prompt speeds of recent answers: every app's when ts5-tui can read the server's log (see [Logs](#logs-f5)), else its own.

**F** steps the fan through auto, cool, early and max (as `/fan` does).

## Settings (F4)

**↑ ↓** choose, **← →** change, **Enter** edits a text setting (Esc cancels). Every change is saved at once, in `~/.config/tokenstation/tui.conf`.

| Setting | What it does |
| --- | --- |
| Server | The TokenStation to talk to. An address that can't be right isn't saved, and the reason is shown. One given with `--url` or `TS5_URL` is used for that run only and not saved (the same for `--key` and `TS5_API_KEY`). |
| Server's key | The key ts5-tui itself sends, for a server on another machine that asks for one. Kept in the settings file, readable by you only. The keys other machines use to reach this one are made on the [Access](#access-f6) page. |
| Theme | Ocean, Neon, Sunset, Forest, Mono. |
| Thinking | The model's choice, on, or off, for models that can switch it (Qwen 3, Qwen 3.5, Gemma 4). Off answers sooner. |
| Temperature | **default** with its number beside it (0.7 for every model but Mistral Nemo, 0.3), or 0.1 to 1.5. Lower is steadier (code, facts), higher more varied. Past either end it goes back to default. Set above the default, it changes the thinking and the prose only: while the model writes a tool call or code, it writes at its default. |
| Answer length | No limit, or a cap of 512 to 8,192 tokens. Under 1,024 turns thinking off. |
| Full chat | **is summarized, and goes on** (the default): when the chat is nearly full the model writes a summary of it and the chat continues from that. **stops**: a full chat ends, as before (see [A chat that fills up](#a-chat-that-fills-up)). |
| Web search key | A [Tavily](https://tavily.com) key (free at app.tavily.com). A model ts5-tui loads then searches the whole web when a question needs it (see [Web search](Web-search)); with no key its searches go to the encyclopedia chosen under **Search source**. Kept in the settings file, readable by you only. Load the model again (Models, Enter) for a new key to take hold. For the always-on service, the key goes in `/etc/default/tokenstation` instead. |
| Tools | **on**: the model may use the tools switched on below. **off**: it gets no tools at all, and the rows below are hidden until it is on again. |
| Tool: search | **on**: the model searches when a question needs it: the whole web with the key above, an encyclopedia without one. **off**: it never searches. |
| Search source | Which encyclopedia is searched when there is no web search key: **Wikipedia** (the default), **Grokipedia** (xAI's encyclopedia, written by its AI; when it can't be searched, Wikipedia is), or **Both**. It counts from the next message. With a server from an older build the row says Wikipedia and can't be changed. |
| Tool: pages | **on**: the model can read a web page you give it a link to. **off**: it can't open links. |
| Tool: terminal | **on, asks first**: the model can run a command on this machine, and you see each one and say yes or no. **on, never asks**: commands run without showing you first, for a model you trust and no pages from strangers. **off**: it can't run commands. |
| Sandbox | **on, with network** (the default): the model's commands see the system read-only and none of your home folder, and change only the folder ts5-tui started in and their own toolbox. **on, no network**: the same, offline. **off**: commands run as you, with all your files. Shows **unavailable** and why when bubblewrap can't run here. |
| Sandbox folder | Where the toolbox is (what the model installs), with its size and the free space on that drive. **Enter** and type another folder to move it there, or `default`. |
| Tool: files | **off** (the default): the model can't read or change files. **on, asks first**: it reads and changes files; each change is shown first, lines out and lines in, for your yes or no. **on, never asks**: without asking (`/undo` still takes a change back). |
| Tool: browser | **off** (the default): the model can't click through web pages. **on, asks first**: it opens pages and follows links by itself; a button or typing waits for your yes. **on, never asks**: it presses buttons and types without asking. Switched on with no browser installed, it asks once whether to install one (see [Browser](#browser)). |
| Browser | Which program shows the pages: **Auto** (Chromium if one is installed, else ts5's own page reader), **Chromium**, **Obscura**, **Text view**. The line under it says what is installed. |
| Outside tools | One row for each program you added with `/mcp add`: **off**, **on, asks first**, **on, never asks**. The line under it says how many tools it has and what they cost in every message, or why it isn't running. |
| Tool: memory | **on**: the model keeps notes when you ask it to remember, and reads them at the start of every chat (`/memory`). **off**: it keeps none and doesn't see the saved ones. |
| Context (own) | The context for a model ts5-tui starts itself: auto (the default: the longest that fits), or 4,096 to 32,768. A number that has no memory beside a model doesn't stop the load (see [Models](#models-f2)); auto is the safe choice. A setting saved as 8,192 by an older version reads as auto. |
| Speed or length | For a model with a helper that ts5-tui starts itself, with Context on auto. **faster** (the default) keeps GPU memory for the helper: the model writes faster, and the chat is somewhat shorter. **longest chat** gives the memory to the chat, and the model runs without its helper when both don't fit. It takes hold the next time the model loads (Enter on it on Models). |
| Fan (PS5) | auto (quiet), cool (keeps long prompts under the 85 °C limiter), early (ramps from 40 °C, for long prompts back to back), max (loud). |
| tmux | Its own tmux session from a plain terminal. |
| Mouse | **on** (the default): clicks open the thinking, copy, and change pages; a drag opens the text to select. **off**: the terminal keeps the mouse, and dragging selects what's on the screen. |
| When you quit | **the model is unloaded** (the default): a model ts5-tui loaded stops with it and its GPU memory is free. **the model stays loaded**: it keeps serving the API after ts5-tui quits (stop it later with `kill`, or `/unload` from ts5-tui). |
| Opening title | The title at start. |

The engine's own switches (prompt math, cache, helper) are per model: built in for the models tested on the PS5, and set by `ts5-tune` for any other (Models → **T**).

## Logs (F5)

What the engine writes, colored: errors red, warnings yellow, speed lines green. **PgUp/PgDn** and **↑ ↓** scroll, **End** follows the newest again, **Home** goes to the start.

It follows the API-only service's journal (`journalctl -u tokenstation`; reading it may need the `systemd-journal` or `adm` group), the TV app's `/var/lib/tokenstation/ts5.log`, or the log of the `ts5 serve` it started, whichever is running. `TS5_TUI_LOG=/path/to/file ts5-tui` follows a file of your choice.

## Access (F6)

Who can use this TokenStation5 from another machine, and with which key.

```
    Access  who can reach this TokenStation5, and with which key

     Other devices     ‹ yes ›
                       http://192.168.1.20:8081/v1 · the address to give an app on another machine
     They need a key   ‹ yes ›
                       another machine gets in only with a key that is on. This machine never needs one
     Tailscale         ‹ on ›
                       connected as tokenstation5 · 100.101.102.103 · 4 other devices · D signs out
     SSH via Tailscale ‹ on ›
                       ssh you@100.101.102.103, from your tailnet
     Browser terminal  ‹ on ›
                       http://100.101.102.103:8081/ts5/terminal · Enter: as a QR code
     Connect an app    a new key, as QR codes
                       Enter: your tailnet's address and a new key, as QR codes for a phone

     Keys  shown once, when made · kept as SHA-256 in ~/.local/share/tokenstation/access/keys.tsv
   ▌ Open WebUI on the tower                  on   made 2 days ago   last used 5 minutes ago
     laptop                                   off  made 2 days ago   last used never
```

**↑ ↓** choose a row, **← →** or **Enter** change it, **N** makes a key, **D** removes the chosen key (it asks first), **R** reads the keys again, **Esc** goes back to the chat.

| Row | What it does |
| --- | --- |
| Other devices | Whether a model ts5-tui loads can be reached from other machines at all. **no** (the default): only this machine. **yes**: the page shows the address to give an app, and offers to load the model again, which is when it takes hold. |
| They need a key | **yes**: another machine gets in only with a key that is on; without one it gets HTTP 401 and a line that says where keys are made. **no**: any machine that can reach the port gets in. Switching Other devices on turns this on too. |
| Tailscale | **on**: the server joins your tailnet, so your phones and computers reach it at home or away ([Tailscale](Away-from-home-with-Tailscale)). The first time, the sign-in link opens as a QR code to scan with your phone. A device on the tailnet always needs a key. **D** signs out. |
| SSH via Tailscale | **on**: the tailnet's port 22 reaches this machine's own SSH server, so `ssh` and an editor's remote SSH (VS Code Remote-SSH, Zed, JetBrains Gateway) work from away too. Off by default. SSH's own login applies (your keys or password), not the API's keys. It needs an SSH server (`sudo apt install openssh-server`); the row says so when there is none. See [Tailscale](Away-from-home-with-Tailscale#ssh-from-away). |
| Browser terminal | **on**: your tailnet's devices open ts5-tui in a web browser, with a key. Off by default. **Enter** shows its address as a QR code. See [Tailscale](Away-from-home-with-Tailscale#in-a-browser-the-chat-page-and-the-terminal). |
| Connect an app | Makes a key and shows the address (the tailnet's when Tailscale is connected, else this network's) and the key as two QR codes, for the app on a phone. **V** shows them as plain text. |
| Keys | **N** makes one: name it after the app or the machine. The key (such as `ts5-k7qp2-wx9md-3hf8r-tz6na`) is shown once, with the address, as QR codes (**V**: as plain text to select), and copied when your terminal allows it; ts5 keeps only its SHA-256. **Enter** turns a key off or on. Each row says when the key was made and when it was last used. |

- **This machine never needs a key**: ts5-tui, or a program beside it that talks to `127.0.0.1`.
- A key made, turned off or removed holds from the next request. The model stays loaded.
- In the app on the other machine, the server is the address the page shows and the API key is the key (see [Connecting apps](Connecting-apps)).
- The page works on the keys of the server in use. For a model ts5-tui loaded they are in `~/.local/share/tokenstation/access/keys.tsv`. For the background service they are in `/var/lib/tokenstation/access/keys.tsv` (your user has to be in the `tokenstation` group), and Other devices is the service's `HOST` setting in `/etc/default/tokenstation`, shown here but not changed. The TV app's keys are listed, and made on its own Settings page.
- In a script, `ts5 keys` does the same: `ts5 keys new "laptop"`, `ts5 keys`, `ts5 keys off laptop`, `ts5 keys remove laptop`, `ts5 keys require on`. Add `--file /var/lib/tokenstation/access/keys.tsv` for the background service's keys.
- Away from home, use the Tailscale row and a key ([Tailscale](Away-from-home-with-Tailscale)). Don't forward port 8081 on your router: the API is plain HTTP.

## Connectors (F7)

Other services the model can work with, each signed in with a QR code. The first is **GitHub**.

```
   Connectors  other services the model can work with, signed in with a QR code

   GitHub
  ▌ Account           connected
  ▌                   as your-name · D disconnects
    Repositories      2
                      Enter: change them on GitHub (a QR code)
    Model's tool      ‹ on, asks first ›
                      reads freely; a change (a branch, a file, a pull request, an issue) waits…

    your-name:        website  notes
```

**Connect GitHub:**

1. **Enter** on Account. A QR code of GitHub's sign-in page opens, with a short code under it (such as `WDJB-MJHT`; **C** copies it).
2. Scan it with a phone that is signed in to GitHub (or open the page on any device), enter the code, and press **Authorize TokenStation5**.
3. GitHub's page for the TokenStation5 app then opens as a QR code too: pick your account, then **All repositories** or only the ones the model may use, and **Install**. The QR code closes by itself once GitHub lists them.

| Row | What it does |
| --- | --- |
| Account | Who is connected. **Enter** connects (or shows the QR code again), **X** stops a sign-in that is waiting, **D** disconnects after asking. |
| Repositories | How many the app may use, listed by account below. **Enter** opens GitHub's page for the app as a QR code, to add or take away repositories. **R** asks GitHub again. |
| Model's tool | **off**, **on, asks first** (the default: reads freely, a change waits for your yes) or **on, never asks**. See [GitHub](#github) under Tools. |

- **Nothing to register and no token to paste.** ts5 signs in as the TokenStation5 GitHub app with GitHub's device flow, the same way `gh auth login` does.
- The sign-in is kept in `~/.local/share/tokenstation/connectors/github.json`, readable by you only. It lasts 8 hours and ts5 renews it when it is used; left unused for 6 months, it needs a new sign-in. The model never sees it, and it stays out of the sandbox.
- **D** forgets it here. To take the app's access away on GitHub's side too: GitHub → Settings → Applications → Authorized GitHub Apps → TokenStation5.
- In a script: `ts5 github sign-in`, `ts5 github status --repos`, `ts5 github sign-out`, and `ts5 github api GET /user/repos`.

GitHub is a trademark of GitHub, Inc.; TokenStation5 is not affiliated with or endorsed by GitHub.

## tmux: SSH that can drop

From a plain terminal (an SSH session from Terminal, iTerm, PuTTY…), ts5-tui starts itself inside its own **tmux** session, called `ts5`, when tmux is installed (the packages recommend it). If the connection drops, nothing stops: the answer finishes, the model keeps loading, tuning goes on. Run `ts5-tui` again and you're back where you were.

- **Ctrl+A** then **D** leaves it running in the background; `ts5-tui` comes back to it. Ctrl+A is this session's command key, so for the start of the line use **Home** (or Ctrl+A twice).
- The mouse wheel scrolls the chat, and clicks work as everywhere else. To copy text, see [The mouse, and copying](#the-mouse-and-copying).
- It uses its own tmux server and settings (`~/.cache/tokenstation/tmux.conf`): your `~/.tmux.conf` and your other tmux sessions are left alone, and tmux's own status bar is hidden.
- Quitting ts5-tui ends the session.

In an **IDE's terminal** (VS Code, Cursor, JetBrains, including their Remote-SSH) tmux stays off, because the IDE keeps its terminals alive itself and its shortcuts would clash. `--tmux` turns it on there anyway, and `--no-tmux` or Settings → tmux turns it off everywhere.

> **Plain SSH leaves more memory for models.** An IDE's Remote-SSH server runs on the PS5 and takes memory the GPU could use (Cursor's took 4.7 to 6.5 GB). For coding, run the IDE on your computer and point it at the PS5's API instead ([Connecting apps](Connecting-apps)).

## When you quit

**Ctrl+Q** (always shown at the bottom right), `exit`, `quit` or `q` at the prompt, `/quit`, or **q** on a page other than Chat. While a tuning, a check or a download runs, quitting asks first. The same keys and **Ctrl+C** (when no answer is running; while one is, Ctrl+C stops the answer). The chat is printed into your terminal, so it stays in its scrollback (also after a tmux session).

**Quitting stops the model too.** A model that ts5-tui loaded is unloaded when you quit, and its GPU memory is free for the TV app, a game or the next model; the last line printed says so. Three things are left alone: a model the service or the TV app serves (ts5-tui didn't start it), a session you only left with **Ctrl+A d** (ts5-tui and the model both keep running, and `ts5-tui` takes you back), and the model itself when Settings → **When you quit** is set to keep it loaded.

## If something's off

| What you see | Why, and what to do |
| --- | --- |
| **● offline** in the top bar | Nothing answers at the address. Start the service (`sudo systemctl start tokenstation`) or the TV app, or load a model on the Models page, which starts `ts5 serve`. Check the address in Settings. |
| **needs an API key** | The server is on another machine and asks for a key: make one there (its Access page, or the TV app's Settings), and put it in Settings → Server's key, or start with `--key`. |
| No temperatures, fan or GPU memory | ts5-tui isn't running on the PS5 (a Mac, or `--url` to another machine), or the PS5 kernel's sensors are missing. Run it over SSH on the PS5. |
| Load or fan: "needs the tokenstation group" | `sudo adduser $USER tokenstation`, then log out and in. |
| Colors look washed out or wrong | The terminal has no 24-bit color, so ts5-tui uses 256 colors. Terminals that support it set `COLORTERM=truecolor`; try another theme, or `Mono`. |
| Odd characters instead of boxes | The terminal isn't UTF-8. Set a UTF-8 locale (`export LANG=C.UTF-8`). |
| tmux opens when you don't want it | `ts5-tui --no-tmux`, or Settings → tmux → off. |
| A model "didn't load": not enough GPU memory | The message says how much it needs and how much is free. If another program holds the memory (the TV app, a game, a tune or check still running, a model left loaded by an older ts5-tui: `pkill -f "ts5 serve"`), stop that. If the model is simply too big for this GPU, pick a smaller file type (Q4_K_M) or a smaller model. |
| Copied, but nothing to paste | Your terminal doesn't let programs set the clipboard (Apple's Terminal never does). Select the text in the plain view that opens after every copy, and press Cmd+C. |
| A web page comes back as "a simpler text view" | No browser is installed, or the one chosen didn't start (the page says why, once). `/browser install` offers Chromium or Obscura; after that, `/browser auto`. |
| An outside tool's program is "not running" | `/mcp` says why (the program isn't installed, it ended as it started, it said nothing for two minutes). Its own words are in `~/.local/share/tokenstation/mcp-NAME.log`. Fix it, then `/mcp NAME restart`. `ts5-tui --mcp` shows the same without a chat. |
| Clicks do nothing | Settings → Mouse must be on. In Apple's Terminal, View → **Allow Mouse Reporting** must be ticked. |

More in [Troubleshooting](Troubleshooting).
