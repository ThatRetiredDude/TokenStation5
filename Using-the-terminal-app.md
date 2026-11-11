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
| `ts5-tui --key KEY` | An API key, when the server asks for one (the TV app's **Require a key**). |
| `ts5-tui --theme Neon` | Colors: `Ocean` (default), `Neon`, `Sunset`, `Forest`, `Mono`. |
| `ts5-tui --no-intro` | Skips the opening title. |
| `ts5-tui --tmux` / `--no-tmux` | Forces its own tmux session on or off (see [tmux](#tmux-ssh-that-can-drop)). |

`TS5_URL` and `TS5_API_KEY` in the environment do the same as `--url` and `--key`.

Run it **on the PS5** (over SSH) for the whole experience. Pointed at a PS5 from another machine with `--url`, chat works, but the sensors, models and logs it shows are that machine's, and loading a model is refused: models are switched where they are served.

## The screen

```
 ◆ TokenStation5   F1 Chat  F2 Models  F3 Status  F4 Settings  F5 Logs        09:52   ← pages
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

**Five pages** fill the middle. Switch with **F1** to **F5**, **Tab** (next) and **Shift+Tab** (previous), or **Alt+1** to **Alt+5**. On every page but Chat, **1** to **5** switch too, **q** quits, and **/** jumps to the prompt with a command started. From the prompt, `/chat`, `/models`, `/status`, `/settings` and `/logs` go to a page.

> **On a Mac**, macOS keeps the F keys for itself (F3 is Mission Control). Use **Tab** / **Shift+Tab**, **1** to **5** on the pages, the commands above, or **fn+F1** to **fn+F5**. **Option+1** to **Option+5** work when the terminal sends Option as Meta: in Terminal, Settings → Profiles → Keyboard → *Use Option as Meta key*; in iTerm2, Profiles → Keys → *Left Option key: Esc+*.

## Chat (F1)

Type at `ts5>` and press **Enter**. The answer streams in with its formatting: headings, lists, quotes, and code blocks with highlighting. A thinking model's reasoning (Qwen 3, Qwen 3.5, DeepSeek R1, Gemma 4) is folded away: while it thinks you see a pulsing **thinking…** line with its last words, then **▸ Thought for 4.1 s**. **Click that line** to read what it thought, and click it again to fold it; **Ctrl+T** unfolds every answer's thinking at once (and folds it again). Under each answer: its speed, tokens, prompt speed and time, and a **[copy]** mark (see [The mouse, and copying](#the-mouse-and-copying)).

| Key | What it does |
| --- | --- |
| Enter | Send. While an answer is coming, the message is queued and sent by itself when that answer is done (see [Messages while it answers](#messages-while-it-answers)). |
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

You don't have to wait for an answer to finish. Type the next message while the model is still writing and press **Enter**: it is queued, shown over the prompt as **queued**, and sent by itself the moment the answer before it is done. Queue as many as you like; they go in order, one answer each.

- **↑** with nothing typed takes the last queued message back into the prompt, to change it or (Esc) throw it away.
- **Esc** stops the answer that is coming. The queue then waits instead of firing: **Enter** sends it on, **Esc** drops it. The same when an answer fails (the server went away).
- Commands (`/think off`, `/model …`) still run at once. A new chat empties the queue.

### A chat that fills up

The top bar always shows how full the chat is (**chat 7.8k/32k**), and the line over the prompt reads "chat N of M tokens": yellow from 75%, red from 90%.

A chat grows until the model's memory for it (its context) is nearly full. Then, before your next message goes out, the model writes **a detailed summary of the chat so far**, and the chat goes on from that summary and its last messages, with room again. Nothing is lost from the screen or the saved chat: the summary is a folded block in the chat, and a click opens it. An answer that ran into the end of the context gets a note and a summary too; ask the model to go on.

- `/compact` writes the summary now. `/compact keep the file names and the error text` says what must not be lost.
- **Esc** stops a summary and leaves the chat as it was.
- Settings → **Full chat**: **stops** turns this off. A full chat then says so under the answer that was cut, with the numbers, and **Ctrl+N** starts a new one (the full one stays saved). For a model loaded at a fixed context, the note adds that Settings → Context → auto gives it more.
- A single message that is longer than the model's whole context can't be summarized away: it comes back to the prompt with a note.

If an answer hit your **Answer length** cap instead, the note under it says that.

### Tools: web pages, a terminal, memory

The model in a chat has tools it can reach for by itself: the **web search** (the whole web with a key, Wikipedia without one), **reading a web page**, **running a command** on the machine ts5-tui runs on, and **notes it keeps between chats**. Each one has its own switch, so you decide what it may do: Settings → **Tool: search**, **Tool: pages**, **Tool: terminal**, **Tool: memory**, or `/tools terminal off`. `/tools` lists them with what is on.

In the chat a tool call is a line of its own (**⚙ $ ls -la**, **⚙ reads https://…**, **⚙ remembers: …**, **⚙ searches the web: …**) with a line under it saying how it went (**▸ exit 0 · 12 lines · 0.3 s**). Click that line to see exactly what the model got back.

**Web search.** Ask for something to be looked up: "look up on Wikipedia which river Bratislava lies on". With no key the model searches Wikipedia: it gets the top articles' titles, links and opening text, and it is told that this is an encyclopedia, with no news and no live data, and when the articles it got are only a loose match. With a [Tavily](https://tavily.com) key (Settings → **Web search key**) it searches the whole web. Each search is a line in the chat, **⚙ searches the web: Bratislava river**, with what came of it under it (**5 results (0.5 s, Wikipedia)**).

**Web pages.** Give it a link: "read https://example.com/post and sum it up". It gets the page as plain text: the article where the page marks one, without the site's menus and sidebars, the first 8,000 characters. Pages that only show something in a browser (built by scripts), files, images and PDFs can't be read, and the model is told why.

**Terminal.** When the model wants to run a command, nothing runs yet. You see the command and a question:

`Run it?  ‹ Run it ›  Every command in this chat  No`

- **Enter** runs it. **Esc** says no: the model is told you said no, and answers in words instead of trying another command.
- **→** moves the answer to **Every command in this chat** (no more questions until a new chat; `/tools ask` brings them back) or **No**, then Enter.
- Letters never answer the question. You can keep typing your next message while it shows; it waits in the queue as always.
- A command runs as you, in the folder ts5-tui was started in, with no keyboard to read from (a program that waits for typing gets nothing), for 2 minutes at most. **Esc** stops one that is running.
- The model gets the output and the exit code. Of long output it gets the start and the end.

**Memory.** Say "remember that I write Rust and like short answers" and the model saves a note. It reads its notes at the start of every chat.

- `/memory` lists the notes, `/memory forget 2` removes one, `/memory clear` removes all, `/memory TEXT` adds one yourself.
- They are plain lines in `~/.local/share/tokenstation/memory.md`; edit the file if you like.

Good to know:

- Every model can call tools. Qwen 3 is the most reliable at it; small models (1B to 4B) call tools less surely, and DeepSeek R1 is asked in words because its file has no tool format of its own.
- A web page is text from a stranger and can try to talk the model into something. With the question on, the worst case is a command you say no to. With **Every command in this chat** chosen, leave pages you don't trust alone.
- A model that keeps calling tools gets twelve rounds for one message, then has to answer with what it has.
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
| `/models` | The Models page. `/chat`, `/status`, `/settings` and `/logs` go to the others. |
| `/download NAME` | Downloads one of the models tested on the PS5 (see [Downloading models](#downloading-models)): `qwen3-14b`, `qwen35-9b`, `deepseek-r1-14b`, `qwen3-8b`, `llama31-8b`, `qwen25-coder-7b`, `qwen25-coder-14b`, `gemma3-12b`, `gemma4-12b`, `llama32-3b`, and the helpers `qwen3-0.6b`, `llama32-1b`, `gemma3-1b`. Tab completes the name. `/download` alone opens the list on the Models page. |
| `/download URL [FILE] [--sha256 HEX]` | Downloads any other `.gguf`. A `ts5 fetch URL FILE --sha256 HEX` line, as on [Models tested on PS5](Models-tested-on-PS5), works pasted at the prompt as it is. |
| `/search WORDS` | Looks on Hugging Face for other models that fit this GPU: `/search mistral 7b` (see [Searching Hugging Face](#searching-hugging-face)). |
| `/new` | A new chat (`/clear` too). The one you leave is kept. |
| `/chats [TEXT]` | Your saved chats, or the ones with TEXT (see [Saved chats](#saved-chats)). |
| `/open N` | Opens saved chat N from `/chats` (or `/open TEXT`: the newest with TEXT). |
| `/save [FILE]` | Saves this chat as Markdown, thinking included. Without a file: `~/tokenstation-chat-YYYYMMDD-HHMM.md`. |
| `/copy` | Copies the last answer (Ctrl+Y). `/copy code`: its last code block. `/copy think`: its thinking. `/copy all`: the whole chat. See [The mouse, and copying](#the-mouse-and-copying). |
| `/compact [WHAT TO KEEP]` | Has the model summarize the chat so far and go on from the summary, with room again (see [A chat that fills up](#a-chat-that-fills-up)). `/summarize` too. |
| `/tools` | Lists the model's tools and which are on. `/tools search`, `pages`, `terminal` or `memory` followed by `on` or `off` sets one (see [Tools](#tools-web-pages-a-terminal-memory)). `/tools ask` has the terminal ask before each command again. |
| `/memory` | The notes the model keeps between chats. `/memory forget N` removes one, `/memory clear` all, `/memory TEXT` adds one. |
| `/find TEXT` | Scrolls to the newest message containing TEXT. |
| `/tune [NAME]` | Runs `ts5-tune` on the loaded model (or NAME), live on the Models page. |
| `/check [NAME]` | The GPU check on the loaded model (or NAME): `ts5 check`, the GPU's answers against the CPU's. |
| `/bench [NAME]` | The speed test on the loaded model (or NAME): `ts5 bench`, prompt and answer tok/s. |
| `/usage` | The API's use over the last day and week, by app (its API key's name, else its address) and by model: requests, prompt and generated tokens, tok/s. From the TV or desktop app's `usage.tsv`. |
| `/keys` | The API keys: name, on or off, when made, when last used. Keys are made on the TV app's Settings page, and only their SHA-256 is kept, so they can't be shown again. |
| `/think on`, `off`, `auto` | Thinking before answering, for models that can switch it (Qwen 3, Qwen 3.5, Gemma 4). `auto` leaves it to the model. |
| `/fan auto`, `cool`, `early`, `max` | The PS5's fan (see [PS5 hardware on Linux](PS5-hardware-on-Linux)). `early` ramps from 40 °C and is at full speed by 55 °C. |
| `/theme NAME` | Colors: Ocean, Neon, Sunset, Forest, Mono. Without a name, the next one. |
| `/status`, `/settings`, `/logs` | Those pages. |
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

- **The models tested on the PS5 know their helper**: Qwen3 0.6B for the Qwen3, DeepSeek R1 and Qwen2.5 Coder models, Llama 3.2 1B for the Llama ones. It downloads right after the model, into the same folder, and **H** fetches it for a model that's already here. Qwen3.5 9B can't use a helper, and the Gemma 12B models run without one (Gemma 3 1B speeds up short answers and slows long prompts; T measures it on your machine).
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
| API key | Only when the server asks for one. Kept in the settings file, readable by you only. |
| Theme | Ocean, Neon, Sunset, Forest, Mono. |
| Thinking | The model's choice, on, or off, for models that can switch it (Qwen 3, Qwen 3.5, Gemma 4). Off answers sooner. |
| Temperature | **default** with its number beside it (0.7 for every model but Mistral Nemo, 0.3), or 0.1 to 1.5. Lower is steadier (code, facts), higher more varied. Past either end it goes back to default. |
| Answer length | No limit, or a cap of 512 to 8,192 tokens. Under 1,024 turns thinking off. |
| Full chat | **is summarized, and goes on** (the default): when the chat is nearly full the model writes a summary of it and the chat continues from that. **stops**: a full chat ends, as before (see [A chat that fills up](#a-chat-that-fills-up)). |
| Web search key | A [Tavily](https://tavily.com) key (free at app.tavily.com). A model ts5-tui loads then searches the whole web when a question needs it (see [Web search](Web-search)); with no key its searches go to Wikipedia. Kept in the settings file, readable by you only. Load the model again (Models, Enter) for a new key to take hold. For the always-on service, the key goes in `/etc/default/tokenstation` instead. |
| Tool: search | **on**: the model searches when a question needs it: the whole web with the key above, Wikipedia without one. **off**: it never searches. |
| Tool: pages | **on**: the model can read a web page you give it a link to. **off**: it can't open links. |
| Tool: terminal | **on, asks first**: the model can run a command on this machine, and you see each one and say yes or no. **off**: it can't run commands. |
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
| **needs an API key** | The server has **Require a key** on: put a key in Settings → API key, or start with `--key`. |
| No temperatures, fan or GPU memory | ts5-tui isn't running on the PS5 (a Mac, or `--url` to another machine), or the PS5 kernel's sensors are missing. Run it over SSH on the PS5. |
| Load or fan: "needs the tokenstation group" | `sudo adduser $USER tokenstation`, then log out and in. |
| Colors look washed out or wrong | The terminal has no 24-bit color, so ts5-tui uses 256 colors. Terminals that support it set `COLORTERM=truecolor`; try another theme, or `Mono`. |
| Odd characters instead of boxes | The terminal isn't UTF-8. Set a UTF-8 locale (`export LANG=C.UTF-8`). |
| tmux opens when you don't want it | `ts5-tui --no-tmux`, or Settings → tmux → off. |
| A model "didn't load": not enough GPU memory | The message says how much it needs and how much is free. If another program holds the memory (the TV app, a game, a tune or check still running, a model left loaded by an older ts5-tui: `pkill -f "ts5 serve"`), stop that. If the model is simply too big for this GPU, pick a smaller file type (Q4_K_M) or a smaller model. |
| Copied, but nothing to paste | Your terminal doesn't let programs set the clipboard (Apple's Terminal never does). Select the text in the plain view that opens after every copy, and press Cmd+C. |
| Clicks do nothing | Settings → Mouse must be on. In Apple's Terminal, View → **Allow Mouse Reporting** must be ticked. |

More in [Troubleshooting](Troubleshooting).
