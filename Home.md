# TokenStation5

Your PS5 as a private AI assistant: chat on the TV, talk to it with the controller, and serve an OpenAI-compatible API to the coding tools and chat apps on your network, or anywhere with Tailscale. Everything runs on the console's own GPU through **ts5**, the project's own inference engine. No cloud, no subscription.

There are two editions, built from the same code:

| Edition | Runs on | Get it |
| --- | --- | --- |
| **PS5 title** (TokenStation5, `PPSA99555`) | A jailbroken PS5 on firmware 13.60 or lower | [Install on PS5 (Prospero Store)](Install-on-PS5-(Prospero-Store)) |
| **Linux** | A PS5 running Linux, or any Linux PC with a Vulkan GPU | [Install on Linux](Install-on-Linux) |

> **Linux edition: testers wanted.** The newest Linux build is the [linux-latest pre-release](https://github.com/ThatRetiredDude/TokenStation5/releases/tag/linux-latest). On Debian or Ubuntu:
>
> `curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation_latest_amd64.deb && sudo apt install ./tokenstation_latest_amd64.deb`
>
> Fedora, Arch, Bazzite and SteamOS are in [Install on Linux](Install-on-Linux), with how to report what you find.

## Start here

1. Install one edition (links above).
2. [Pick a model](Models). The Models page downloads the checked ones straight to the console.
3. [Connect an app](Connecting-apps), or just chat on the TV (or, on Linux, [in a terminal over SSH](Using-the-terminal-app)).
4. Optional: [reach it from anywhere with Tailscale](Away-from-home-with-Tailscale), and [give it web search](Web-search).

## Pages

- [Using the TV app](Using-the-TV-app): the five pages and what every button does
- [Using the terminal app (ts5-tui)](Using-the-terminal-app): chat, models, live temperatures and tuning in a terminal, over SSH (Linux)
- [Models](Models): what fits, what's checked, adding your own
- [Connecting apps](Connecting-apps): Kilo Code, Continue, Cline, Open WebUI, aider, opencode, your own scripts
- [Away from home with Tailscale](Away-from-home-with-Tailscale)
- [Web search](Web-search): a Tavily key and the model looks things up
- [PS5 hardware on Linux](PS5-hardware-on-Linux): GPU memory, compute units, clock boost, fan, tuning
- [Benchmarks](Benchmarks): measured speeds on the console and on Linux
- [Test results on PS5 Linux](Test-results-on-PS5-Linux): what was tested on a PS5 with Linux, and what came out
- [Recommended settings](Recommended-settings): what to run, and which model for what
- [Building from source](Building-from-source): the title, the Linux packages, the engine
- [Troubleshooting](Troubleshooting)
- [Licensing](Licensing): what you may do with it

## Safety in one paragraph

The API is plain HTTP on port 8081. Turn on **Require a key** in Settings and make a key per app. On a jailbroken PS5 the jailbreak's own FTP and payload ports have no password, so anyone on your network can read the title's files, keys included: keep the console on a network you trust, and never forward port 8081 on your router. Use Tailscale instead.

TokenStation5 is an independent project, not affiliated with or endorsed by Sony Interactive Entertainment, Tailscale Inc. or any model maker.
