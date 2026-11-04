# Using the TV app

Both editions show the same five pages. **L1** goes to the next page, **circle** goes back to Home, **OPTIONS** closes the app. On Linux with a keyboard: arrows, Enter (cross), Esc (circle), F1 (square), F2 (triangle), F3 (OPTIONS), Page Up (L1), Page Down (R1).

## Home

How your apps have used the API, like an AI provider's usage page: the model running now, and over the chosen range the tokens written and read, the requests (and how many weren't answered), and the writing speed. Below, tokens by model, requests by app (the TV's own chat is "Chat on this PS5"; other apps by their key's name or address), and a live line of the last two minutes.

| Button | |
| --- | --- |
| left / right | 24 hours, 7 days or 30 days |
| cross | Chat, or Models when nothing is running |
| square | when some requests failed: Logs, at the problems |

## Models

The models on the console, the checked ones you can download, and **Search Hugging Face**: type a name ("qwen coder") and the most downloaded GGUF repos of a kind ts5 runs are searched for files that fit in memory with at least 8,192 tokens of context.

| Button | |
| --- | --- |
| up / down | choose a model |
| cross | start it (apps can use it, and it starts again next time); on the running model, stop it; on a model to download, download, stop or continue |
| square | **GPU check**: the model's output on the GPU against ts5's CPU reference, PASS or FAIL. The first check saves the CPU's answers next to the model; later checks take about a second |
| triangle | **speed test**: a 512-token prompt and 128 generated tokens, in tokens a second |
| right | remove a model that isn't in use (it asks first) |

More in [Models](Models).

## Chat

A conversation with the running model, streamed as it's written. Cross opens the on-screen keyboard (a USB or Bluetooth keyboard types on it too). **Hold R1 and talk** into the controller's mic; let go and what you said is turned into text on the CPU (whisper.cpp) and sent. Replies are read aloud a sentence at a time in a Kokoro voice, on the TV or the controller's speaker, if a voice is downloaded (Settings turns it on and off and picks the voice and speed). Speech recognition and the voice run on the console; nothing you say leaves it.

| Button | |
| --- | --- |
| cross / Enter | type a message |
| R1 (hold) | talk |
| up / down | scroll |
| triangle | stop the reply and its voice |
| square | new chat |

## Settings

- **Apps**: Instructions (guides shown on the TV: set up an app, Tailscale, add models from a computer), **Require a key**, **New key**, and each key's on/off (cross) and removal (square twice).
- **Away from home**: Tailscale on/off, the sign-in QR code, the PS5's tailnet address, and who is connected. See [Away from home with Tailscale](Away-from-home-with-Tailscale).
- **Models** (Linux): the helper model for faster replies. See [PS5 hardware on Linux](PS5-hardware-on-Linux#helper-model).
- **PS5 hardware** (Linux on a PS5): fan, clock boost, GPU memory, compute units, the CU test and temperatures.
- The rest: longest reply, speech model and threads, spoken replies (voice, speed, speaker), the Tavily key for [web search](Web-search), keeping the console awake, screen dimming.

## Logs

The title's own log, `ts5.log`, with a line for every request and every problem in plain words, and the engine's details beneath. The pages say what happened; Logs says why.
