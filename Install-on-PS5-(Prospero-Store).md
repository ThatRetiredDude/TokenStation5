# Install on PS5 (Prospero Store)

The PS5 title, **TokenStation5** (title ID `PPSA99555`), runs on a jailbroken PS5. It is listed on the [Prospero Store](https://github.com/darkxex/ProsperoStore), a homebrew app store that runs on the console itself.

## What you need

- A PS5 on **firmware 13.60 or lower**, with the jailbreak loaded. TokenStation5 does not jailbreak anything itself.
- The jailbreak environment that the Prospero Store needs: **ShadowMountPlus** (1.7beta4 or newer) and an ELF loader on port 9021 (elfldr). **ftpsrv** (port 2121) is strongly recommended too: the title uses it to bring your models back after an update, and it is how you copy models in from a computer.
- A home network. The PS5 downloads models from Hugging Face, so it needs internet access for that.
- Optional: a USB or Bluetooth keyboard for typing in Chat, and a free [Tavily](https://tavily.com) key for [web search](Web-search).

## Install from the store

1. Install the Prospero Store itself, following [its README](https://github.com/darkxex/ProsperoStore#install): unzip `PPSA99000.zip` into `/data/homebrew/PPSA99000` over FTP, wait for ShadowMountPlus to add it to the home screen, and open it. If the console says it can't start the app (CE-107750-0), set the folder's permissions to 777 in your FTP client. Make sure the console's clock is right, or the store's HTTPS downloads fail.
2. In the store, find **TokenStation5** (under Apps, category AI, or search for it) and press cross on its page. The store downloads the release ZIP from this repository's GitHub release, checks it against the catalog's SHA-256, unpacks it and puts it in place.
3. Wait for the home screen tile, then launch **TokenStation5**.

The title lands in `/data/homebrew/PPSA99555` (or the drive you chose in the store's settings). Its models, settings, keys and chats live in that folder and in the title's own save area.

## Install by hand

If you'd rather not use the store, download `PPSA99555.zip` from the [latest TokenStation5 release](https://github.com/ThatRetiredDude/TokenStation5/releases/latest), unzip it, and copy the `PPSA99555` folder to `/data/homebrew/` over FTP so that `eboot.bin` ends up at `/data/homebrew/PPSA99555/eboot.bin`. Or, from a checkout of this repository on a computer:

```sh
export PS5_HOST=192.168.1.50
scripts/ts5-ps5.sh install          # title -> /data/homebrew/PPSA99555
```

An app you installed by hand can be handed to the store afterwards with "Manage with ProsperoStore" on its page, so updates come through the store.

## First start

The title opens on the Models page when there is no model yet. Choose one of the built-in downloads (Llama 3.2 3B is the quickest way to see it work, Qwen 2.5 Coder 7B is the one for coding) and press cross. The download runs on the console, checked against its SHA-256, and a stopped download picks up where it left off. When it's done the model starts, and a notification on the TV gives the address apps use.

Downloading on the console is slow past the first few hundred MB (the PS5 lets a title write at about 1 to 4 MB/s). Copying a model in over FTP is much faster: see [Models](Models#adding-models-from-a-computer).

## Updating

Install the new version from the store's Updates section (or Update all). A store update replaces the title's folder but keeps your settings, keys and chats, and the **first start of the new version brings back the models you downloaded**, through the jailbreak's FTP server. Start the title once after each update: the store keeps the old folder only until the next update. Without ftpsrv, put the models back from a computer or download them again.

## Next

- [Using the TV app](Using-the-TV-app)
- [Recommended settings](Recommended-settings)
- [Connecting apps](Connecting-apps)
- [Troubleshooting](Troubleshooting)

> **Keep it on a network you trust.** The jailbreak's FTP and payload ports have no password, so anyone on your network can read the title's files, including its API keys, the Tavily key and the Tailscale keys. The API itself asks for a key once Settings > Require a key is on.
