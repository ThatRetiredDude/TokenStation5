# Troubleshooting

Start with the **Logs** page on the TV (or `ts5.log` in the app's folder): every request and every problem has a line there in plain words.

## PS5 title

- **The title doesn't appear on the home screen.** ShadowMountPlus must be running and scanning the folder (`/data/homebrew`). If the console says it can't start the app (CE-107750-0), set the folder and its files to permissions 777 in your FTP client; apps the Prospero Store installs are set for you.
- **Models disappeared after an update.** Start the title once: the first start of a new version brings the models back through the jailbreak's FTP server (port 2121). Without ftpsrv, copy them in again from a computer or download them again. The store keeps the previous folder in `/data/prosperostore/previous/PPSA99555` until the next update.
- **A download is slow.** The PS5 lets a title write at about 1 to 4 MB/s after the first few hundred MB. Copy models in over FTP instead: [Models](Models#adding-models-from-a-computer).
- **A download stopped.** Cross on it continues from where it left off. A dropped connection is retried on its own; any other reason is on the page and in Logs.
- **The model won't start.** Square on it (the GPU check) says why, and Logs has the details. A model that doesn't fit the memory with at least some context is refused; pick a smaller quant.
- **R1 talk does nothing.** A speech model must be downloaded (Settings > Speech model). The mic is open only while R1 is held.
- **No spoken replies.** Download Kokoro and a voice in Settings, and check that spoken replies are on and aimed at the TV or the controller.

## Linux

- **"no Vulkan device" / ts5 finds no GPU on a PS5.** Mesa is too old for the PS5's GPU (it needs 25.3 or newer). Debian 12's Mesa doesn't know the chip; Debian 13 with `trixie-backports`, Fedora, Ubuntu 26.04 and Bazzite work. `vulkaninfo --summary` should show "RADV GFX1013".
- **A 14B model doesn't fit.** `tokenstation-memory status`, then `sudo tokenstation-memory set max` and reboot (re-run the loader). See [PS5 hardware on Linux](PS5-hardware-on-Linux).
- **Port 8081 is in use.** The TV service and the API-only service can't both run; the desktop app also serves the API itself. Disable the one you don't want (`sudo systemctl disable --now tokenstation`).
- **The hardware settings do nothing.** Your user must be in the `tokenstation` group (`sudo usermod -aG tokenstation $USER`, log in again), and the boost needs `ps5_control` from ps5-linux-tools on the PATH or in `PS5_CONTROL`.
- **The boost keeps coming back on.** Disable ps5-linux-tools' `ps5boost.service`.
- **The console won't boot after a compute-unit change.** Open `cmdline.txt` on the boot drive's FAT32 partition from another computer and set `amdgpu.ps5_cu_unlock=0`. Keep the file on one line.
- **The GPU check fails after a compute-unit change.** Go back to 36. That chip has a faulty unit among the extra ones.
- **It's slower than the numbers here.** Check the boost is off, the GPU memory isn't swapping (`tokenstation-memory status`), and nothing else is using the GPU (close Steam or the desktop for the TV service). `ts5 bench -m model.gguf` gives the raw engine speed.

## Apps and the API

- **The app can't connect.** It must use `http://`, not `https://`, with `/v1` at the end, and the PS5's current address. A request that never shows in Logs never reached the PS5.
- **401 invalid_api_key.** Require a key is on and the app sent none, or sent one that's off. Make a new key in Settings > Apps.
- **"maximum context length".** The conversation is longer than the model's context. Tell the app the `context_length` from `/v1/models`, or start a new task.
- **Away from home nothing connects.** The PS5 needs to be on your tailnet and an API key is always required over Tailscale. Check Settings shows the device, and your tailnet's access rules allow port 8081.
- **Web search never happens.** The key must start with `tvly-` and the console's clock must be right (HTTPS fails otherwise). Only chats that look like they need the web get the tool; `"web_search": true` in the request forces it.
