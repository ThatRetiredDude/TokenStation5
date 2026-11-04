# PS5 hardware on Linux

On a PS5 running Linux, Settings has a **PS5 HARDWARE** section, all from the controller, and the same controls exist as commands. Everything here was measured on one PS5 (Debian 13, Mesa 26.1.6); every console differs a little.

The app changes these through a small root helper, `/usr/lib/tokenstation/hw`, which checks every value against a fixed list. `/etc/sudoers.d/tokenstation` lets the `tokenstation` group run that one script as root without a password; the TV service's user is in the group, and so is whoever installed the package with sudo. For another desktop user: `sudo usermod -aG tokenstation $USER`, then log in again.

## GPU memory for models

Linux on a PS5 sees 14.1 GiB of memory, and the kernel lets the GPU map half of it by default: 7.0 GiB plus the 0.5 GiB carve-out, less than the console's title gets. Raising it is what makes 14B models fit.

```sh
tokenstation-memory status          # what the GPU may use now, and what fits
sudo tokenstation-memory set max    # all but 1 GiB at the next boot (13.1 GiB measured); reboot and run the loader again
sudo tokenstation-memory set 11     # or a fixed amount, to keep room for other apps
```

The packages and `install.sh` set `max` on a PS5 when no limit is set yet. It's a cap, not a reservation: Linux keeps using the memory, and TokenStation sizes each model to what the driver says is free when it loads, so a model that doesn't fit is refused instead of pushing Linux into swap. In Settings this is **GPU memory for models**: Most available, or a fixed size, for the next start.

## Clock boost

ps5-linux-tools' `ps5_control --boost on` puts the console in its game clocks (CPU 3.5 GHz), but its power management then holds the GPU under its 2.23 GHz. With the boost **off**, the kernel's own GPU driver runs it at full speed. Measured on every model: **boost off is about 8 to 10% faster on prompts** and equal or slightly faster on generation. It runs the chip hotter (die peaks near 85 to 90 C, where the driver's own limiter steps in), which cost nothing measurable.

Boost off is therefore the default (`PS5_BOOST=0`, and Off in Settings). Disable ps5-linux-tools' `ps5boost.service`, which turns the boost back on at every boot. Nothing in TokenStation5 sets clocks past 2.23 GHz, and there is no fan-off setting.

## Fan

Settings: the PS5's own curve, or full speed. Full speed uses the ps5-linux kernel's fan driver (`pwm1_enable=0`); `ps5_control --fan max` alone is overridden by that driver within a second. The services turn the fan on at start. A steeper fan curve cooled the die by about 5 C in testing with no change in speed.

## Compute units

The PS5's GPU has 40 compute units and Sony ships it with 36 on. The ps5-linux kernel can turn the other four on (`amdgpu.ps5_cu_unlock`), but on some chips one of them is faulty: the GPU then hangs, or gets some answers wrong. `tokenstation-cu` lets you see what your chip can do and choose for yourself. CUs turn on in pairs, so the choices are 36, 38 and 40.

```sh
tokenstation-cu status            # active CUs now and the boot setting
tokenstation-cu test 300          # at 36 first: saves the baseline the other settings must match exactly
sudo tokenstation-cu probe        # next boot logs which GPU sections have disabled pairs
sudo tokenstation-cu set 40       # or 38, or 36 (stock); reboot and run the loader again
tokenstation-cu test 300          # must match the baseline bit for bit; then run the GPU check on your models
```

`set 38` uses the kernel's "first GPU section only" mode; if that section has no disabled pair on your chip you stay at 36. If a setting won't boot, open `cmdline.txt` on the boot drive's FAT32 partition from another computer and set `amdgpu.ps5_cu_unlock=0`.

On the test console, 40 passed a 300-second self-test with results identical to 36, every model's GPU check, and a clean kernel log; prompts got 3% (Qwen 3 8B) to 13% (Qwen 3 14B) faster, generation unchanged. Another console may not pass. Settings > **Test the compute units** runs the same test from the TV, and the Settings page shows what the kernel reports in `/sys/class/hwmon` as temperatures.

## Helper model

Settings > Models > **Helper model**, Automatic by default: when the models folder has a small model of the same family as the one running (same architecture and vocabulary, a quarter of its size or less, such as Qwen 3 0.6B beside Qwen 3 8B), it guesses a few tokens ahead and the big model checks them in one pass. The output is exactly what the big model alone writes; only the speed changes, and a cost model turns drafting off where nothing is accepted.

Measured on the PS5: the helper pays on **code** with Qwen 3 (about +9%), and costs 6 to 10% on prose with every family, so Automatic leaves it off there. Drafting from the conversation's own text (prompt lookup) is always on and helps most when editing code. Off turns it off; `TS5_DRAFT_MODEL=/path/to.gguf` in `ts5.env` picks one.

## Tune for your machine

The engine's kernel choices were tuned on the PS5's own driver; Linux runs the same GPU through RADV (Mesa), whose shader compiler differs. With the app closed:

```sh
ts5-tune                 # about 10 minutes for an 8B model
ts5-tune MODEL.gguf      # with another model
ts5-tune --reset         # back to the built-in settings
```

It benches each kernel choice one at a time, keeps a change only when it beats twice the run-to-run noise and at least 2%, confirms with `ts5 check` that the output still matches the CPU, and saves the winners to `~/.config/tokenstation/tune.env`, which the app and `ts5` read at start. On the test console the built-in defaults (wave32, a 128-wide prompt split) were already the best point, so expect no change on a PS5; on another GPU it may find some.

## Temperatures seen

Idle 47 to 59 C; benchmarks 55 to 65 C at the main SoC sensor; a 4,096-token prompt peaks around 74 C (boost on) or 84 C (boost off) with the fan at about 75%, and recovers within seconds.
