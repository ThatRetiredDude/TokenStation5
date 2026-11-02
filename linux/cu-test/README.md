# PS5 compute unit test

The PS5's GPU has 40 compute units (CUs). Sony ships it with 36 on. The
ps5-linux kernel can turn the other 4 on with `amdgpu.ps5_cu_unlock`, but
on some chips one of them is faulty: the GPU then hangs, or worse, gets
some answers wrong. These two tools let you see what your chip can do and
choose for yourself. Nothing changes the setting unless you ask.

CUs turn on in pairs, so the choices are 36, 38 and 40.

## tokenstation-cu

```sh
tokenstation-cu status            # active CUs now and the boot setting
sudo tokenstation-cu probe        # next boot logs which GPU sections have disabled pairs
sudo tokenstation-cu set 40       # or 38, or 36 (stock); reboot and run the loader again
tokenstation-cu test [SECONDS]    # self-test, plus ts5 check and bench if TokenStation has a model
```

Run `test` once at 36 first: it saves the float results as the baseline that
later runs at 38 or 40 must match exactly. A run fails on any wrong result,
a GPU hang, or an amdgpu timeout or reset in the kernel log.

`set 38` uses the kernel's "first GPU section only" mode. If that section has
no disabled pair on your chip, you stay at 36; `status` shows what you got.

If a setting won't boot, put the boot drive in another computer, open
`cmdline.txt` on its FAT32 partition and change the setting to
`amdgpu.ps5_cu_unlock=0`. `cmdline.txt` must stay one line. Set `CMDLINE`
if yours isn't in `/boot/efi`, `/efi` or `/boot/firmware`.

## gpu-selftest

A standalone Vulkan program; it needs nothing else from TokenStation and works
on any Vulkan GPU. Every pass runs 4M invocations of each test so that every
CU gets work:

- integer hash chains, checked value by value on the CPU
- shared memory exchange inside each workgroup, checked value by value
- float FMA, sqrt and divide chains: every run must give the same bits, and
  with `--compare` the same bits as the saved baseline
- a memory pattern written and read back (`--mem-mb`, default 1024)

```sh
gpu-selftest --seconds 120 --save baseline.txt      # at 36
gpu-selftest --seconds 120 --compare baseline.txt   # at 38 or 40
```

Exit status 0 means every result was right, 1 means wrong results or a hang,
2 means it couldn't set up. Vulkan can't aim work at one CU, so it tells you
whether the chip is right at a setting, not which CU is faulty.

## License

MIT (`LICENSE`), unlike the rest of TokenStation, so other PS5 Linux
projects can use it.
