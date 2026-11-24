# TokenStation5 OS

The PS5 boots straight into TokenStation: a Linux kernel and a small RAM
image with nothing else in it. Debian stays on the disk as the fallback
(kexec).

What is in the image (about 0.9 MB, gzipped):

- **toybox** (0BSD): init and the everyday tools, including `dhcp` and `route`.
- **mksh** (MirOS licence): the shell.
- **dropbear** (MIT): ssh, with key logins only.
- **os/rootfs**: start-up (`/etc/rc`), the boot banner and the login profile.

At start-up `/etc/rc`:

1. mounts the partition labelled `TS5DATA` (the models SSD) on `/data`,
2. brings up Ethernet with DHCP, then Wi-Fi when `/etc/rc.wifi` is present,
3. starts ssh with the keys in `/data/ssh/authorized_keys` and keeps the
   host keys in `/data/ssh`, so they survive reboots,
4. starts `ts5 serve` from `/data/ts5` in the background, with the model loaded,
5. prints the banner and opens a root shell on the console.

Kernel command line settings: `ts5.data=LABEL`, `ts5.ip=ADDR/BITS,GATEWAY,DNS`
for a fixed address, and `ts5.debug` to print the start-up log.

Build and test (Linux, needs musl-tools, cpio and qemu-system-x86):

    os/build.sh out
    os/tests/qemu-boot.sh /boot/vmlinuz-$(uname -r) out/ts5os-initrd.cpio.gz

Still to come: the PS5 kernel with its small amdgpu patches, Wi-Fi, and
the kexec entry that boots this image.
