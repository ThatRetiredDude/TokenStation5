#!/bin/sh
# Boots the OS initrd in QEMU with a stock x86-64 kernel and checks:
# init runs, the banner and shell come up, DHCP on Ethernet works, and
# dropbear takes a key login.
#   os/tests/qemu-boot.sh KERNEL INITRD
set -eu
K=$1 I=$2
T=$(mktemp -d)
[ -n "${KEEP:-}" ] || trap 'kill $(cat $T/qemu.pid 2>/dev/null) 2>/dev/null || true; rm -rf $T' EXIT
ssh-keygen -q -t ed25519 -N '' -f "$T/key"
mkdir -p "$T/x/etc/ts5"
cp "$T/key.pub" "$T/x/etc/ts5/authorized_keys"
(cd "$T/x" && find . | cpio -o -H newc -R 0:0 --quiet | gzip) >"$T/key.cpio.gz"
cat "$I" "$T/key.cpio.gz" >"$T/initrd"
ACCEL=; [ -w /dev/kvm ] && ACCEL="-enable-kvm -cpu host"
qemu-system-x86_64 $ACCEL -m 512 -smp 2 -no-reboot -kernel "$K" -initrd "$T/initrd" \
  -append "console=ttyS0 panic=-1 quiet ts5.debug" \
  -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2222-:22 \
  -chardev pty,id=s0,logfile="$T/console.log" -serial chardev:s0 -monitor none -display none -daemonize -pidfile "$T/qemu.pid"
ok=
for i in $(seq 1 ${WAIT:-120}); do
  grep -q "ts5: boot done" "$T/console.log" 2>/dev/null && ok=1 && break
  sleep 1
done
echo "--- console"; head -c 4000 "$T/console.log"
[ -n "$ok" ] || { echo "FAIL: no 'boot done' within 120 s"; exit 1; }
grep -q "pushes tokens" "$T/console.log" || { echo "FAIL: no banner on the console"; exit 1; }
SSH="ssh -i $T/key -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=5 root@127.0.0.1"
for i in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 1; done
echo "--- over ssh"
$SSH 'uname -r; hostname; ifconfig eth0; cat /etc/resolv.conf; ps -o pid,args | grep -v "\["'
$SSH 'ifconfig' | grep -q "10.0.2.15" || { echo "FAIL: Ethernet did not get its DHCP address"; exit 1; }
echo "qemu-boot: ok"
