#!/bin/sh
# TokenStation5, the portable build: for Linux without apt, or with a
# read-only system (Bazzite, SteamOS, Fedora Atomic), or anywhere apt isn't
# wanted. Run it from the unpacked folder, as yourself (not with sudo):
#
#   ./install.sh                  the app in ~/.local, plus the PS5 hardware
#                                 controls (asks for your password once)
#   ./install.sh --no-hardware    the app only; nothing outside your home
#   ./install.sh --uninstall      removes both (your models and chats stay)
#
# The app goes in ~/.local/opt/tokenstation, with tokenstation and ts5 in
# ~/.local/bin and an entry in the app menu. Its data (models, settings,
# chats) is in ~/.local/share/tokenstation, as with the .deb.
#
# The hardware controls (Settings > PS5 HARDWARE: fan, clock boost, GPU
# memory, compute units, the compute unit test) need a few root-owned files
# outside /usr, which is read-only on those systems:
#   /usr/local/lib/tokenstation/    the helper and the tools it runs
#   /usr/local/bin/                 tokenstation-memory, tokenstation-cu
#   /etc/sudoers.d/tokenstation     lets the tokenstation group run the
#                                   helper, and only it, without a password
# and adds you to the tokenstation group (log out and in again after).
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd)
opt=$HOME/.local/opt/tokenstation
bin=$HOME/.local/bin
apps=$HOME/.local/share/applications
icons=$HOME/.local/share/icons/hicolor/512x512/apps
sys=/usr/local/lib/tokenstation
sudoers=/etc/sudoers.d/tokenstation

if [ "$(id -u)" = 0 ]; then
    echo "Run ./install.sh as yourself, not with sudo: the app goes in your home folder." >&2
    echo "It asks for your password itself for the hardware controls." >&2
    exit 2
fi

hardware=1
case "${1:-}" in
    --no-hardware) hardware=0 ;;
    --uninstall)
        rm -rf "$opt"
        rm -f "$bin/tokenstation" "$bin/ts5" "$apps/tokenstation.desktop" "$icons/tokenstation.png"
        if [ -e "$sys" ] || [ -e "$sudoers" ]; then
            echo "Removing the hardware controls (sudo):"
            sudo rm -rf "$sys" "$sudoers" /usr/local/bin/tokenstation-memory /usr/local/bin/tokenstation-cu
        fi
        echo "TokenStation5 removed. Your models and settings are still in ~/.local/share/tokenstation."
        exit 0
        ;;
    "") ;;
    *) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

# What the app needs from the system.
missing=
for lib in libSDL2-2.0.so.0 libvulkan.so.1; do
    if ! { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -q "$lib"; then
        missing="$missing $lib"
    fi
done
[ -z "$missing" ] || echo "Warning: this system seems to be missing$missing. The app won't start without it (SDL2 and the Vulkan loader)." >&2

echo "Installing the app in $opt"
rm -rf "$opt.new"
mkdir -p "$opt.new" "$bin" "$apps" "$icons"
cp -r "$here/bin" "$here/lib" "$here/share" "$opt.new/"
rm -rf "$opt"
mv "$opt.new" "$opt"
ln -sf "$opt/bin/tokenstation" "$bin/tokenstation"
ln -sf "$opt/bin/ts5" "$bin/ts5"
cp "$opt/share/tokenstation.png" "$icons/tokenstation.png"
sed "s#^Exec=.*#Exec=$opt/bin/tokenstation#" "$opt/share/tokenstation.desktop" > "$apps/tokenstation.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$apps" 2>/dev/null || true

if [ "$hardware" = 1 ]; then
    echo "Installing the PS5 hardware controls (sudo asks for your password):"
    sudo sh -eu -c '
        install -d -o root -g root -m 755 "$1"
        for f in hw gpu-selftest; do install -o root -g root -m 755 "$2/lib/$f" "$1/$f"; done
        for f in tokenstation-memory tokenstation-cu; do install -o root -g root -m 755 "$2/bin/$f" "/usr/local/bin/$f"; done
        getent group tokenstation >/dev/null || groupadd -r tokenstation
        usermod -a -G tokenstation "$3"
        tmp=$(mktemp)
        printf "%s\n" "# TokenStation5 (installed by install.sh): Settings > PS5 HARDWARE runs this" \
            "# one helper, which checks every value against a fixed list." \
            "%tokenstation ALL=(root) NOPASSWD: $1/hw" > "$tmp"
        if visudo -cqf "$tmp"; then
            install -o root -g root -m 440 "$tmp" "$4"
        else
            echo "The sudo rule did not check out; the hardware controls will ask for a password." >&2
        fi
        rm -f "$tmp"
    ' sh "$sys" "$opt" "$(id -un)" "$sudoers"
    echo "Log out and back in once, so the hardware controls work without a password."
fi

echo
echo "Done. Start TokenStation5 from the app menu, or run: $bin/tokenstation"
case ":$PATH:" in *":$bin:"*) ;; *) echo "($bin isn't on your PATH yet; it is after you log in again on most systems.)" ;; esac
echo "Put .gguf models in ~/.local/share/tokenstation/models, or download them in the app."
