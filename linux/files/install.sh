#!/bin/sh
# TokenStation5, the portable build: for Linux without apt, or with a
# read-only system (Bazzite, SteamOS, Fedora Atomic), or anywhere apt isn't
# wanted. Run it from the unpacked folder, as yourself (not with sudo):
#
#   ./install.sh                  the app in ~/.local, plus the PS5 hardware
#                                 controls (asks for your password once)
#   ./install.sh --no-hardware    the app only; nothing outside your home
#   ./install.sh --no-start       don't offer to start it when done
#   ./install.sh --no-steam       don't add it to Steam (Game Mode)
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

hardware=1 start=1 steam=1 uninstall=0
for a in "$@"; do
    case "$a" in
        --no-hardware) hardware=0 ;;
        --no-start) start=0 ;;
        --no-steam) steam=0 ;;
        --uninstall) uninstall=1 ;;
        *) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
    esac
done
case "$uninstall" in
    1)
        rm -rf "$opt"
        rm -f "$bin/tokenstation" "$bin/ts5" "$bin/ts5-tune" "$apps/tokenstation.desktop" "$icons/tokenstation.png"
        if [ -e "$sys" ] || [ -e "$sudoers" ]; then
            echo "Removing the hardware controls (sudo):"
            sudo rm -rf "$sys" "$sudoers" /usr/local/bin/tokenstation-memory /usr/local/bin/tokenstation-cu
        fi
        echo "TokenStation5 removed. Your models and settings are still in ~/.local/share/tokenstation."
        echo "If you added it to Steam, remove it there too (Library, TokenStation5, Manage)."
        exit 0
        ;;
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
ln -sf "$opt/bin/ts5-tune" "$bin/ts5-tune"
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
        # On a PS5: the GPU may use the most memory (Settings can change it).
        if [ -e /dev/mp1 ] || [ -e /dev/icc ]; then
            /usr/local/bin/tokenstation-memory first || true
        fi
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
fi

# Bazzite and SteamOS: offer it to Steam, for Game Mode.
in_steam=0
if [ "$steam" = 1 ] && command -v steamos-add-to-steam >/dev/null 2>&1 && pgrep -x steam >/dev/null 2>&1; then
    echo "Adding TokenStation5 to Steam, for Game Mode (--no-steam to skip)."
    steamos-add-to-steam "$opt/bin/tokenstation" >/dev/null 2>&1 && in_steam=1 || true
fi

cat <<DONE

TokenStation5 is installed. It's a full-screen TV app, like on the console
(there's no web page; the API is for other apps).

  Start it
    Desktop Mode:  the app menu, TokenStation5, or run $bin/tokenstation
    Game Mode:     $( [ "$in_steam" = 1 ] && echo "it's in your Steam library (non-Steam games)" || echo "in Desktop Mode, Steam > Add a Non-Steam Game, and pick
                   $opt/bin/tokenstation" )

  Drive it
    Controller:    as on the console. Cross selects, Circle goes back
                   (Home), L1/R1 switch pages, hold R1 to talk in Chat,
                   OPTIONS closes it.
    Keyboard:      arrows; Enter = Cross, Esc = Circle, F1 = Square,
                   F2 = Triangle, F3 = OPTIONS, Page Up = L1,
                   Page Down = R1 (hold to talk). F11: window or full screen.

  First steps      Models: download one, then Cross on it to serve it. Then
                   Chat. Or copy .gguf files to
                   ~/.local/share/tokenstation/models
  API              http://localhost:8081/v1 (OpenAI-style), from other
                   computers too, once a model is served
  Faster           With a model downloaded and the app closed, run
                   ts5-tune once (about 10 minutes): it finds the fastest
                   GPU settings for this machine and saves them.
DONE
if [ "$hardware" = 1 ]; then
    cat <<DONE
  PS5 hardware     Settings > PS5 HARDWARE: fan, clock boost, GPU memory,
                   compute units, temperatures. Log out and back in once so
                   they work without a password (started from here, they
                   already do).
DONE
fi
case ":$PATH:" in *":$bin:"*) ;; *) echo "  ($bin is on your PATH after you log in again on most systems.)" ;; esac
echo "  Remove it        ./install.sh --uninstall (your models and chats stay)"
echo

# Start it now, with the tokenstation group already (sg), so the hardware
# controls work before logging in again.
if [ "$start" = 1 ] && [ -t 0 ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
    printf "Start TokenStation5 now? [Y/n] "
    read -r ans || ans=n
    case "$ans" in
        [nN]*) ;;
        *)
            if [ "$hardware" = 1 ] && command -v sg >/dev/null 2>&1; then
                setsid sg tokenstation -c "exec '$opt/bin/tokenstation'" </dev/null >/dev/null 2>&1 &
            else
                setsid "$opt/bin/tokenstation" </dev/null >/dev/null 2>&1 &
            fi
            echo "Starting. It fills the screen; F11 for a window."
            ;;
    esac
fi
