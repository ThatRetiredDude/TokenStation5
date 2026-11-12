#include "browser_install.h"

#include "app.h"
#include "store.h"

namespace tui {

namespace {

// $1: chromium or obscura. TS5_BROWSERS_DIR: where Obscura goes.
// TS5_DRY: say what would be done, and stop.
// Ends 0 installed, 2 can't be done here, 3 the download failed or was not
// the file expected, 4 installed but not found, else the installer's own.
const char kScript[] = R"SH(
say() { printf '%s\n' "$*"; }
what=${1:-}
if [ "$what" = chromium ]; then
    cmd=
    if command -v apt-get >/dev/null 2>&1; then
        # The light one first: Debian's chromium-headless-shell has no window parts.
        pkg=
        for p in chromium-headless-shell chromium chromium-browser; do
            if apt-cache policy "$p" 2>/dev/null | grep -q 'Candidate: [0-9a-zA-Z]'; then
                pkg=$p
                break
            fi
        done
        if [ -n "$pkg" ]; then cmd="apt-get install -y $pkg"; else cmd="apt-get update && apt-get install -y chromium"; fi
    elif command -v dnf >/dev/null 2>&1; then
        cmd="dnf install -y chromium"
    elif command -v pacman >/dev/null 2>&1; then
        cmd="pacman -S --noconfirm --needed chromium"
    elif command -v zypper >/dev/null 2>&1; then
        cmd="zypper --non-interactive install chromium"
    else
        say "ts5 knows no package program here (apt-get, dnf, pacman, zypper)."
        say "Install Chromium the way this system installs programs, and ts5-tui finds it."
        exit 2
    fi
    if [ "$(id -u)" != 0 ]; then
        if ! command -v sudo >/dev/null 2>&1; then
            say "This needs the administrator, and there is no sudo here. As root, run:"
            say "  $cmd"
            exit 2
        fi
        case "$cmd" in
            *'&&'*) cmd="sudo sh -c '$cmd'" ;;
            *) cmd="sudo $cmd" ;;
        esac
    fi
    say "Installing Chromium from this system's packages:"
    say "  $cmd"
    [ -n "${TS5_DRY:-}" ] && exit 0
    sh -c "$cmd"
    rc=$?
    if [ $rc != 0 ]; then
        say "The install did not finish (the package program ended with $rc)."
        exit $rc
    fi
    for b in chromium chromium-browser chromium-headless-shell chrome-headless-shell headless_shell; do
        if command -v "$b" >/dev/null 2>&1; then
            say "Chromium is installed: $(command -v "$b")"
            exit 0
        fi
    done
    say "The package is installed, but ts5 finds no Chromium program on the PATH."
    exit 4
elif [ "$what" = obscura ]; then
    ver=0.2.6
    url=${TS5_TEST_OBSCURA_URL:-https://github.com/h4ckf0r0day/obscura/releases/download/v$ver/obscura-x86_64-linux.tar.gz}
    sum=${TS5_TEST_OBSCURA_SHA256:-eed986e478c5ec5439f531f2d7fb6b0e36a356f1df7a6c44a8e8fd4f2486bd76}
    if [ "$(uname -m)" != x86_64 ]; then
        say "Obscura's download is for x86_64 machines, and this one is $(uname -m)."
        exit 2
    fi
    libc=$(getconf GNU_LIBC_VERSION 2>/dev/null | sed 's/^[^0-9]*//')
    major=${libc%%.*}
    minor=${libc#*.}
    minor=${minor%%.*}
    case "$major$minor" in
        '' | *[!0-9]*) major=0 minor=0 ;;
    esac
    if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 35 ]; }; then
        say "Obscura needs glibc 2.35 or newer, and this system has ${libc:-another C library}."
        exit 2
    fi
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        say "Neither curl nor wget is installed, so there is nothing to download with."
        exit 2
    fi
    dir=${TS5_BROWSERS_DIR:-$HOME/.local/share/tokenstation/browsers}/obscura-$ver
    part=$dir.part
    say "Downloading Obscura $ver (72 MB, 150 MB once unpacked) from its makers' GitHub release, into $dir:"
    say "  $url"
    [ -n "${TS5_DRY:-}" ] && exit 0
    rm -rf "$part"
    mkdir -p "$part" || exit 2
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$part/obscura.tar.gz" "$url"
    else
        wget -O "$part/obscura.tar.gz" "$url"
    fi
    if [ $? != 0 ]; then
        say "The download failed: nothing was installed."
        rm -rf "$part"
        exit 3
    fi
    got=$(sha256sum "$part/obscura.tar.gz" | cut -d' ' -f1)
    if [ "$got" != "$sum" ]; then
        say "The download is not the file ts5 expects: nothing was installed."
        say "  its SHA-256  $got"
        say "  expected     $sum"
        rm -rf "$part"
        exit 3
    fi
    say "Checked: its SHA-256 is the one ts5 expects ($sum)."
    if ! tar -xzf "$part/obscura.tar.gz" -C "$part"; then
        say "The download could not be unpacked: nothing was installed."
        rm -rf "$part"
        exit 3
    fi
    rm -f "$part/obscura.tar.gz"
    bin=$(find "$part" -maxdepth 2 -type f -name obscura | head -n 1)
    if [ -z "$bin" ]; then
        say "The download holds no obscura program: nothing was installed."
        rm -rf "$part"
        exit 3
    fi
    chmod +x "$bin"
    rm -rf "$dir"
    mv "$part" "$dir" || exit 2
    bin=$dir${bin#"$part"}
    # Where ts5 looks for it. (Its helper program stays beside it.)
    mkdir -p "$HOME/.local/bin"
    ln -sf "$bin" "$HOME/.local/bin/obscura"
    if [ -f "$bin-worker" ]; then
        chmod +x "$bin-worker"
        ln -sf "$bin-worker" "$HOME/.local/bin/obscura-worker"
    fi
    say "Obscura $ver is installed: $HOME/.local/bin/obscura"
    exit 0
else
    say "ts5-tui --install-browser chromium|obscura [--dry-run]"
    exit 2
fi
)SH";

}  // namespace

std::string browser_install_cmd(const std::string &what, bool dry) {
    return std::string(dry ? "TS5_DRY=1 " : "") + "TS5_BROWSERS_DIR=" + shell_quote(data_dir() + "/browsers") + " sh -c " +
           shell_quote(kScript) + " ts5-browser-install " + shell_quote(what);
}

}  // namespace tui
