#include "update.h"

#include "app.h"

namespace tui {

namespace {

// $1: the ts5-tui program. $2: the file the line for the user goes in.
// TS5_UPDATE_URL: where the newest build is (tests). TS5_DRY: say what
// would be done, and stop.
const char kScript[] = R"SH(
self=${1:-ts5-tui}
note=${2:-/dev/null}
base=${TS5_UPDATE_URL:-https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest}
say() { printf '%s\n' "$*"; }
tell() { printf '%s\n' "$*" > "$note"; say "$*"; }

# How this copy was installed: the portable build lives in the home folder,
# the packages in /usr.
kind=
case "$self" in
    "$HOME"/*) kind=portable ;;
esac
if [ -z "$kind" ]; then
    if command -v dpkg-query >/dev/null 2>&1 && dpkg-query -W -f '${Status}' tokenstation 2>/dev/null | grep -q 'ok installed'; then
        kind=deb
    elif command -v rpm >/dev/null 2>&1 && rpm -q tokenstation >/dev/null 2>&1; then
        kind=rpm
    elif command -v pacman >/dev/null 2>&1 && pacman -Q tokenstation >/dev/null 2>&1; then
        kind=arch
    fi
fi
if [ -z "$kind" ]; then
    tell "No update: this ts5-tui ($self) isn't from a TokenStation5 package or the portable build, so there is nothing to update it with."
    exit 2
fi
if [ "$(uname -m)" != x86_64 ]; then
    tell "No update: TokenStation5's builds are for x86_64 machines, and this one is $(uname -m)."
    exit 2
fi
case "$kind" in
    deb) file=tokenstation_latest_amd64.deb ;;
    rpm) file=tokenstation-latest.x86_64.rpm ;;
    arch) file=tokenstation-latest-x86_64.pkg.tar.zst ;;
    portable) file=tokenstation-latest-linux-amd64.tar.gz ;;
esac

installed() {
    case "$kind" in
        deb) dpkg-query -W -f '${Version}' tokenstation 2>/dev/null ;;
        rpm) rpm -q --qf '%{VERSION}' tokenstation 2>/dev/null ;;
        arch) pacman -Q tokenstation 2>/dev/null | cut -d' ' -f2 ;;
        portable) "$self" --version 2>/dev/null | sed -e 's/^ts5-tui //' -e 's/^linux-//' ;;
    esac
}
was=$(installed)

sudo=
if [ "$(id -u)" != 0 ] && [ "$kind" != portable ]; then
    if ! command -v sudo >/dev/null 2>&1; then
        tell "No update: installing needs the administrator, and there is no sudo here."
        exit 2
    fi
    sudo=sudo
fi
say "TokenStation5 ${was:-(version unknown)} is installed here ($kind). The newest build:"
say "  $base/$file"
[ -n "${TS5_DRY:-}" ] && exit 0

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    tell "No update: neither curl nor wget is installed, so there is nothing to download with."
    exit 2
fi
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ts5-update.XXXXXX") || exit 2
trap 'rm -rf "$tmp"' EXIT
# The package program reads it as its own user (apt's _apt).
chmod 755 "$tmp"
say "Downloading it:"
if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$tmp/$file" "$base/$file"
else
    wget -O "$tmp/$file" "$base/$file"
fi
if [ $? != 0 ] || [ ! -s "$tmp/$file" ]; then
    tell "No update: the download failed (nothing was changed)."
    exit 3
fi
chmod 644 "$tmp/$file"

# The version in the download: the same one needs no password and no install.
case "$kind" in
    deb) new=$(dpkg-deb -f "$tmp/$file" Version 2>/dev/null) ;;
    rpm) new=$(rpm -qp --qf '%{VERSION}' "$tmp/$file" 2>/dev/null) ;;
    arch) new=$( (bsdtar -xOf "$tmp/$file" .PKGINFO || tar -xOf "$tmp/$file" .PKGINFO) 2>/dev/null | sed -n 's/^pkgver = //p') ;;
    portable)
        if ! tar -xzf "$tmp/$file" -C "$tmp"; then
            tell "No update: the download could not be unpacked (nothing was changed)."
            exit 3
        fi
        inst=$(find "$tmp" -maxdepth 2 -type f -name install.sh | head -n 1)
        bin=$(find "$tmp" -maxdepth 3 -type f -name ts5-tui | head -n 1)
        if [ -z "$inst" ]; then
            tell "No update: the download holds no install.sh (nothing was changed)."
            exit 3
        fi
        new=$([ -n "$bin" ] && "$bin" --version 2>/dev/null | sed -e 's/^ts5-tui //' -e 's/^linux-//')
        ;;
esac
if [ -n "$was" ] && [ "$new" = "$was" ]; then
    tell "Already the newest: $was."
    exit 10
fi

say "Installing ${new:-it}:"
case "$kind" in
    deb) cmd="apt-get install -y $tmp/$file" ;;
    rpm) cmd="dnf install -y $tmp/$file" ;;
    arch) cmd="pacman -U --noconfirm $tmp/$file" ;;
    portable) cmd="$inst --no-start --no-steam" ;;
esac
say "  ${sudo:+sudo }$cmd"
[ -n "$sudo" ] && say "sudo asks for your password next. Ctrl-C leaves, with nothing installed."
$sudo $cmd
rc=$?
if [ $rc != 0 ]; then
    tell "No update: the install did not finish (it ended with $rc). ${was:+$was is still installed.}"
    exit $rc
fi
now=$(installed)
tell "Updated: ${was:-the old build} to ${now:-${new:-the newest build}}."
exit 0
)SH";

}  // namespace

std::string update_cmd(const std::string &self, const std::string &note, bool dry) {
    return std::string(dry ? "TS5_DRY=1 " : "") + "sh -c " + shell_quote(kScript) + " ts5-update " + shell_quote(self) +
           " " + shell_quote(note);
}

}  // namespace tui
